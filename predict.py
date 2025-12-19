import argparse
import torch
from model.ka_gnn import KA_GNN, KA_GNN_two
from model.mlp_sage import MLPGNN, MLPGNN_two
from model.kan_sage import KANGNN, KANGNN_two
from utils.graph_path import path_complex_mol
from rdkit import Chem
import yaml
import dgl

# Add property names mapping
TASK_LABELS = {
    "tox21": ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
              'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'],
    "sider": ['Hepatobiliary disorders','Metabolism and nutrition disorders','Product issues',
              'Eye disorders','Investigations','Musculoskeletal and connective tissue disorders',
              'Gastrointestinal disorders','Social circumstances','Immune system disorders',
              'Reproductive system and breast disorders','Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
              'General disorders and administration site conditions','Endocrine disorders','Surgical and medical procedures',
              'Vascular disorders','Blood and lymphatic system disorders','Skin and subcutaneous tissue disorders',
              'Congenital, familial and genetic disorders','Infections and infestations','Respiratory, thoracic and mediastinal disorders',
              'Psychiatric disorders','Renal and urinary disorders','Pregnancy, puerperium and perinatal conditions',
              'Ear and labyrinth disorders','Cardiac disorders','Nervous system disorders','Injury, poisoning and procedural complications'],
    "clintox": ['FDA_APPROVED', 'CT_TOX'],
    "bbbp": ['label']
}

def update_node_features(g):
    def message_func(edges):
        return {'feat': edges.data['feat']}

    def reduce_func(nodes):
        num_edges = nodes.mailbox['feat'].size(1)  
        agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges  
        return {'agg_feats': agg_feats}

    g.send_and_recv(g.edges(), message_func, reduce_func)
    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)
    return g

def get_target_dim(task_name):
    target_map = {
        'tox21':12, 'muv':17, 'sider':27, 'clintox':2, 'bace':1, 'bbbp':1, 'hiv':1
    }
    if task_name not in target_map:
        raise ValueError(f"Unknown task name: {task_name}")
    return target_map[task_name]

TASK_CONFIG = {
    "tox21":   {"model_select": "ka_gnn_two",        "loss": "bce"},
    "sider":   {"model_select": "mlp_sage_two",    "loss": "bce"},
    "clintox": {"model_select": "ka_gnn_two",  "loss": "bce"},
    "bbbp":    {"model_select": "ka_gnn_two",        "loss": "bce"}
}

def main():
    parser = argparse.ArgumentParser(description="Test model on single SMILES")
    parser.add_argument("--task", type=str, required=True, help="Task name, e.g. clintox")
    parser.add_argument("--smiles", type=str, required=True, help="Input SMILES string")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Optional: custom model path. If not provided, use ./models/{task}.pth")
    parser.add_argument("--config", type=str, default="./config/c_path.yaml", help="Config yaml file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (ignore GPU)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task = args.task.lower()
    smiles = args.smiles.strip()

    # Override model_select based on task
    TASK_MODEL = {
        "tox21":   "ka_gnn_two",
        "sider":   "mlp_sage_two",
        "clintox": "ka_gnn_two",
        "bbbp":    "ka_gnn_two"
    }

    if task not in TASK_MODEL:
        raise ValueError(f"Unknown task: {task}")

    model_select = TASK_MODEL[task]   # <-- override from dictionary

    # YAML config still used for everything else:
    encoder_atom = config['encoder_atom']
    encoder_bond = config['encoder_bond']
    pooling      = config['pooling']
    grid_feat    = config['grid_feat']
    num_layers   = config['num_layers']

    # Auto model path if none provided
    if args.model_path is None:
        args.model_path = f"./models/{task}.pth"

    print(f"Loading model from: {args.model_path}")

    encode_dim = [0, 0]
    encode_dim[0] = 92  # cgcnn atom encoding
    encode_dim[1] = 21  # bond encoding dim_14

    # Build graph from SMILES
    graph = path_complex_mol(smiles, encoder_atom, encoder_bond)
    if graph is False:
        print("Failed to generate graph from SMILES")
        return

    graph = update_node_features(graph).to(device)
    node_features = graph.ndata['feat'].to(device)

    # Instantiate model
    if model_select == 'ka_gnn':
        model = KA_GNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=get_target_dim(task), 
                       grid_feat=grid_feat, num_layers=num_layers, pooling=pooling, use_bias=True)
    elif model_select == 'ka_gnn_two':
        model = KA_GNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=get_target_dim(task), 
                           grid_feat=grid_feat, num_layers=num_layers, pooling=pooling, use_bias=True)
    elif model_select == 'mlp_sage':
        model = MLPGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=get_target_dim(task), 
                       grid_feat=grid_feat, num_layers=num_layers, pooling=pooling, use_bias=True)
    elif model_select == 'mlp_sage_two':
        model = MLPGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=get_target_dim(task), 
                           grid_feat=grid_feat, num_layers=num_layers, pooling=pooling, use_bias=True)
    elif model_select == 'kan_sage':
        model = KANGNN(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=get_target_dim(task), 
                       grid_feat=grid_feat, num_layers=num_layers, pooling=pooling, use_bias=True)
    elif model_select == 'kan_sage_two':
        model = KANGNN_two(in_feat=encode_dim[0]+encode_dim[1], hidden_feat=64, out_feat=32, out=get_target_dim(task), 
                           grid_feat=grid_feat, num_layers=num_layers, pooling=pooling, use_bias=True)
    else:
        raise ValueError(f"Unknown model_select: {model_select}")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(graph, node_features)
        output = output.cpu().numpy().flatten()  # flatten in case of multi-output

    # Print predictions with property names
    names = TASK_LABELS.get(task, [f"Prop_{i}" for i in range(len(output))])
    print(f"Prediction for SMILES '{smiles}':")
    for name, val in zip(names, output):
        print(f"{name:30s}: {val:.4f}")

if __name__ == "__main__":
    main()
