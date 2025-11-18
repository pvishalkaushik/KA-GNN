# KA-GNN

conda env create -f environment.yml  
conda activate ka_gnn_cpu  
  
python predict.py --task [task name] --smiles [smiles string]  
  
Currently, allowed task names are: bbbp, clintox, tox21, sider  
