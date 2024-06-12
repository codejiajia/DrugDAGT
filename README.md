# DrugDAGT
A dual-attention graph transformer framework with contrastive learning for 86 drug-drug interaction types prediction.<br/>
![image](https://github.com/codejiajia/DrugDAGT/blob/main/model.jpg)
<br/>
Please see our manuscript for more details.<br/>
## Requirements
* python>=3.6<br/>
* cuda >= 8.0 + cuDNN<br/>
* pytorch>=1.4.0<br/>
* numpy>=1.18.1<br/>
* pandas>=1.0.3<br/>
* flask>=1.1.2<br/>
* gunicorn>=20.0.4<br/>
* hyperopt>=0.2.3<br/>
* matplotlib>=3.1.3<br/>
* pandas-flavor>=0.2.0>=3.1.3<br/>
* rdkit>=2020.03.1.0<br/>
* scipy>=1.4.1<br/>
* tensorboardX>=2.0<br/>
* torchvision>=0.5.0<br/>
* tqdm>=4.45.0<br/>
* einops>=0.3.2<br/>
* seaborn>=0.11.1
## Data
* `drugbank_drugs.csv`:The drugbank IDs corresponding to the 1706 drugs in our dataset.<br/>
* `drugbank_smiles.csv`:The SMILES sequences corresponding to 1706 drugs.<br/>
* `{train/val/test}_pair_{left/right}.csv`:The left or right drugs of drug pairs in train/validation/test sets in warm-start scenario.<br/>
* `{train/val/test}_pair_{left/right}_cold2.csv`:The left or right drugs of drug pairs in train/validation/test sets in cold-start scenario.
## Usage
Here, we provide a demo for DDI prediction in the warm-start scenario <br/>
* Run train.py using 
  ```
  >> python train.py --data_path <path1> --data_path_right <path2> --index_path <path3> --smiles_path <path4> --separate_val_path <path5> --separate_val_path_right <path6> --separate_test_path <path7> --separate_test_path_right <path8> --save_dir <path9> --bond_fast_attention --atom_attention --adjacency --adjacency_path <path10> --distance --distance_path <path11> --coulomb --coulomb_path <path12> --normalize_matrices --features_path <path13> --no_features_scaling --gpu <gpu> --batch_size <batch_size> --epoch <epoch> 
  ```
Explanation of parameters
* `data_path`Path to train_pair_left.csv file
* `data_path_right`Path to train_pair_right.csv file
* `index_path`Path to drugbank_drugs.csv file
* `smiles_path` Path to drugbank_smiles.csv file
* `separate_val_path` Path to val_pair_left.csv file
* `separate_val_path_right` Path to val_pair_right.csv file
* `separate_test_path` Path to test_pair_left.csv file
* `separate_test_path_right` Path to test_pair_right.csv file
* `save_dir`Directory where model checkpoints will be saved
* `bond_fast_attention` Fast attention in message passing phase
* `atom_attention` Self attention in readout phase
* `adjacency` Whether to add adjacency feature matrix to the attention weight
* `adjacency_path` Path to adj.npz file
* `distance` Whether to add distance matrix to the attention weight
* `distance_path` Path to dist.npz file
* `coulomb` Whether to add coulomb matrix to the attention weight
* `coulomb_path` Path to clb.npz file
* `normalize_matrices` Whether to perform apply softmax norm on generated atomic matrices
* `features_path` Path to rdkit_norm.npz file
* `no_features_scaling` Turn off scaling of features
* `gpu` Which GPU to use
* `batch_size` Batch size-{200}
* `epoch` Epoch-{20}
