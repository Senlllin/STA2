# STA

This repository implements the **Symmetry-parameter Transformer Attention** (STA) model for point cloud completion.

## Requirements
- CUDA 10.2 – 11.1
- Python 3.7
- PyTorch 1.8 – 1.9
- numpy, lmdb, msgpack-numpy, ninja, termcolor, tqdm, open3d, h5py

## Build PointNet++ Ops
```bash
cd util/pointnet2_ops_lib
python setup.py install
```

## Code Structure
- `network/sta.py` – core generator and discriminator with STA head and losses.
- `network/sta_pcn.py` – PCN variant of the STA model.
- `network/sta_classifier.py` – classifier-augmented version.
- `network/train.py` – training entry script.
- `network/test.py` – inference script.
- `util/pointnet2_model_api.py` – encoder/decoder modules.
- `util/loss_util.py` – Chamfer and symmetry losses.

## Training
Example command:
```bash
python network/train.py \
    --lmdb_train data/RealComData/realcom_data_train.lmdb \
    --lmdb_valid data/RealComData/realcom_data_test.lmdb \
    --lmdb_sn    data/RealComShapeNetData/shapenet_data.lmdb \
    --class_name all \
    --batch_size 4
```

Useful flags:
```
--use_sta          # enable STA head and losses
--use_sta_bias     # enable STA attention bias
--sta_k_mirror K   # number of mirrored neighbors
--sta_beta B       # bias magnitude
--sta_temperature T
```

## Evaluation
```bash
python network/test.py \
    --lmdb_valid data/RealComData/realcom_data_test.lmdb \
    --lmdb_sn    data/RealComShapeNetData/shapenet_data.lmdb \
    --class_name all \
    --log_dir weights/STA \
    --last_epoch 120
```

## Dataset Layout
```
data/
  RealComData/
    realcom_data_train.lmdb
    realcom_data_test.lmdb
  RealComShapeNetData/
    shapenet_data.lmdb
```

The dataset paths can be placed anywhere; pass them via command-line arguments.
