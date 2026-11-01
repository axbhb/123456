# Asymmetry Vulnerability and Physical Attacks on Online Map Construction for Autonomous Driving

This repository provides the official implementation of the paper: *Asymmetry Vulnerability and Physical Attacks on Online Map Construction for Autonomous Driving* (CCS 2025).

> 📦 **Note:** A directly runnable version with pre-configured environment is available at [Zenodo](https://zenodo.org/records/17116870).

Our attack framework consists of two main components:

- **Asymmetry scene identification**.
- **Four types of physical attacks** on the online map construction model `MapTR`, combining two attack vectors and two objectives:
  - Attack vectors:
    1. *Camera blinding*: simulates lens flare effects using a roadside flashlight.
    2. *Adversarial patch*: simulates an adversarial patch placed on the roadside.
  - Attack objectives:
    1. *Road Straightening Attack (RSA)*: misleads the model into omitting diverging paths, making turns unreachable.
    2. *Early Turn Attack (ETA)*: causes the predicted road boundary to turn earlier, potentially leading to collisions with actual road edges.

## 🛠️ Installation

### Prerequisites

- **Hardware**: NVIDIA GPU with 10GB+ VRAM, 32GB+ RAM, >500GB+ disk space (include nuscenes dataset and attack results)
- **Software**: Python 3.8 + CUDA 11.1 + conda/miniconda
- **Our Setup**:
  - GPU: 10GB VRAM (RTX 3080)
  - RAM: 64GB
  - Ubuntu 22.04
  - CUDA: 11.4

### Installation

```bash
# create conda environment
conda create -n maptr python=3.8 -y
conda activate maptr

# pytorch
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# gcc>=5 in conda env (optional)
conda install -c omgarcia gcc-5 # gcc-5.3.1

# mmlab toolkits
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1

# others
pip install timm
pip install -r requirements.txt  # necessary for mmdetection3d installation

cd mmdetection3d/
rm -rf build/ mmdet3d.egg-info/
python setup.py develop
cd ..

cd projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install
cd ../../../../../..
```

## 📊 Dataset Preparation

### Download Required Data

#### 1. [nuScenes Dataset](https://www.nuscenes.org/download)

```bash
# Two options:
# Option 1: Download and extract nuScenes dataset
# - Full dataset (v1.0)
# - Map expansion pack (v1.3)
# - Extract to data/nuscenes/

# Option 2. Directly use our mocked nuscenes (lightweight) at data/nuscenes
# - can be download from https://drive.google.com/file/d/1rweKfqa1zKd7VQ2Q4HB3sVBlz39IrJrB/view?usp=sharing
# - contains the data for our 100 identified asymmetric scenes.

# For both options, you should also download the can_bus data from nuScenes. 
```

#### 2. Model Checkpoints

```bash
# Create checkpoints directory
mkdir -p ckpts

# Download backbone models
cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth

# Download pre-trained model provided by MapTR (MapTR-tiny, R50 Backbone, 24e Epochs) from: https://drive.google.com/file/d/16PK9XohV55_3qPVDtpXIl4_Iumw9EnfA/view
# Save as: maptr_tiny_r50_24e_bevpool.pth

# Note that we choose maptr_tiny_r50_24e_bevpool cause it provides the best performance among same backbone & epoch setting.
```

#### 3. Generate annotation files

*We genetate custom annotation files following MapTR*

```
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0 --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

#### Expected Directory Structure

```
mapattack_artifact/
├── data/
│   ├── can_bus/
│   ├── lanelet2_for_nuScenes/
│   └── nuscenes/
│       ├── maps/                    # maps_v1.3
│       ├── samples/
│       │   ├── CAM_FRONT/
│       │   ├── CAM_BACK/
│       │   ├── CAM_FRONT_LEFT/
│       │   ├── CAM_FRONT_RIGHT/
│       │   ├── CAM_BACK_LEFT/
│       │   ├── CAM_BACK_RIGHT/
│       ├── v1.0-trainval/           # ~3GB after extraction
│       └── nuScenes_infos_temporal_val.pkl
│       └── nuscenes_map_anns_val.json
├── ckpts/
    ├── resnet50-19c8e357.pth
    ├── maptr_tiny_r50_24e_bevpool.pth
    └── ...
```

## 🔬 Reproducing Paper Results

### Step 1: Asymmetry scene identification

- Please follow the instructions in [dataset_processing/README.md](dataset_processing/README.md). 
- Or you can directly [download](https://drive.google.com/file/d/1g6Ob-8cTARO2LU-oMO1E1m-eeVVQGovr/view?usp=sharing) and use our processed dataset at [dataset/](dataset/) (recommended).

### Step 2: Physical attacks

#### Individual Experiments (optional 1)

Run specific attack types:

```bash
# Road Straightening Attack (RSA) using camera blinding (~6 hours)  
bash run_rsa_blind.sh

# Early Turn Attack (ETA) using camera blinding (2~3 hours)
bash run_eta_blind.sh 

# Road Straightening Attack (RSA) using adversarial patch (2~3 hours)
bash run_rsa_patch.sh

# Early Turn Attack (ETA) using adversarial patch (2~3 hours)
bash run_eta_patch.sh

# Show results
python print_attack_results.py
```

#### All Experiments (optional 2)

```bash
# Run all experiments and show results (~16 hours)
bash run_all_experiments.sh
```

## 📈 Expected Results

> ⚠️ **Note:** The reproduced results may differ slightly from the results reported in the paper due to inherent randomness in the experiments.

### Road Straightening Attack (RSA) Results

**Map AP (%) on asymmetric scenes under road straightening attacks**

| Method     |             | Blinding (Black-box) |        |      |             | Adv Patch (White-box) |        |      |     |
| ---------- | ----------- | -------------------- | ------ | ---- | ----------- | --------------------- | ------ | ---- | --- |
|            | AP_boundary | AP_divider           | AP_ped | mAP  | AP_boundary | AP_divider            | AP_ped | mAP  |     |
| Clean      | 48.9        | 54.2                 | 38.2   | 47.1 | 48.9        | 54.2                  | 38.2   | 47.1 |     |
| RSA (Ours) | 39.9        | 41.4                 | 36.4   | 40.2 | 39.0        | 49.0                  | 37.6   | 41.9 |     |

**Unreachable Goal Rate (%) on asymmetric scenes under road straightening attacks**

| Method     | Blinding (Black-box) | Adv Patch (White-box) |
| ---------- | -------------------- | --------------------- |
| Clean      | 27                   | 27                    |
| RSA (Ours) | 44 (+17)             | 44 (+17)              |

### Early Turn Attack (ETA) Results

**Map AP (%) on asymmetric scenes under early turn attacks**

| Method     |             | Blinding (Black-box) |        |      |             | Adv Patch (White-box) |        |      |     |
| ---------- | ----------- | -------------------- | ------ | ---- | ----------- | --------------------- | ------ | ---- | --- |
|            | AP_boundary | AP_divider           | AP_ped | mAP  | AP_boundary | AP_divider            | AP_ped | mAP  |     |
| Clean      | 48.9        | 54.2                 | 38.2   | 47.1 | 48.9        | 54.2                  | 38.2   | 47.1 |     |
| ETA (Ours) | 46.2        | 52.5                 | 34.5   | 44.4 | 44.2        | 51.1                  | 38.3   | 44.5 |     |

**Unsafe Planned Trajectory Rate (%) on asymmetric scenes under early turn attacks**

| Method     | Blinding (Black-box) | Adv Patch (White-box) |
| ---------- | -------------------- | --------------------- |
| Clean      | 10                   | 10                    |
| ETA (Ours) | 27 (+17)             | 21 (+11)              |

### Output Structure

```
dataset/maptr-bevpool/
├── train_blind_rsa_asymmetric/
│   ├── cams/                   # Adversarial image input visualization
│   ├── vis_seg/                # Map results visualization
│   ├── results/
│   │   ├── map/                # Map evaluation metrics
│   │   └── planning/           # Planning evaluation metrics
└── train_patch_rsa_asymmetric/
    └── ... (similar structure)
```

## 🚨 Troubleshooting

### Common Issues

- We have observed that installing mmdetection3d, a dependency of this artifact, may fail on A4000 GPUs. For other mmdetection3d installation issues, please refer to troubleshooting in [MapTR's official repository](https://github.com/hustvl/MapTR/) or [mmdetection3d's official repository](https://github.com/open-mmlab/mmdetection3d).

## 📜 Citation

```bibtex
@inproceedings{lou2025asymmetry,
  title     = {Asymmetry Vulnerability and Physical Attacks on Online Map Construction for Autonomous Driving},
  author    = {Lou, Yang and Hu, Haibo and Song, Qun and Xu, Qian and Zhu, Yi and Tan, Rui and Lee, Wei-Bin and Wang, Jianping},
  booktitle = {Proceedings of the 32nd ACM Conference on Computer and Communications Security (CCS)},
  year      = {2025},
}
```


## 🙏 Acknowledgements

Our codebase is built upon [MapTR](https://github.com/hustvl/MapTR). We thank the authors of MapTR for open-sourcing their work.