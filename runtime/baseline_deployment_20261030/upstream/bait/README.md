# When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining

Code for ICLR 2026 Paper ["When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining"](https://openreview.net/forum?id=ssWi0rC3mx) by Zhihao Li, Gezheng Xu, Jiale Cai, Ruiyi Fang, Di Wu, Qicheng Lao, Charles Ling, Boyu Wang.

## Abstract
Unlearnable Examples (UEs) serve as a data protection strategy that generates imperceptible perturbations to mislead models into learning spurious correlations instead of underlying semantics. In this paper, we uncover a fundamental vulnerability of UEs that emerges when learning starts from a pretrained model. Specifically, our empirical analysis shows that even when data are protected by carefully crafted perturbations, pretraining priors still allow the model to circumvent the shortcuts introduced by UEs and capture genuine representations, thereby nullifying unlearnability. To address this, we propose BAIT (Binding Artificial perturbations to Incorrect Targets), a novel bi‑level optimization formulation. Specifically, the inner level aims at associating the perturbed samples with real labels to simulate standard data-label alignment, while the outer level actively disrupts this alignment by enforcing a mislabel-perturbation binding that maps samples to designated incorrect targets. This mechanism effectively overrides the semantic guidance of priors, forcing the model to rely on the injected perturbations and consequently preventing the acquisition of true semantics. Extensive experiments on standard benchmarks and multiple pretrained backbones demonstrate that BAIT effectively mitigates the influence of pretraining priors and maintains data unlearnability. 

## Requirements
- Set up a conda environment and install some prerequisite packages.
```
conda create -n your_env_name python=3.13
conda activate your_env_name
pip install -r requirements.txt
```

## Running Experiments
We provide the generator checkpoints of BAIT trained on CIFAR-10, CIFAR-100, and SVHN on [Google Drive](https://drive.google.com/drive/folders/1qHDSl2hP9XzHL4kOGJ5nGi5ZpVanJYEi?usp=sharing). You can also train your own model using the following scripts.

### Training
- Train BAIT on CIFAR-10:
```
bash train_cifar10.sh
```

### Evaluation
- Evaluate BAIT on CIFAR-10:
```
bash test_cifar10.sh
```

## Acknowledgement
This codebase is partially based on [EMN](https://github.com/HanxunH/Unlearnable-Examples).


## Citation
If you find our paper/code useful, please cite the following paper:
```
@inproceedings{
li2026when,
title={When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining},
author={Zhihao Li and Gezheng Xu and Jiale Cai and Ruiyi Fang and Di Wu and Qicheng Lao and Charles Ling and Boyu Wang},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}
``` 


