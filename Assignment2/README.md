# Homework 2 – Image Classification with Transfer Learning

Classification of the 37 breeds of the Oxford-IIIT Pet dataset with a pre-trained
MobileNetV3-Small, comparing three transfer-learning strategies:

## Environment

| | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.15.0 (CUDA 13.2, cuDNN 9.26) |
| TorchVision | 0.29.2 |
| scikit-learn | 1.9.1 |
| GPU used | NVIDIA GeForce RTX 4060 Laptop GPU |

All dependencies with exact versions are in `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Oxford-IIIT Pet Dataset — O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar,
*Cats and Dogs*, CVPR 2012. <https://www.robots.ox.ac.uk/~vgg/data/pets/>
License: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0).

The dataset is not included in the repository. It is downloaded automatically by
`torchvision.datasets.OxfordIIITPet` into `./data` the first time `main.py` runs.

**Splits**
- test: the official `test` partition (3669 images), used only for the final evaluation.
- train / validation: the official `trainval` partition (3680 images), split 80/20 with
  `sklearn.model_selection.train_test_split`, stratified by breed, `random_state=1`.

## Reproducing the experiments

```bash
python main.py    # trains and evaluates all three approaches
python plots.py   # results table, confusion matrices, metric plot and prediction examples
```


## Training setup

| | Full fine-tuning | Classifier head | Feature extraction + SVM |
|---|---|---|---|
| Trainable parameters | 1,555,781 (all) | 628,773 (classifier) | 0 in the network, 21,349 in the SVM |
| Loss | cross-entropy | cross-entropy | squared hinge (LinearSVC default) |
| Optimizer | Adam, lr 1e-4 | Adam, lr 1e-3 | – |
| Batch size | 32 | 32 | 32 (feature extraction) |
| Epochs | 10 | 10 | – |
| Model selection | epoch with best validation macro F1 | epoch with best validation macro F1 | C ∈ {1e-4, 1e-3, 1e-2, 1e-1} with best validation macro F1 |
