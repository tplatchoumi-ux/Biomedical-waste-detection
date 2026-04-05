# Biomedical-waste-detection
This repository provides code, a representative dataset subset, and full reproducibility resources (including configuration files and environment setup) to enable replication of the proposed biomedical waste classification framework.

## Dataset
A representative anonymized subset of the dataset is included in this repository under:

data/
 ├── images/
 └── annotations/
     └── annotations.json

The dataset contains bounding-box annotations in COCO format.

### Classes
- Sharps  
- Infectious  
- Pharmaceutical  
- Pathological  

The dataset is partially derived from:
https://www.kaggle.com/datasets/engineeringubu/pharmaceutical-and-biomedical-waste

The final dataset includes additional preprocessing, filtering, and annotation performed in this study.

## Reproducibility

### Step 1: Clone repository
git clone https://github.com/tplatchoumi-ux/Biomedical-waste-detection

### Step 2: Install dependencies
pip install -r requirements.txt

### Step 3: Train model
python src/train.py

### Step 4: Run inference
python src/inference.py

### Notes
- Fixed random seed ensures reproducibility
- Data leakage is prevented via proper dataset splitting
- Augmentation is applied only to training data

## Docker

Build:
docker build -t biomedical-waste -f docker/Dockerfile .

Run:
docker run biomedical-waste

## Data Access
A representative dataset subset is publicly available in this repository.

The complete dataset cannot be publicly released due to privacy and regulatory constraints.

However, access can be obtained upon reasonable request from the corresponding author, subject to a Data Use Agreement (DUA).

## Additional Information
Detailed preprocessing, augmentation, training, and inference procedures are described in the manuscript and supplementary material.
