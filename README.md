# News Model Evaluation and Dataset Cleaning

This project contains a few models I experimented with, each having different performances. The final model that I recommend you try is `saved_my_trained_model50.pth`. 

## Model Evaluation

To evaluate the trained model, use the following command:

```bash
python newsTestModel.py --input cleaned_articles.txt --train saved_my_trained_model50.pth --evaluate --batch-size 64 --context-size 128 --n-embd 128 --n-head 4 --n-layer 4 --dropout 0.1

# Files in the Project

### 1. `newsTestModel.py`
- **Description:** Python script to test and evaluate the trained model.
- **Usage:** This file is used for evaluating the performance of the trained model on unseen data. It will load the model and perform inference on a given test dataset.

### 2. `cleanDataset.ipynb`
- **Description:** Jupyter notebook for cleaning the dataset.
- **Usage:** This notebook is used to preprocess and clean the raw dataset. It will handle tasks like removing unwanted characters, normalizing text, and splitting the dataset for training and validation.

### 3. `saved_my_trained_model50.pth`
- **Description:** Final model to be used for evaluation.
- **Usage:** This file contains the trained model that you can load for testing or further evaluation. The model is saved in PyTorch's `.pth` format.

