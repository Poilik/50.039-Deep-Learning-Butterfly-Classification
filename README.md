# 50.039 Deep Learning Butterfly Classification

This project classifies butterfly images into two classes:

- `0_non-hybrid`
- `1_hybrid`

It includes dataset preparation, image resizing, training, hyperparameter tuning, checkpoint saving, best-model selection, and test evaluation across several model families, including CNN, ResNet, AutoEncoder, VGG-16, AlexNet, CoCa, and EfficientNet-based experiments.

## Project Structure

The repository is organized as follows:

- `butterfly_image_csv/`  
	CSV files used to download the raw butterfly images.
- `butterfly_anomaly_image/`  
	Original dataset split into train, validation, and test folders.
- `butterfly_anomaly_image_resized/`  
	Resized dataset used for training and evaluation.
- `corrupted_images/`  
	Lists of corrupted files identified during preprocessing.
- `models/`  
	Model definitions and checkpoint wrapper classes.
- `saved_models/`  
	Saved checkpoint files from training runs.
- `Training_Notebooks/`  
	Jupyter notebooks for training, evaluation, and plotting.
- `utils/`  
	Helper functions for preprocessing, data loading, training, evaluation, plotting, and checkpoint handling.

## Dataset and Saved Models

The `saved_models/` folder is too large to store in GitHub and the eDimension portal, so it is stored separately in OneDrive.

To use the trained checkpoints:

1. Download the folder from this OneDrive link: https://tinyurl.com/yc8yyxbp
2. Move the `saved_models/` folder into the root directory of this project.
3. Unzip all the folders within `saved_models/`

The CSV files used to download the images are stored in `butterfly_image_csv/`.

The butterfly image dataset used in this project was provided by the Imageomics HDR anomaly challenge repository:

https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/files/butterfly_anomaly_train.csv

The resized train, validation, and test images are stored in `butterfly_anomaly_image_resized/`.

## Utilities

The `utils/` folder contains the main reusable functions for the project:

- `data_processing_utils.py`  
	Functions to remove corrupted images, split the dataset into train/validation/test sets, and resize images.
- `dataloader_utils.py`  
	Functions to create DataLoaders from dataset folders or `ImageFolder`-style inputs.
- `display_image_utils.py`  
	Functions to show augmented images for verification.
- `train_val_utils.py`  
	Training loop for classification models using weighted cross-entropy loss, with checkpointing and metric logging.
- `train_val_ae_utils.py`  
	Training loop for autoencoders using reconstruction loss, with checkpointing and metric logging.
- `show_best_model_utils.py`  
	Loads checkpoint history and selects the best epoch based on validation F2 score for class 1.
- `test_utils.py`  
	Evaluates a trained model on the test dataset.
- `load_best_model_utils.py`  
	Loads the best classification model from a checkpoint file and evaluates it on the test set.
- `load_best_model_ae_utils.py`  
	Same as `load_best_model_utils.py`, but for autoencoders using reconstruction loss.
- `plot_train_val_curve_utils.py`  
	Plots training and validation loss, F1, and F2 curves.

## Models

The `models/` folder contains the model definitions used in the project:

- `AutoEncoder/`  
	Autoencoder variants for anomaly detection.
- `CNN/`  
	CNN baseline models.
- `RESNET/`  
	ResNet-based models.
- `state_of_the_art_model/`  
	Wrappers for state-of-the-art models and checkpointed model classes.

## Training Notebooks

The `Training_Notebooks/` folder contains notebooks for:

- hyperparameter tuning
- training individual model variants
- selecting the best checkpoint using validation F2 score
- loading and evaluating the best model on the test set
- plotting training curves

Because the notebooks are not in the project root, they add the repository root to the Python path before importing project modules:

```python
import os
import sys
sys.path.append(os.path.abspath("../.."))
```

Model imports from the notebooks use paths relative to the project root, for example:

```python
from models.state_of_the_art_model.RESNET_18_CKPT import CheckpointedModel
```

## Typical Workflow

1. Load and preprocess the dataset.
2. Train a model with a chosen hyperparameter setting.
3. Save checkpoint history during training.
4. Use `show_best_model_utils.py` to identify the best epoch.
5. Load the best checkpoint and evaluate on the test set.
6. Plot training and validation curves.

## Notes

- Checkpoints store model weights, optimizer state, and metric history for each epoch.
- Best model selection is based on validation F2 score for class 1.
- Autoencoder models use reconstruction error and thresholding to identify class 1 anomalies.

## Acknowledgments

The butterfly image dataset used in this project was provided by the Imageomics HDR anomaly challenge repository:

https://github.com/Imageomics/HDR-anomaly-challenge/blob/main/files/butterfly_anomaly_train.csv

## Setup Reminder

If you open a notebook from `Training_Notebooks/`, make sure the project root is added to `sys.path` before importing local modules:

```python
import os
import sys
sys.path.append(os.path.abspath("../.."))
```
