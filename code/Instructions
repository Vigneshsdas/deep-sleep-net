## FOLDERS 

###  01_data_gen
Contains the notebook to generate CSV files from the raw PSG and Hypnograms using the MNE Python library

### 02_preprocessing
Contains the .py file to preprocess the raw data on a Google Cloud Dataproc cluster.
Needs three arguments to be passed:
1. Path to the data file in the Google Cloud bucket. 2. Path to the labels file in the bucket
3. Path where the preprocessed data is to be saved

### 03_models
Contains 2 files – model.py to be used for single machine training on Google Colab and model_elephas.py to be used for distributed training using Elephas on Google Cloud Platform.

### 04_training
Contains 2 files – train_colab.ipynb to be run on Google Colab for single machine training and train_gcp_elephas.py to be used in a Google Cloud Platform Dataproc cluster (TensorFlow and Elephas must be installed) for distributed training.
Training on the cloud needs one argument to be passed – location of the preprocessed file.
The directories in the ipynb files must be changed as per user requirements.
