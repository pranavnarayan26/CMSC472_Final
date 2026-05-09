This directory contains the code for Pranav Narayan's final project for CMSC472, focused on the application of deep-learning methodologies to the spacecraft anomaly detection problem. 

To run the algorithms, select the train_{model_name}() function desired from the bottom of models.py, and watch the terminal for outputs. 

The code for this project is split into three main files: 

    **models.py**: all learning models, including the Transformer-Autoencoder, standard MLP, CNN, and TCN. This file also includes the custon BCE loss modification that accounts specifically for F0.5 score rather than simple BCE, and the adaptive thresholding approach appended to the Transformer-Autoencoder. 

    **data_processing.py**: all functions used to preprocess the ESA dataset. This includes the position encoding for the dataset, as that is not explicitly given, the dataset itself, and the file that extracts the data  from the parquet file given by ESA. 

    **main.py**: this is the actual executable script used for testing these models. It includes functions for a k-fold validation loop, an inference loop, and a downsampling function to ensure anomalies in the dataset while saving some time. Each network is trained independently. 

In addition, this directory contains the following files from the ESA dataset:

    **train.parquet**: the dataset to be used in training, with labeled timesteps and the full number of channels. 
    **test.parquet**: the dataset on which the final model is tested, without labels. 
    **sample_submission.parquet**: self-explanatory. 
    **target_channels.csv**: the channels that are actually relevant for the detection task, as defined by ESA. 

These files can also be found on the ESA dataset website on Kaggle, visible here: https://www.kaggle.com/competitions/esa-adb-challenge/overview. 

I would additionally like to note that while the TCN is not described in the final project (for lack of time), it is included here for completeness (and because I wrote the code, it doesn't need to go to waste!). 