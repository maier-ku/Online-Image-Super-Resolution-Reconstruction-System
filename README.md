# Image-Super-Resolution-Reconstruction-Online-System

How to train models to implement the system:

Download datasets in different websites, put them under folder ISRR/data.

Run create_data_lists.py to construct JSON files corresponding to the images.

Adjust scaling_factor to 2, 4, 8 in order to construct different level of models. 

Run train_srresnet.py to build SRResNet model.

Run train_srgan.py to build SRGAN model using SRResNet as Generator, also, change model selection parameter to build different models.

Run weighted_model_truncated.py to determine the weights for model ensemble.

The training process costs a long time, so if you don't want to train models manually, please follow this link below and download models, put them under results folder. Remember the 4X_SRGAN model uses 4X_SRGAN_4.pth.

https://drive.google.com/drive/folders/1vVkVSG_BBKcttBeVcGZQEW8NvIYbaSSH?usp=sharing