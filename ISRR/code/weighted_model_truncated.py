
import numpy as np
import torch
from tqdm import tqdm

from utils import *
from torch import nn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from datasets import SRDataset
from models import SRResNet, TruncatedVGG19
from chosen_model import Generator
import time
import imquality.brisque as brisque
import PIL.Image
import matplotlib.pyplot as plt


large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
scaling_factor = 4
ngpu = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg19_i = 5
vgg19_j = 4

if __name__ == '__main__':

    # Testset catalogue
    data_folder = "./data/"
    test_data_names = ["Set5", "Set14", "BSD100"]

    # Pre-training models
    model_0 = r".results\model_0.pth"
    model_1 = r".results\model_1.pth"
    model_2 = r".results\model_2.pth"
    model_3 = r".results\model_3.pth"
    model_4 = r".results\model_4.pth"
    srgan_checkpoints = [model_0, model_1, model_2,
                         model_3, model_4]
    sr_imgs_list_all = []
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19 = truncated_vgg19.to(device)
    count = 0

    for srgan_checkpoint in srgan_checkpoints:
        checkpoint = torch.load(srgan_checkpoint)
        generator = Generator(large_kernel_size=large_kernel_size,
                              small_kernel_size=small_kernel_size,
                              n_channels=n_channels,
                              n_blocks=n_blocks,
                              scaling_factor=scaling_factor,
                              model_selection=count)

        generator = generator.to(device)
        generator.load_state_dict(checkpoint['generator'])

        generator.eval()
        model = generator

        sr_imgs_list = []
        hr_imgs_list = []
        for test_data_name in test_data_names:


            test_dataset = SRDataset(data_folder,
                                     split='test',
                                     crop_size=0,
                                     scaling_factor=4,
                                     lr_img_type='imagenet-norm',
                                     hr_img_type='imagenet-norm',
                                     test_data_name=test_data_name)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                      pin_memory=True)

            with torch.no_grad():

                for i, (lr_imgs, hr_imgs) in enumerate(test_loader):

                    lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                    hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), imagenet-normed

                    sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]
                    sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm')
                    sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
                    hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

                    sr_imgs_list.append(sr_imgs_in_vgg_space)
                    hr_imgs_list.append(hr_imgs_in_vgg_space)
        count += 1
        sr_imgs_list_all.append(sr_imgs_list)

    def mse_func(weights):
        final_prediction = []
        count = 0
        for weight, each_imgs in zip(weights, sr_imgs_list_all):
            # print('each_imgs:', len(each_imgs))
            if count == 0:
                for i in range(len(each_imgs)):
                    final_prediction.append(weight * each_imgs[i])
            else:
                for j in range(len(each_imgs)):
                    # print(final_prediction[j].shape)
                    # print(np.asarray(each_imgs[j]).shape)
                    final_prediction[j] += weight * each_imgs[j]
            count += 1
        avg_score = 0
        criterion = nn.MSELoss(reduction='mean')
        for i in range(len(final_prediction)):
            inputs = final_prediction[i]
            targets = hr_imgs_list[i]
            loss = criterion(inputs, targets)
            avg_score += loss / len(final_prediction)
        return avg_score


    step = 0.01
    min_score = 10000
    best_weights = [0, 0, 0, 0, 1]
    count = 0
    for weight_1 in np.arange(0, 1.2, step):
        for weight_2 in np.arange(0, 1.2, step):
            for weight_3 in np.arange(0, 1.2, step):
                for weight_4 in np.arange(0, 1.2, step):
                    for weight_5 in np.arange(0, 1.2, step):
                        if weight_1 + weight_2 + weight_3 + weight_4 + weight_5 == 1:
                            weights = [weight_1, weight_2, weight_3, weight_4, weight_5]
                            current_score = mse_func(weights)
                            print('current weights:', weights)
                            print('current score:', current_score)
                            print('*' * 100)
                            if current_score < min_score:
                                min_score = mse_func(weights)
                                best_weights = weights
                            print('best weights：', best_weights)
                            print('min_socre：', min_score)

