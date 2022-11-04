import numpy as np

from utils import *
from torch import nn
from chosen_model import SRResNet, Generator
import time
from PIL import Image

# Test images
# imgPath = './data/BSD100/302008.jpg'
imgPath = './results/test.jpg'

# Model parameters
large_kernel_size = 9  # Kernel size for the first and last layers of convolution
small_kernel_size = 3  # Kernel size for middle layer convolution
n_channels = 64  # Number of intermediate level channels
n_blocks = 16  # Number of residual modules
scaling_factor = 4  # enlargement factor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # pre-trained
    start = time.time()
    model_0 = r".\results\model_0.pth"
    model_1 = r".\results\model_1.pth"
    model_2 = r".\results\model_2.pth"
    model_3 = r".\results\model_3.pth"
    model_4 = r".\results\model_4.pth"

    srgan_checkpoints = [model_0, model_1, model_2,
                         model_3, model_4]
    weights = [0.2, 0.1, 0.2, 0.2, 0.3]    # current best weights
    # weights = [0.1, 0.1, 0.2, 0.2, 0.4]
    # weights = [0, 0, 0, 0, 1]
    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
    count = 0
    # Load model SRResNet or SRGAN
    for weight, srgan_checkpoint in zip(weights, srgan_checkpoints):
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

        # Transferring data to the device
        lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

        # Model reasoning
        with torch.no_grad():
            #  detach(): returns a new tensor, detached from the current computed graph, but still pointing to the original variable's storage location, the difference being that require_grad is False and the resulting tensor never needs to compute gradients and does not have grad
            sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            if count == 0:
                final_img = sr_img * weight
            else:
                final_img += sr_img * weight
        count += 1
    final_img = convert_image(final_img, source='[-1, 1]', target='pil')
    print(np.asarray(final_img))
    final_img.save('./results/weighted_srgan.jpg')
    print('Spending time  {:.3f} seconds'.format(time.time() - start))
