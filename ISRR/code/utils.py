
from PIL import Image
import os
import json
import random
import torchvision.transforms.functional as FT
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 常量
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456,
                                   0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224,
                                  0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor(
    [0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor(
    [0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def create_data_lists(train_folders, test_folders, min_size, output_folder):
    """
    Create training set and test set list files.
        train_folders: Collection of training folders; the images in each folder will be combined into a single image list file
        test_folders: A collection of test folders; each folder will form a picture list file
        min_size: Minimum values for image width and height
        output_folder: The final list of generated files, in json format
    """
    print("\nDocument list being created... Please be patient.\n")
    train_images = list()
    for d in train_folders:
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                train_images.append(img_path)
    print("There are %d images in the training set\n" % len(train_images))
    with open(os.path.join(output_folder, 'train_images.json'), 'w') as j:
        json.dump(train_images, j)

    for d in test_folders:
        test_images = list()
        test_name = d.split("/")[-1]
        for i in os.listdir(d):
            img_path = os.path.join(d, i)
            img = Image.open(img_path, mode='r')
            if img.width >= min_size and img.height >= min_size:
                test_images.append(img_path)
        print("There are %d images in the test set %s\n" %
              (test_name, len(test_images)))
        with open(os.path.join(output_folder, test_name + '_test_images.json'), 'w') as j:
            json.dump(test_images, j)

    print("Generation is complete. The list of training set and test set files is saved under %s \n" % output_folder)


def convert_image(img, source, target):
    """
    Convert the image format.

    :img: input image
    :source: The data source format, there are 3 types
                   (1) 'pil' (PIL Image)
                   (2) '[0, 1]'
                   (3) '[-1, 1]' 
    :target: data target format, 5 types
                   (1) 'pil' (PIL Image)
                   (2) '[0, 1]' 
                   (3) '[-1, 1]' 
                   (4) 'imagenet-norm'  (normalised by the mean and variance of the imagenet dataset)
                   (5) 'y-channel' (For calculating PSNR and SSIM)
    :return: Converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'
                      }, "Unable to convert image source format %s!" % source
    assert target in {
        'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm', 'y-channel'
    }, "Unable to convert image source format %s!" % target

    # 转换图像数据至 [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)  # Image to a Tensor of shape [C,H,W] with values in the range [0,1.0].

    elif source == '[0, 1]':
        pass

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target format
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :],
                           rgb_weights) / 255. + 16.

    return img


class ImageTransforms(object):
    """
    Image transformation.
    """

    def __init__(self, split, crop_size, scaling_factor, lr_img_type,
                 hr_img_type):
        """
        :split: 'train' or 'test'
        :crop_size: High resolution image crop size
        :scaling_factor: Magnification ratio
        :lr_img_type: Low resolution image pre-processing methods
        :hr_img_type: High resolution image pre-processing methods
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        Cropping and downsampling of the image to form a low resolution image
        :img: image read by the PIL library
        :return: low and high resolution images of a particular form
        """

        # 裁剪
        if self.split == 'train':
            # Cropping a random sub-block from the original image as a high resolution image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Cropping the largest possible image from the image that is divisible by the magnification scale
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsampling
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor),
                                int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)

        assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)

        return lr_img, hr_img


class AverageMeter(object):
    """
    Track record class for averaging, summing and counting a set of data.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Discard the gradient to prevent it from exploding during the calculation.

    :optimizer: gradient will be truncated
    :grad_clip: truncated value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save training results.
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    adjust learning rate.

    :optimizer: Optimisers to be adjusted
    :shrink_factor: Adjustment factor, in the range (0, 1), to be multiplied by the original learning rate.
    """

    print("\nAdjust learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("New learning rate %f\n" % (optimizer.param_groups[0]['lr'],))
