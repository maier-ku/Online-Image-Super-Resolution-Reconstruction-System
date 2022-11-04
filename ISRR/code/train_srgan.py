import torch.backends.cudnn as cudnn
import torch
from torch import nn
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from models import Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
from chosen_model import Generator

# Dataset parameters
data_folder = './data/'  # Data storage path
crop_size = 96  # High resolution image crop size
scaling_factor = 4  # Enlargement ratio

# Generator model parameters (same as SRResNet)
large_kernel_size_g = 9
small_kernel_size_g = 3
n_channels_g = 64
n_blocks_g = 16
srresnet_checkpoint = "./results/4X_SRResNet.pth"  # pre-trained SRResNet model to initialize

# Discriminator model parameters
kernel_size_d = 3  # Kernel size of all convolution modules
n_channels_d = 64  # Number of channels in the first convolution module, doubling the number of channels in every subsequent module
n_blocks_d = 8  # Number of convolution modules
fc_size_d = 1024  # Number of fully connected layer connections

# Learning parameters
batch_size = 128
# batch_size = 400
start_epoch = 1
epochs = 50
checkpoint = None  # SRGAN pre-trained model, if none, fill in None
workers = 4  # Number of data threads loaded
vgg19_i = 5  # VGG19 network i-th pooling layer
vgg19_j = 4  # VGG19 network j-th convolutional layer
beta = 1e-3  # Discriminating loss multipliers
lr = 1e-4  # learning rate

# Equipment parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ngpu = 1  # Number of gpu's used to run
cudnn.benchmark = True  # Acceleration of convolution
writer = SummaryWriter('runs_4X_SRGAN')  # Real-time monitoring     Use the command 'tensorboard --logdir runs'


def main():
    """
    training.
    """
    global checkpoint, start_epoch, writer

    # Model initialisation
    generator = Generator(large_kernel_size=large_kernel_size_g,
                          small_kernel_size=small_kernel_size_g,
                          n_channels=n_channels_g,
                          n_blocks=n_blocks_g,
                          scaling_factor=scaling_factor,
                          model_selection=0)

    discriminator = Discriminator(kernel_size=kernel_size_d,
                                  n_channels=n_channels_d,
                                  n_blocks=n_blocks_d,
                                  fc_size=fc_size_d)

    # Initialize Optimiser
    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr)

    # Truncated VGG19 network for calculating the loss function
    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    # Loss function
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    # Moving data to the default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Loading pre-trained models
    srresnetcheckpoint = torch.load(srresnet_checkpoint)
    generator.net.load_state_dict(srresnetcheckpoint['model'])

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])

    # Customised dataloaders
    train_dataset = SRDataset(data_folder, split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    # Start of round by round training
    for epoch in range(start_epoch, epochs + 1):

        if epoch == int(epochs / 2):  # Reduced learning rate halfway through execution
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        generator.train()  # Turn on training mode: allows the use of batch sample normalisation
        discriminator.train()

        losses_c = AverageMeter()  # Loss of content
        losses_a = AverageMeter()  # Generating losses
        losses_d = AverageMeter()  # Discriminating losses

        n_iter = len(train_loader)
        import time
        start = time.time()

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24),  imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  imagenet-normed

            # -----------------------1. Generator updates----------------------------

            sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), range [-1, 1]
            sr_imgs = convert_image(
                sr_imgs, source='[-1, 1]',
                target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

            # Calculating VGG feature maps
            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)  # batchsize X 512 X 6 X 6
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # batchsize X 512 X 6 X 6

            # Calculating content loss
            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            # Calculating generation losses
            sr_discriminated = discriminator(sr_imgs)  # (batch X 1)   
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated))

            # Calculating the total perceived loss
            perceptual_loss = content_loss + beta * adversarial_loss

            # Backward propagation.
            optimizer_g.zero_grad()
            perceptual_loss.backward()

            # Update generator parameters
            optimizer_g.step()

            # Recording loss values
            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            # -----------------------2. Discriminator update----------------------------
            # Discriminator judgement
            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                               adversarial_loss_criterion(hr_discriminated, torch.ones_like(
                                   hr_discriminated))

            # Backward propagation
            optimizer_d.zero_grad()
            adversarial_loss.backward()

            # Update discriminator
            optimizer_d.step()

            # Recording losses
            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

            # Monitoring image changes
            if i == (n_iter - 2):
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_1',
                                 make_grid(lr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_2',
                                 make_grid(sr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_3',
                                 make_grid(hr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

            # Print results
            print("第 " + str(i) + " 个batch结束")

        # Manual memory release
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated

        # Monitor changes in loss values
        writer.add_scalar('SRGAN/Loss_c', losses_c.val, epoch)
        writer.add_scalar('SRGAN/Loss_a', losses_a.val, epoch)
        writer.add_scalar('SRGAN/Loss_d', losses_d.val, epoch)

        # Save pre-trained models
        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_d': optimizer_d.state_dict(),
        }, 'results_2/4X_SRGAN_1/checkpoint_{}.pth'.format(epoch))

        end = time.time()
        period = end - start
        print('epoch' + str(epoch) + 'costs' + str(period) + 'seconds')

    # Turn off monitoring at the end of training
    writer.close()


if __name__ == '__main__':
    main()
