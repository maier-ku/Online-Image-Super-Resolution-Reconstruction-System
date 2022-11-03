import re
from urllib.parse import uses_relative
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages, auth
from PIL import Image
from django.urls import reverse
from django.contrib.auth.models import User
from models.utils import *
from models.models_selection import SRResNet, Generator
import torch
import io
import base64

# Create your views here.
large_kernel_size = 9  # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3  # 中间层卷积的核大小
n_channels = 64  # 中间层通道数
n_blocks = 16  # 残差模块数量
scaling_factor = 4  # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())
# 加载模型SRResNet
srresnet_checkpoint_2 = "./models/2X_SRResNet.pth"
checkpoint = torch.load(srresnet_checkpoint_2)
SRResNet_2 = SRResNet(large_kernel_size=9,
                      small_kernel_size=3,
                      n_channels=64,
                      n_blocks=16,
                      scaling_factor=2)
SRResNet_2.to(device)
SRResNet_2.load_state_dict(checkpoint['model'])
SRResNet_2.eval()

srgan_checkpoint_4 = "./models/4X_SRGAN_4.pth"
checkpoint = torch.load(srgan_checkpoint_4)
SRGAN_4 = Generator(large_kernel_size=9,
                    small_kernel_size=3,
                    n_channels=64,
                    n_blocks=16,
                    scaling_factor=4)
SRGAN_4 = SRGAN_4.to(device)
SRGAN_4.load_state_dict(checkpoint['generator'])

srgan_checkpoint_0 = './models/4X_SRGAN_0.pth'
srgan_checkpoint_1 = './models/4X_SRGAN_1.pth'
srgan_checkpoint_2 = './models/4X_SRGAN_2.pth'
srgan_checkpoint_3 = './models/4X_SRGAN_3.pth'
srgan_checkpoint_4 = './models/4X_SRGAN_4.pth'
srgan_checkpoints = [srgan_checkpoint_0, srgan_checkpoint_1, srgan_checkpoint_2, srgan_checkpoint_3, srgan_checkpoint_4]
weights = [0.1, 0.1, 0.2, 0.2, 0.4]

model_map = {
    "2x": SRResNet_2,
    "4x": SRGAN_4,
}


def index(request):
    return render(request, 'login/index.html')


def login(request):
    if request.method == "GET":
        return render(request, 'login/index.html')
    elif request.method == 'POST':
        try:
            username = request.POST.get("username")
            password = request.POST.get("password")
            print(username, password)
            user_obj = auth.authenticate(username=username, password=password)
            if not user_obj:
                return JsonResponse({"msg":
                                         "Username does not exist or password is wrong!"})
            else:
                auth.login(request, user_obj)
                return JsonResponse({"msg": "Log in successfully!"})
        except Exception as e:
            print(repr(e))
            return JsonResponse({"msg": repr(e)})


def signup(request):
    if request.method == 'GET':
        return render(request, 'login/index.html')
    elif request.method == 'POST':
        try:
            username = request.POST.get("username")
            password = request.POST.get("password")
            print(username, password)
            User.objects.create_user(username=username, password=password)
            user_obj = auth.authenticate(username=username, password=password)
            auth.login(request, user_obj)
            return JsonResponse({"msg": "Register successfully!"})
        except Exception as e:
            print(repr(e))
            return JsonResponse({"msg": "Username has been registered!"})


def logout(request):
    auth.logout(request)
    return JsonResponse({"msg": "Log out successfully!"})


def processing(request):
    if request.method == 'POST':
        try:
            print(request)
            pic = request.FILES.get("pic")
            level = request.POST.get("level")
            print(pic.size)
            img = Image.open(pic)
            if img.width * img.height > 2073600:
                raise Exception('Image size exceeds size limit!')
            if level == '2x' or level == '4x':
                model = model_map[level]
                sr_img = get_prediction(img, model)
            else:
                sr_img = get_weighted_prediction(img)
            buf = io.BytesIO()
            sr_img.save(buf, 'jpeg')
            buf.seek(0)
            encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            return render(request, 'login/index.html', {'image_uri': image_uri})
        except Exception as e:
            print(repr(e))
            messages.error(request, repr(e))
            torch.cuda.empty_cache()
            return redirect(reverse('index'))


def get_prediction(img, model):
    img = img.convert('RGB')
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
    lr_img = lr_img.to(device)
    with torch.no_grad():
        # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = model(lr_img).squeeze(0).detach()
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        return sr_img


def get_weighted_prediction(img):
    img = img.convert('RGB')
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
    lr_img = lr_img.to(device)
    count = 0
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

        with torch.no_grad():
            # detach(): 返回一个新的tensor，从当前计算图中分离下来，但是仍指向原变量的存放位置，不同之处只是require_grad为False，得到的tensor永远不需要计算梯度，不具有grad
            sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            if count == 0:
                final_img = sr_img * weight
            else:
                final_img += sr_img * weight
        count += 1
        final_img = convert_image(final_img, source='[-1, 1]', target='pil')
        return final_img
