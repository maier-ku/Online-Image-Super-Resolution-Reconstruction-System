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
from models.models import SRResNet
import torch
import io
import base64
# Create your views here.
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 2      # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srresnet_checkpoint = "./models/2X_SRResNet.pth"

# 加载模型SRResNet
checkpoint = torch.load(srresnet_checkpoint)
SRResNet_2 = SRResNet(large_kernel_size=large_kernel_size,
                      small_kernel_size=small_kernel_size,
                      n_channels=n_channels,
                      n_blocks=n_blocks,
                      scaling_factor=scaling_factor)
SRResNet_2.load_state_dict(checkpoint['model'])

SRResNet_2.eval()

model_map = {
    "2x": SRResNet_2,
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
            print(pic, level)
            model = model_map[level]
            # messages.info(request, "Processing...Please wait...")
            sr_img = get_prediction(pic, model)
            # cache_path = "./cache/"+str(request.session["_auth_user_hash"])+".jpg"
            buf = io.BytesIO()
            sr_img.save(buf, 'jpeg')
            buf.seek(0)
            encoded_img = base64.b64encode(buf.getvalue()).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            return render(request, 'login/index.html', {'image_uri': image_uri})
            response = FileResponse(buf)
            # response['content_type'] = "application/octet-stream"
            # response['Content-Disposition'] = 'attachment; filename="SR_result.jpg"'
            # return response
        except Exception as e:
            print(repr(e))
            messages.error(request, repr(e))
            return redirect(reverse('index'))


def get_prediction(pic, model):
    img = Image.open(pic)
    img = img.convert('RGB')
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)
    with torch.no_grad():
        # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = model(lr_img).squeeze(0).detach()
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        return sr_img
