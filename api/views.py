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
SRResNet_2 = SRResNet_2.to(device)
SRResNet_2.load_state_dict(checkpoint['model'])

SRResNet_2.eval()


def index(request):
    if request.user.is_authenticated:
        return redirect(reverse('home'))
    else:
        return render(request, 'login/index.html')


def home(request):
    if request.user.is_authenticated:
        return render(request, "home/index.html", {"user": request.user.username})
    else:
        messages.error(request, "Unauthorized access! Please log in first!")
        return redirect(reverse('index'))


def login(request):
    if request.method == "GET":
        return render(request, 'login/index.html')
    elif request.method == 'POST':
        try:
            username = request.POST.get("username")
            password = request.POST.get("password")
            user_obj = auth.authenticate(username=username, password=password)
            if not user_obj:
                messages.error(
                    request, "Username does not exist or password is wrong!")
                return redirect(reverse('index'))
            else:
                auth.login(request, user_obj)
                messages.info(request, "Log in successfully!")
                return redirect(reverse('home'))
        except Exception as e:
            print(repr(e))
            messages.error(request, repr(e))
            return redirect(reverse('index'))


def signup(request):
    if request.method == 'GET':
        return render(request, 'login/index.html')
    elif request.method == 'POST':
        try:
            username = request.POST.get("username")
            password = request.POST.get("password")
            User.objects.create_user(username=username, password=password)
            messages.info(request, "Register successfully!")
            user_obj = auth.authenticate(username=username, password=password)
            auth.login(request, user_obj)
            return redirect(reverse('home'))
        except Exception as e:
            print(repr(e))
            messages.error(request, "Username has been registered!")
            return redirect(reverse('index'))


def logout(request):
    auth.logout(request)
    messages.info(request, "Log out successfully!")
    return redirect(reverse('index'))


def processing(request):
    scaling_factor = 4
    if request.method == 'POST':
        pic = request.FILES.get("pic")
        img = Image.open(pic)
        Bicubic_img = img.resize(
            (int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
        Bicubic_img.save('./test_bicubic.jpg')
    return render(request, 'login/index.html', {"pic": Bicubic_img})


def get_prediction():
    pass
# def login(request):
#     if request.method == 'POST':
#         email = request.POST.get('email')
#         password = request.POST.get('password')
#         try:
#             rep = redirect(reverse('home'))
#             user_info = UserInfo.objects.filter(email=email)
#             user_pass = False
#             # print(user_info)
#             for i in user_info:
#                 if str(i.password) == password:
#                     request.session['email'] = email
#                     messages.info(request, "Log in successfully!")
#                     return rep
#                 else:
#                     print("Password is wrong!")
#                     messages.error(request, "Password is wrong!")
#                     return redirect(reverse('index'))
#             if not user_pass:
#                 messages.error(request, "Email does not exist!")
#                 return redirect(reverse('index'))
#         except Exception as e:
#             print(repr(e))
#             messages.error(request, repr(e))
#             return redirect(reverse('index'))


# def signup(request):
#     if request.method == 'GET':
#         return render(request, 'home/index.html')
#     elif request.method == 'POST':
#         try:
#             rep = redirect(reverse('home'))
#             email = request.POST.get('email')
#             password = request.POST.get('password')
#             UserInfo.objects.create(
#                 email=email,
#                 password=password
#             )
#             request.session['email'] = email
#             messages.info(request, "Register successfully!")
#             return rep
#         except Exception as e:
#             print(repr(e))
#             messages.error(request, repr(e))
#             return redirect(reverse('index'))
