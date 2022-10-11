from urllib.parse import uses_relative
from django.shortcuts import redirect, render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages, auth
from api.models import UserInfo
from django.urls import reverse
from django.contrib.auth.models import User
# from django.contrib.auth.decorators import login_required
# Create your views here.


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
            print(username)
            print(password)
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
            rep = redirect(reverse('home'))
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
    messages.info(request, "Log in successfully!")
    return redirect(reverse('index'))


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
