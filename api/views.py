import re
from urllib.parse import uses_relative
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.http import require_http_methods
from api.models import UserInfo
# Create your views here.


def index(request):
    print(request.COOKIES.get('is_login'))
    return render(request, 'home/index.html')


def test(request):
    return render(request, "test/index.html")


def login(request):
    if request.method == "GET":
        return render(request, "home/index.html")
    email = request.POST.get('email')
    password = request.POST.get('password')
    print(email)
    print(password)
    try:
        rep = JsonResponse({'msg': "Log in successfully!"})
        user_info = UserInfo.objects.filter(email=email)
        user_pass = False
        print(user_info)
        for i in user_info:
            if str(i.password) == password:
                rep.set_cookie('is_login', True, max_age=600)
                return rep
            else:
                return JsonResponse({
                    'msg': "Password is wrong!"
                })
        if not user_pass:
            return JsonResponse({
                'msg': "Email does not exist!"
            })
    except Exception as e:
        print(repr(e))
        return JsonResponse({
            'msg': repr(e)
        })


def register(request):
    try:
        email = request.POST.get('email')
        password = request.POST.get('password')
        UserInfo.objects.create(
            email=email,
            password=password
        )
        return JsonResponse({'msg': "Register successfully!"})
    except Exception as e:
        print(repr(e))
        return JsonResponse({'msg': repr(e)})
