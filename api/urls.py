from django.urls import path
from api import views

urlpatterns = [
    path('', views.index, name='index'),
    path('test/', views.test, name='test'),
    path('login/', views.login, name='login'),
    path('register/', views.register, name='register'),
]