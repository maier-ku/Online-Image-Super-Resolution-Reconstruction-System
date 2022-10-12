from django.urls import path, re_path
from api import views

urlpatterns = [
    re_path('^$', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('login/', views.login, name='login'),
    path('signup/', views.signup, name='signup'),
    path('logout/', views.logout, name='logout'),
    path('processing/', views.processing, name='processing'),
]