import email
from django.db import models

# Create your models here.
class UserInfo(models.Model):
    email = models.CharField(max_length=30, unique=True)
    password = models.CharField(max_length=30)

    class Meta:
        db_table = 'UserInfo'
        verbose_name = "用户信息表"