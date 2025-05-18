from django.contrib import admin
from django.urls import path
from .views import *
from django.urls import path
from django.urls import path

urlpatterns = [
    path('upload/', upload_mat, name='upload_mat'),
]