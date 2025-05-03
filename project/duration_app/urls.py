from django.contrib import admin
from django.urls import path
from .views import *


urlpatterns = [
    path('about/', about,name='about'),
    path('blog/', blog,name='blog'),
    path('contact/', contact,name='contact'),
    path('department/', department,name='department'),
    path('doctors/', doctors,name='doctors'),
    path('element/', element,name='element'),
    path('index/', index,name='index'),
    path('singleblog/', singleblog,name='singleblog'),
]
