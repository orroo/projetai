 
from django.urls import path
from .views import face 
urlpatterns = [
path('recognition/',face,name='face'),]

 