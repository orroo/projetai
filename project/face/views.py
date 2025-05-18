from django.shortcuts import render

# Create your views here.
def face(request):
    return render(request,"face.html")