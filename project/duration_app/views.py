from django.shortcuts import render

# Create your views here.

def about(request):
    return render(request,"about-us.html")


def blog(request):
    return render(request,"blog.html")


def contact(request):
    return render(request,"contact.html")


def department(request):
    return render(request,"department.html")

    
def doctors(request):
    return render(request,"doctors.html")


    
def element(request):
    return render(request,"element.html")


    
def index(request):
    return render(request,"index.html")



    
def singleblog(request):
    return render(request,"single-blog.html")

def sign_in(request):
    return render(request,"sign_in.html")


def sign_up(request):
    return render(request,"sign_up.html")



