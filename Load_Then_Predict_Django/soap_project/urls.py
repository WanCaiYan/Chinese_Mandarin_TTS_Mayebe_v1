"""soap_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.10/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.contrib import admin
from django.urls import path
from rango import views


urlpatterns = [
    path('', views.index1, name='index1'),
    #url(r'^admin/', admin.site.urls),
    #url(r'^rango/index3', views.index3, name='index3'),
    #url(r'^rango/index2', views.index2, name='index2'),
    #url(r'^rango/index/$', views.index, name='index'),
    url(r'^rango/index/(?P<path>.+)/$', views.index, name='index'),
    #url(r'^rango/index/(?P<prm>)/$', views.index, name='index'),
    #url(r'^$', admin.site.urls),
]
