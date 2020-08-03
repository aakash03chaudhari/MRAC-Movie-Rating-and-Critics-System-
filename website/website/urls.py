"""website URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from mysite import views

urlpatterns = [
    path('',views.home,name='home'),
    path('kmeans.html/',views.kmeans,name="kmeans"),
    path('kmeans.html/Graph1.html', views.graph_view1, name="graph"),
    path('kmeans.html/Graph2.html', views.graph_view2, name="graph"),
    path('kmeans.html/Graph3.html', views.graph_view3, name="graph"),
    path('kmeans.html/Graph4.html', views.graph_view4, name="graph"),
    path('kmeans.html/Graph5.html', views.graph_view5, name="graph"),
    path('kmeans.html/Graph6.html', views.graph_view6, name="graph"),
    path('kmeans.html/Graph7.html', views.graph_view7, name="graph"),
    path('kmeans.html/Graph8.html', views.graph_view8, name="graph"),
    path('naivebayes.html/',views.naivebayes,name="naivebayes"),
    path('home.html',views.home,name='home'),
    path('abstract.html',views.abstract,name='abstract'),
]
