from django.shortcuts import render
from mysite import KMeans as km
from mysite import NaiveBayes as nb

genre1 = ""
genre2 = ""
genre3 = ""
review = ""
#output=""
moviename=""
review1=""
list = ["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","IMAX","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"]
def home(request):
    return render(request,'home.html',{})
def abstract(request):
    return render(request,'abstract.html',{})

def kmeans(request):
    #type_code = "Kmeans"
    genre1 = request.POST.get('Genre1')
    genre2 = request.POST.get('Genre2')
    genre3 = request.POST.get('Genre3')

    print(genre1)
    print(genre2)
    print(genre3)

    if genre1 is None and genre2 is None and genre3 is None:
        return render(request, 'kmeans.html', {})
    else:
        km.call_KM(genre1,genre2,genre3)
        print('method')
        print(genre1)
        print(genre2)
        print(genre3)
        return render(request,'kmeans.html',{})
# Create your views here.


def naivebayes(request):
    moviename = request.POST.get('moviename')
    review1 = request.POST.get('review')

    print(moviename)
    print(review1)

    if moviename is None :
        a=0
        return render(request, 'naivebayes.html', {})
    else :
        NB_RESULT = nb.call_NB(moviename,review1)
        #movie,review,freshness,avgfreshness = nb.call_NB(moviename, review1)
        return render(request, 'naivebayes.html', {'moviename': NB_RESULT[0],'review': NB_RESULT[1],'freshness':NB_RESULT[2],'avgfreshness':NB_RESULT[3]})
        #return render(request, 'naivebayes.html',{'moviename': movie, 'review': review, 'freshness': freshness,'avgfreshness': avgfreshness})
        #return render(request,'naivebayes.html',{})

#-------------------------------------------------------------------------------------------------------
#--------------------------------------------Views for Kmeans Graph-------------------------------------------
def graph_view1(request):
    return render(request, "Graph1.html", {})

def graph_view2(request):
    return render(request, "Graph2.html", {})

def graph_view3(request):
    return render(request, "Graph3.html", {})

def graph_view4(request):
    return render(request, "Graph4.html", {})

def graph_view5(request):
    return render(request, "Graph5.html", {})

def graph_view6(request):
    return render(request, "Graph6.html", {})

def graph_view7(request):
    return render(request, "Graph7.html", {})

def graph_view8(request):
    return render(request, "Graph8.html", {})


