from django.shortcuts import render

from .classifiers.classifier import LogisticRegressionClassifier
from .classifiers.collector import collect_tweets


logit_classifier = LogisticRegressionClassifier()


def index(request):
    return render(request, 'clasificacion/index.html')


def hashtag(request):
    if request.method == 'POST':
        data = collect_tweets()
        prediction = logit_classifier.classify(data)

        if len(prediction) == 0:
            context = { 'info': 'No existen tweets que cumplan con los parámetros de búsqueda ingresados.' }
        else:
            context = {
                'resultados': True,
                'pedidosAyuda': prediction.get(0, []),
                'ofertas': prediction.get(1, []),
                'ninguna': prediction.get(2, [])
            }
        return render(request, 'clasificacion/hashtag.html', context)
    else:
        return render(request, 'clasificacion/hashtag.html')


def tweet(request):
    return render(request, 'clasificacion/tweet.html')
