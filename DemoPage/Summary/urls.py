from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_sentence_page, name='sentence'),
    path('summary/', views.summarize_sentence, name='summary'),
]