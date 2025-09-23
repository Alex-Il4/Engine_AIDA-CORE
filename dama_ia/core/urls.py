from django.urls import path
from .views import generate_response

urlpatterns = [
    path('generate/', generate_response, name='generate_response'),
]