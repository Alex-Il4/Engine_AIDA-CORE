from django.urls import path
from .views import generate_response

urlpatterns = [
    path('generate/', generate_response, name='generate_response'),
    path('clear_chat_history/<str:user_id>/', generate_response, name='clear_chat_history'),
]