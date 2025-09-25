from django.urls import path
from .views import generate_response, clear_chat_history

urlpatterns = [
    path('generate/', generate_response, name='generate_response'),
    path('chat/history/clear/<str:user_id>/', clear_chat_history, name='clear_chat_history'),
]