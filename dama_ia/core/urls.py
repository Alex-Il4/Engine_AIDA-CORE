from django.urls import path
from .views import smart_generate_response_view, clear_chat_history, get_chat_history

urlpatterns = [
    # Rutas para interactuar con el modelo de LLM
    path('chat/history/clear/<str:user_id>/', clear_chat_history, name='clear_chat_history'),
    path('chat/history/<str:user_id>/', get_chat_history, name='get_chat_history'),
    path('chat/response/', smart_generate_response_view, name='smart_generate_response'),
]