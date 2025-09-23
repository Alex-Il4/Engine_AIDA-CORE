from django.db import models

class ChatHistory(models.Model):
    """
    Modelo para almacenar el historial de la conversación.
    """
    # Identificador único para cada sesión de chat.
    session_id = models.CharField(max_length=255, db_index=True)

    # Mensaje del usuario
    user_message = models.TextField()

    # Respuesta del modelo
    bot_response = models.TextField()

    # Fecha y hora de la interacción
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Sesión: {self.session_id} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    class Meta:
        verbose_name_plural = "Historial de Chat"
        ordering = ['timestamp']