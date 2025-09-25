import json
import requests
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from core.models import ChatHistory
import datetime

# Cargar modelos y base de datos vectorial una sola vez
# Esto se ejecuta solo al iniciar el servidor para optimizar el rendimiento.

# Ruta a la DB vectorial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'DB', 'Chroma_storageDB')

# Iniciando modelos de ollama.
try:

    ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    llm = Ollama(model="AIDAl-CORE")

    # Cargar la DB vectorial
    vector_db = Chroma(
        embedding_function=ollama_embeddings,
        persist_directory=DB_PATH
    )
    print("Sistemas de RAG y LLM cargados correctamente.")

except Exception as e:
    print(f"Error al cargar modelos o base de datos: {e}")
    ollama_embeddings = None
    llm = None
    vector_db = None

# Plantilla para el prompt con RAG y memoria
prompt_template = """
Eres un sistema de Inteligrncia Artificial (IA) para Asistencia Inteligente de Diagnóstico y Análisis, útil y amable. Tu nombre es SAM.
Utiliza la información de contexto y el historial de conversación para responder la pregunta del usuario, intenta evitar repetir su nombre para evitar incomodidad .
La fecha actual es {current_date}.
Si no tienes el contexto o no sabes la respuesta, responde basándote en tu conocimiento general. No inventes información.

---
HISTORIAL DE LA CONVERSACIÓN:
{history}
---
CONTEXTO DE LA BASE DE CONOCIMIENTO:
{context}
---
PREGUNTA:
{question}
"""
rag_prompt = PromptTemplate.from_template(prompt_template)


@api_view(['POST'])
def generate_response(request):
    """
    Vista que recibe la pregunta y el user_id, y devuelve una respuesta contextualizada.
    """
    if 'prompt' not in request.data or 'user_id' not in request.data:
        return Response(
            {"error": "El 'prompt' y el 'user_id' son requeridos."},
            status=status.HTTP_400_BAD_REQUEST
        )

    prompt = request.data.get('prompt')
    user_id = request.data.get('user_id')

    if not llm or not vector_db:
        return Response(
            {"error": "El servidor no está disponible. Revisar conexion con Ollama o la DB vectorial."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    try:
        # Obtener la fecha actual en un formato legible
        current_date = datetime.date.today().strftime("%d de %B de %Y")
        # 1. Recuperar el historial de la conversación (los últimos 5 mensajes)
        history_objects = ChatHistory.objects.filter(session_id=user_id).order_by('-timestamp')[:5]
        history_formatted = ""
        for turn in reversed(history_objects): # Invertir para que el orden sea cronológico
            history_formatted += f"Usuario: {turn.user_message}\n"
            history_formatted += f"Asistente: {turn.bot_response}\n"

        # 2. Fase de Recuperación RAG: Buscar los documentos más relevantes
        results = vector_db.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        # 3. Fase de Aumento: Crea el prompt aumentado con el contexto y la memoria
        formatted_prompt = rag_prompt.format(
            history=history_formatted,
            context=context,
            question=prompt,
            current_date=current_date
        )

        # 4. Fase de Generación: Envía el prompt completo a Ollama
        response_data = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'AIDAl-CORE',
                'prompt': formatted_prompt,
                'stream': False,
            },
            headers={'Content-Type': 'application/json'}
        ).json()
        generated_text = response_data.get('response', '')

        # 5. Guardar la nueva interacción en el historial
        ChatHistory.objects.create(
            session_id=user_id,
            user_message=prompt,
            bot_response=generated_text
        )

        return Response({
            "generated_text": generated_text,
            "status": "success"
        })

    except requests.exceptions.RequestException as e:
        return Response(
            {"error": "Error al conectar con el servidor de Ollama.", "details": str(e)},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
 #Inicio del endpoint para eliminar historial
@api_view(['DELETE'])
def clear_chat_history(request, user_id):
    """
    Vista que elimina todo el historial de chat para un user_id específico.
    """
    try:
        # Contar los registros que se van a eliminar
        records_to_delete = ChatHistory.objects.filter(session_id=user_id)
        count = records_to_delete.count()

        if count == 0:
            return Response(
                {"message": f"No se encontró historial para el usuario con ID {user_id}."},
                status=status.HTTP_404_NOT_FOUND
            )

        # Eliminar todos los registros que coinciden con el user_id
        records_to_delete.delete()

        return Response(
            {"message": f"Se ha eliminado el historial de chat para el usuario {user_id}. {count} registros eliminados."},
            status=status.HTTP_200_OK
        )

    except Exception as e:
        return Response(
            {"error": "Ocurrió un error al intentar limpiar el historial.", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )