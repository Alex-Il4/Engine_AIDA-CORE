import json
import requests
import base64
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from core.models import ChatHistory
import datetime
from dama_ia.firebase.authentication_fb import firebase_auth_required
from django.http import StreamingHttpResponse
from django.http import HttpResponse


# Cargar modelos y base de datos vectorial una sola vez
# Esto se ejecuta solo al iniciar el servidor para optimizar el rendimiento.

# Ruta a la DB vectorial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'DB', 'Chroma_storageDB')
OLLAMA_URL = 'http://localhost:11434/api/generate'

# --- INICIALIZACIÓN DE MODELOS LLM/RAG ---
# NOTA: Este código se ejecuta solo una vez al cargar views.py
try:
    ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    llm = Ollama(model="AIDAl-CORE")

    # Cargar la DB vectorial
    vector_db = Chroma(
        embedding_function=ollama_embeddings,
        persist_directory=DB_PATH
    )
    print("Sistemas RAG y LLM cargados correctamente.")

except Exception as e:
    print(f"Error al cargar modelos o base de datos: {e}")
    ollama_embeddings = None
    llm = None
    vector_db = None

# --- PLANTILLA DE PROMPT (RAG) ---
prompt_template = """
Eres un sistema de Inteligrncia Artificial (AI) para Asistencia Inteligente de Diagnóstico y Análisis, útil y amable. Tu nombre es SAM.
Utiliza la información de contexto y el historial de conversación para responder la pregunta del usuario, intenta evitar repetir su nombre para evitar incomodidad y responde en el mismo lenguaje que el usuario.
Para el tema de las recetas, Transcribe el texto visible en la imagen. Identifica y lista claramente los medicamentos,
las dosis, la frecuencia y cualquier otra instrucción legible. NO ofrezcas consejos ni análisis médicos, solo la información de la receta.
Responde con el mismo lenguaje que el usuario. si no sabes el idioma, responde en español.
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


# --- Funciones Auxiliares ---
def _call_llava_model(image_file, user_prompt, user_id):
    """ Función que maneja la lógica de codificación y llamada al modelo LLaVA con streaming. """
    try:
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        analysis_prompt = user_prompt.strip() or (
            "Transcribe el texto visible en la imagen. Identifica y lista claramente los medicamentos, "
            "las dosis, la frecuencia y cualquier otra instrucción legible. "
            "NO ofrezcas consejos ni análisis médicos, solo la información de la receta."
            "Responde con el mismo lenguaje que el usuario. si no sabes el idioma, responde en español."
        )

        payload = {
            "model": "llava",
            "prompt": analysis_prompt,
            "images": [image_base64],
            "stream": True  # Cambiado a True
        }
        
        full_response = ""
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line)
                    if 'response' in json_data:
                        full_response += json_data['response']
                        yield json.dumps({"type": "chunk", "content": json_data['response']}) + "\n"
                    if json_data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        yield json.dumps({"type": "done"}) + "\n"
        return full_response

    except Exception as e:
        print(f"Error en LLaVA: {e}")
        yield json.dumps({"type": "error", "content": f"Error al procesar la imagen con LLaVA: {str(e)}"}) + "\n"
        return f"Error al procesar la imagen con LLaVA: {str(e)}"

def _call_llm_rag_model(user_prompt, user_id):
    """ Función que maneja la lógica de RAG, memoria y llamada al modelo AIDAl-CORE con streaming. """
    if not vector_db:
        yield json.dumps({"type": "error", "content": "Error: Los modelos LLM/RAG no se cargaron correctamente al inicio."}) + "\n"
        return "Error: Los modelos LLM/RAG no se cargaron correctamente al inicio."
        
    try:
        current_date = datetime.date.today().strftime("%d de %B de %Y")
        
        history_objects = ChatHistory.objects.filter(session_id=user_id).order_by('-timestamp')[:3]
        history_formatted = ""
        for turn in reversed(history_objects):
            history_formatted += f"Usuario: {turn.user_message}\n"
            history_formatted += f"Asistente: {turn.bot_response}\n"

        results = vector_db.similarity_search(user_prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        formatted_prompt = rag_prompt.format(
            history=history_formatted,
            context=context,
            question=user_prompt,
            current_date=current_date
        )

        payload = {
            'model': 'AIDAl-CORE', 
            'prompt': formatted_prompt, 
            'stream': True  # Cambiado a True
        }
        
        full_response = ""
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line)
                    if 'response' in json_data:
                        full_response += json_data['response']
                        yield json.dumps({"type": "chunk", "content": json_data['response']}) + "\n"
                    if json_data.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
        
        yield json.dumps({"type": "done"}) + "\n"
        return full_response

    except Exception as e:
        print(f"Error en LLM/RAG: {e}")
        yield json.dumps({"type": "error", "content": f"Error al procesar la solicitud de texto con LLM: {str(e)}"}) + "\n"
        return f"Error al procesar la solicitud de texto con LLM: {str(e)}"

# --- VISTAS DEL API ---
@firebase_auth_required  
@api_view(['POST'])
def smart_generate_response_view(request):
    user_id = request.user_id 
    user_prompt = request.data.get('prompt', '')
    image_file = request.FILES.get('image_file')

    try:
        if image_file:
            # Procesamiento de imagen
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')

            analysis_prompt = user_prompt.strip() or (
                "Transcribe el texto visible en la imagen. Identifica y lista claramente los medicamentos, "
                "las dosis, la frecuencia y cualquier otra instrucción legible. "
                "NO ofrezcas consejos ni análisis médicos, solo la información de la receta."
                "Responde con el mismo lenguaje que el usuario. si no sabes el idioma, responde en español."
            )

            payload = {
                "model": "llava",
                "prompt": analysis_prompt,
                "images": [image_base64],
                "stream": False  # Mantener en False
            }
            
            response_data = requests.post(OLLAMA_URL, json=payload).json()
            generated_text = response_data.get('response', 'Error en LLaVA')
            
            ChatHistory.objects.create(
                session_id=user_id,
                user_message=f"[IMAGEN ANALIZADA] {user_prompt}",
                bot_response=generated_text
            )
            
            return Response({
                "generated_text": generated_text,
                "status": "success"
            })
            
        else:
            # Procesamiento de texto
            if not user_prompt:
                return Response({"error": "Se requiere un prompt"}, status=400)
            
            current_date = datetime.date.today().strftime("%d de %B de %Y")
            
            history_objects = ChatHistory.objects.filter(session_id=user_id).order_by('-timestamp')[:3]
            history_formatted = ""
            for turn in reversed(history_objects):
                history_formatted += f"Usuario: {turn.user_message}\n"
                history_formatted += f"Asistente: {turn.bot_response}\n"

            results = vector_db.similarity_search(user_prompt, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            formatted_prompt = rag_prompt.format(
                history=history_formatted,
                context=context,
                question=user_prompt,
                current_date=current_date
            )

            payload = {
                'model': 'AIDAl-CORE', 
                'prompt': formatted_prompt, 
                'stream': False  # Mantener en False
            }
            
            response_data = requests.post(OLLAMA_URL, json=payload).json()
            generated_text = response_data.get('response', 'Error en el LLM')
            
            ChatHistory.objects.create(
                session_id=user_id,
                user_message=user_prompt,
                bot_response=generated_text
            )

            return Response({
                "generated_text": generated_text,
                "status": "success"
            })
            
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@firebase_auth_required
@api_view(['DELETE'])
def clear_chat_history(request, user_id):
    """
    Vista que elimina todo el historial de chat para un user_id específico.
    """
    # Chequeo de seguridad: El UID del Token (request.user_id) debe coincidir con el UID de la URL (user_id)
    if str(request.user_id) != str(user_id):
        return Response(
            {"error": "No tiene permiso para limpiar el historial de otro usuario."},
            status=status.HTTP_403_FORBIDDEN
        )
        
    try:
        records_to_delete = ChatHistory.objects.filter(session_id=user_id)
        count = records_to_delete.count()

        if count == 0:
            return Response(
                {"message": f"No se encontró historial para el usuario con ID {user_id}."},
                status=status.HTTP_404_NOT_FOUND
            )

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


@firebase_auth_required
@api_view(['GET'])
def get_chat_history(request, user_id):
    """
    Vista que devuelve el historial de chat para un user_id específico.
    """
    # Chequeo de seguridad: El UID del Token (request.user_id) debe coincidir con el UID de la URL (user_id)
    if str(request.user_id) != str(user_id):
        return Response(
            {"error": "No tiene permiso para ver el historial de otro usuario."},
            status=status.HTTP_403_FORBIDDEN
        )

    try:
        records = ChatHistory.objects.filter(session_id=user_id).order_by('timestamp') # Ordenado por tiempo
        
        records_dict = [{
            'user_message': record.user_message,
            'bot_response': record.bot_response,
            'timestamp': record.timestamp
        } for record in records]

        return Response({
            "records": records_dict,
            "status": "success"
        })

    except Exception as e:
        return Response(
            {"error": "Ocurrió un error al intentar obtener el historial.", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )