import json
import requests
import base64
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
#from .authentication_fb import firebase_auth_required

# Cargar modelos y base de datos vectorial una sola vez
# Esto se ejecuta solo al iniciar el servidor para optimizar el rendimiento.

# Ruta a la DB vectorial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, '..', 'DB', 'Chroma_storageDB')
OLLAMA_URL = 'http://localhost:11434/api/generate'

secure_Id = requests.user_id

# Iniciando modelos de ollama.
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

# Plantilla para el prompt con RAG y memoria
prompt_template = """
Eres un sistema de Inteligrncia Artificial (AI) para Asistencia Inteligente de Diagnóstico y Análisis, útil y amable. Tu nombre es SAM.
Utiliza la información de contexto y el historial de conversación para responder la pregunta del usuario, intenta evitar repetir su nombre para evitar incomodidad y responde en el mismo lenguaje que el usuario.
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

 #Inicio del endpoint para generear respuesta ya sea de LLM/RAG o LlaVA
 # --- Funciones Auxiliares ---

def _call_llava_model(image_file, user_prompt, user_id):
    """
    Función que maneja la lógica de codificación y llamada al modelo LLaVA.
    """
    try:
        # 1. Leer y Codificar Imagen a Base64
        image_data = image_file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 2. Definir Prompt para Análisis
        analysis_prompt = user_prompt.strip() or (
            "Actúa como un asistente médico. Describe los medicamentos, la dosis y las instrucciones "
            "de la receta de forma clara y sencilla."
        )

        payload = {
            "model": "llava",  # Modelo LLaVA
            "prompt": analysis_prompt,
            "images": [image_base64],
            "stream": False
        }
        
        # 3. Llamar a Ollama
        response_data = requests.post(OLLAMA_URL, json=payload).json()
        generated_text = response_data.get('response', 'Error: No se recibió respuesta de LLaVA.')
        
        return generated_text

    except Exception as e:
        print(f"Error en LLaVA: {e}")
        return f"Error al procesar la imagen con LLaVA: {str(e)}"

def _call_llm_rag_model(user_prompt, user_id):
    """
    Función que maneja la lógica de RAG, memoria y llamada al modelo AIDAl-CORE.
    """
    try:
        # 1. Obtener datos de contexto (RAG y Memoria)
        current_date = datetime.date.today().strftime("%d de %B de %Y")
        history_objects = ChatHistory.objects.filter(session_id=user_id).order_by('-timestamp')[:5]
        history_formatted = ""
        for turn in reversed(history_objects):
            history_formatted += f"Usuario: {turn.user_message}\n"
            history_formatted += f"Asistente: {turn.bot_response}\n"

        # 2. RAG: Buscar documentos relevantes
        results = vector_db.similarity_search(user_prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in results])

        # 3. Crear Prompt Aumentado
        formatted_prompt = rag_prompt.format(
            history=history_formatted,
            context=context,
            question=user_prompt,
            current_date=current_date
        )

        # 4. Llamar a Ollama
        response_data = requests.post(
            OLLAMA_URL,
            json={'model': 'AIDAl-CORE', 'prompt': formatted_prompt, 'stream': False},
            headers={'Content-Type': 'application/json'}
        ).json()
        
        generated_text = response_data.get('response', 'Error: No se recibió respuesta del LLM.')
        
        return generated_text

    except Exception as e:
        print(f"Error en LLM/RAG: {e}")
        return f"Error al procesar la solicitud de texto con LLM: {str(e)}"


# --- Vista Unificada Principal ---

@api_view(['POST'])
def smart_generate_response_view(request):
    """
    Vista UNIFICADA que decide si usar LLaVA (si hay imagen) o LLM (si es solo texto).
    """
    user_id = request.data.get('user_id')
    user_prompt = request.data.get('prompt', '')

    if not user_id:
         return Response({"error": "El 'user_id' es requerido."}, status=status.HTTP_400_BAD_REQUEST)
    
    
    #Comprobar si hay un archivo de imagen en la peticion (form-data)
    image_file = request.FILES.get('image_file')

    if image_file:
        #1: Procesamiento de Imagen
        print(f"Ruta: Imagen detectada para user_id {user_id}. Usando LLaVA.")
        generated_text = _call_llava_model(image_file, user_prompt, user_id)
        
        #Si el usuario solo envio la imagen sin prompt, usar un mensaje por defecto
        user_message_to_save = user_prompt.strip() if user_prompt.strip() else "Usuario envió una imagen para análisis."

        ChatHistory.objects.create(
            session_id=user_id,
            #Añadir un tag para que el LLM sepa que fue una respuesta de imagen
            user_message=f"[IMAGEN ANALIZADA] {user_message_to_save}",
            bot_response=generated_text
        )
    else:
        # --- PATH 2: Procesamiento de Texto (LLM/RAG) ---
        
        if not user_prompt:
            return Response({"error": "Se requiere un 'prompt' o un archivo de imagen."}, status=status.HTTP_400_BAD_REQUEST)
            
        print(f"Ruta: Solo texto para user_id {user_id}. Usando LLM/RAG.")
        generated_text = _call_llm_rag_model(user_prompt, user_id)
        
        # Guardar historial SOLO si es una conversación de texto con el LLM/RAG
        ChatHistory.objects.create(
            session_id=user_id,
            user_message=user_prompt,
            bot_response=generated_text
        )

    return Response({
        "generated_text": generated_text,
        "status": "success"
    })
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
    
    #Iincio del endpoint para ver historial
@api_view(['GET'])
def get_chat_history(request, user_id):
    """
    Vista que devuelve el historial de chat para un user_id específico.
    """
    try:
        # Obtener los registros que coinciden con el user_id
        records = ChatHistory.objects.filter(session_id=user_id)

            # Convertir los registros a un diccionario
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