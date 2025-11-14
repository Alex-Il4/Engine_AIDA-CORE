# dama_ia/firebase/auth_settings.py (Revisión)

import firebase_admin
from firebase_admin import credentials
import os

# --- 1. Definir la ruta de las credenciales ---
# La ruta la estás cargando desde el .env (imagen image_94b331.png)
SERVICE_ACCOUNT_PATH = os.environ.get('FIREBASE_CREDENTIALS')

# --- 2. Inicializar la App si aún no se ha hecho ---
# firebase_admin._apps comprueba si ya existe una app predeterminada
if not firebase_admin._apps:
    try:
        # Crea el objeto de credenciales
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        
        # Inicializa la aplicación de Firebase
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK inicializado con éxito.")
        
    except FileNotFoundError:
        print(f"ERROR: Archivo de credenciales no encontrado en la ruta: {SERVICE_ACCOUNT_PATH}")
    except Exception as e:
        # Esto captura errores si la ruta no está bien formada o la clave es incorrecta
        print(f"ERROR al inicializar Firebase Admin SDK: {e}")