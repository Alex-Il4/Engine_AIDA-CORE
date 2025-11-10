import firebase_admin
from firebase_admin import credentials
import os

SERVICE_ACCOUNT_PATH = os.getenv('FIREBASE_CREDENTIALS')

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    
    # Inicializa la aplicación de Firebase en el backend
    firebase_admin.initialize_app(cred)