import firebase_admin
from firebase_admin import credentials

SERVICE_ACCOUNT_PATH = 'chat-f521d-firebase-adminsdk-fbsvc-d83809c855.json'

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    
    # Inicializa la aplicación de Firebase en el backend
    firebase_admin.initialize_app(cred)