from functools import wraps
from django.http import JsonResponse
from firebase_admin import auth

def firebase_auth_required(view_func):
    """
    Decorador que verifica el Firebase ID Token enviado en el encabezado
    de Authorization (Bearer Token) y adjunta el UID verificado al request.
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # 1. Obtener el token del encabezado (Authorization: Bearer <token>)
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return JsonResponse({'error': 'No autorizado. Token Bearer requerido.'}, status=401)
            
        try:
            # Extraer el token de la cadena 'Bearer '
            id_token = auth_header.split('Bearer ')[1]
        except IndexError:
            return JsonResponse({'error': 'Formato de Token Bearer no válido.'}, status=401)

        try:
            # 2. Verificar el ID Token
            # Esta función verifica la firma, la expiración y la autenticidad.
            decoded_token = auth.verify_id_token(id_token)
            
            # 3. Obtener el UID seguro
            uid = decoded_token['uid']
            
            # 4. Adjuntar el UID seguro a la solicitud
            request.user_id = uid 
            
            # Ejecutar la función de vista original
            return view_func(request, *args, **kwargs)

        except auth.InvalidIdTokenError:
            return JsonResponse({'error': 'Token de Firebase inválido o caducado.'}, status=401)
        except Exception as e:
            # Manejo de otros errores de Firebase
            print(f"Error de verificación de Firebase: {e}")
            return JsonResponse({'error': 'Error interno de autenticación.'}, status=500)

    return wrapper