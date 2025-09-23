# Engine_AIDA-CORE
Motor para ejecutar y consumir modelos de Inteligencia Artificial de Ollama, para un sistema medico.

1-Crear entorno virtual
```
python3 -m venv .venv
```
2-Activar entorno
```
.venv\Scripts\actívate
```
3-Instalar dependencias
```
pip install -r requirements.txt
```
4-Crea la siguiente carpeta y sus subcarpetas:
dama_ia
 |--DB
    |--Chroma_storageDB
    |--Documents

5-Iniciar migraciones de la DB
```
python manage.py makemigrations
```
```
python manage.py migrate
```
6-Iniciar servidor
```
python manage.py runserver
```
