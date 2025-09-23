import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ruta a los documentos dentro de la carpeta 'DB'
documents_path = "./DB/documents"

# Directorio para almacenar la base de datos vectorial local
db_path = "./DB/Chroma_storageDB"

def ingest_data():
    documents = []
    if os.path.exists(documents_path):
        for file in os.listdir(documents_path):
            if file.endswith(".txt"):
                loader = TextLoader(os.path.join(documents_path, file), encoding="utf-8")
                documents.extend(loader.load())
    else:
        print(f"Error: El directorio {documents_path} no existe.")
        return
        
    # Dividir los documentos en fragmentos
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Usar el modelo de embeddings de Ollama
    ollama_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    # Crear la base de datos vectorial de Chroma de forma local
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=ollama_embeddings,
        persist_directory=db_path
    )

    print(f"Indexación completa. {len(chunks)} fragmentos procesados.")

if __name__ == "__main__":
    ingest_data()