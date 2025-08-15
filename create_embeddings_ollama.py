import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings 
from langchain_community.vectorstores import FAISS

# --- Configuration ---
DOCUMENTS_PATH = "documents"  # Directory All source files will be read for

#To-do: look at extracting some of the blow into a centralised config file for cross-script sharing.
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"
VECTOR_STORE_PATH = "faiss_index_ollama"  
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 150  

def create_vector_store_ollama():
    print(f"Starting to create vector store using Ollama model: {OLLAMA_EMBEDDING_MODEL_NAME}...")

    if not any(f.endswith((".txt", ".pdf")) for f in os.listdir(DOCUMENTS_PATH)):
        print(f"ERROR: The '{DOCUMENTS_PATH}' directory does not contain any .txt or .pdf files.")
        return

    if os.path.exists(VECTOR_STORE_PATH):
        print(f"Removing existing Ollama vector store at '{VECTOR_STORE_PATH}'...")
        shutil.rmtree(VECTOR_STORE_PATH)
        print("Existing Ollama vector store removed.")

    all_documents = []
    print(f"Loading documents from '{DOCUMENTS_PATH}'...")
    
    #To-do:Look at extending to additional file types in future
    txt_loader = DirectoryLoader(
        DOCUMENTS_PATH, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}, show_progress=True,
        use_multithreading=True, silent_errors=True
    )
    pdf_loader = DirectoryLoader(
        DOCUMENTS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True, use_multithreading=True, silent_errors=True
    )

    try:
        txt_docs = txt_loader.load()
        if txt_docs: all_documents.extend(txt_docs)
        print(f"Loaded {len(txt_docs)} .txt documents.")
    except Exception as e:
        print(f"Error loading .txt documents: {e}")

    try:
        pdf_docs = pdf_loader.load()
        if pdf_docs: all_documents.extend(pdf_docs)
        print(f"Loaded {len(pdf_docs)} .pdf documents.")
    except Exception as e:
        print(f"Error loading .pdf documents: {e}")


    if not all_documents:
        print(f"No documents (.txt or .pdf) were successfully loaded from '{DOCUMENTS_PATH}'.")
        return
    print(f"Total documents loaded: {len(all_documents)}.")


    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    texts = text_splitter.split_documents(all_documents)
    print(f"Split documents into {len(texts)} chunks.")

    if not texts:
        print("No text chunks were generated. Check documents and splitting parameters.")
        return


    print(f"Initializing Ollama embedding model: {OLLAMA_EMBEDDING_MODEL_NAME}...")
    try:
        embeddings_model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_NAME)
        # Test the embedding model with a sample text
        _ = embeddings_model.embed_query("Test query to initialize and check Ollama embeddings.")
        print("Ollama embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama embedding model: {e}")
        print(f"Ensure Ollama service is running and model '{OLLAMA_EMBEDDING_MODEL_NAME}' is pulled (e.g., 'ollama pull {OLLAMA_EMBEDDING_MODEL_NAME}').")
        return


    print("Creating FAISS vector store with Ollama embeddings... This might take a while.")
    try:
        vector_store = FAISS.from_documents(texts, embeddings_model)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store created and saved successfully at '{VECTOR_STORE_PATH}'.")
    except Exception as e:
        print(f"Error creating or saving FAISS vector store: {e}")

if __name__ == "__main__":
    create_vector_store_ollama()
