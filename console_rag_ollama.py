import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama 
from langchain.prompts import PromptTemplate

# --- Configuration ---
# Embedding model used during creation (must match create_embeddings_ollama.py)
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text"
VECTOR_STORE_PATH = "faiss_index_ollama"  

OLLAMA_LLM_MODEL_NAME = "llama3:8b"
# OLLAMA_LLM_MODEL_NAME = "llama3:8b-instruct" # Original, but user had issues
# OLLAMA_LLM_MODEL_NAME = "mistral:7b-instruct" # Alternative

TOP_K_DOCUMENTS = 5 

PROMPT_TEMPLATE = """
Use only the following pieces of context to answer the question at the end.
If you don't know the answer from the context or the context is not relevant to the question,
just say that you don't have enough information from the documents to answer.
Do not use any prior knowledge or information outside of the provided context.

Context:
{context}

Question: {question}

Helpful Answer:
"""

def run_chat_console_ollama():
    """
    Loads the vector store and Ollama LLM, then starts a console-based chat session.
    """
    print(f"Starting chat with documents application using Ollama LLM: {OLLAMA_LLM_MODEL_NAME}...")

    if not os.path.exists(VECTOR_STORE_PATH) or not os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        print(f"Vector store not found at '{VECTOR_STORE_PATH}'.")
        print("Please run the 'create_embeddings_ollama.py' script first.")
        return

    print(f"Initializing Ollama embedding model: {OLLAMA_EMBEDDING_MODEL_NAME} for loading vector store...")
    try:
        embeddings_model = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL_NAME)
        _ = embeddings_model.embed_query("Test query for embeddings.") # Test connection
        print("Ollama embedding model initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama embedding model: {e}")
        print(f"Ensure Ollama service is running and model '{OLLAMA_EMBEDDING_MODEL_NAME}' is available.")
        return

    print(f"Loading FAISS vector store from '{VECTOR_STORE_PATH}'...")
    try:
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True # Necessary for FAISS with custom embeddings
        )
        print("FAISS vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS vector store: {e}")
        return

    print(f"Initializing Ollama LLM: {OLLAMA_LLM_MODEL_NAME}...")
    try:
        llm = Ollama(
            model=OLLAMA_LLM_MODEL_NAME,
            temperature=0.3, # Lower temperature for more factual RAG answers
            # num_ctx=4096, # Example: Explicitly set context window if needed, often inferred
        )
        
        _ = llm.invoke("Briefly explain what an LLM is.")
        print("Ollama LLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama LLM: {e}")
        print(f"Ensure Ollama service is running and model '{OLLAMA_LLM_MODEL_NAME}' is pulled (e.g., 'ollama pull {OLLAMA_LLM_MODEL_NAME}').")
        return

    print("Creating RetrievalQA chain...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': TOP_K_DOCUMENTS}
    )

    qa_prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" is fine for models with larger context windows like Llama 3
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    print("RetrievalQA chain created.")
    print("\n--- Chat with your documents (Ollama Edition)! ---")
    print(f"Using LLM: {OLLAMA_LLM_MODEL_NAME} | Embedding Model: {OLLAMA_EMBEDDING_MODEL_NAME}")
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                print("Exiting chat. Goodbye!")
                break
            if not query.strip():
                continue

            print("Bot is thinking...")
            response = qa_chain.invoke({"query": query})
            answer = response.get("result", "Sorry, I couldn't formulate an answer.").strip()
            print(f"\nBot: {answer}")

            if response.get("source_documents"):
                print("\n--- Source Documents Used ---")
                for i, doc in enumerate(response["source_documents"]):
                    source_name = os.path.basename(doc.metadata.get('source', 'Unknown source'))
                    content_preview = doc.page_content.replace('\n', ' ').strip()[:150] + "..."
                    print(f"  Source {i+1}: (from: {source_name})")
                    print(f"    Content preview: \"{content_preview}\"")
                print("---------------------------")

        except KeyboardInterrupt:
            print("\nExiting chat due to user interruption. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Ensure Ollama is running and the models are available.")

if __name__ == "__main__":
    run_chat_console_ollama()
