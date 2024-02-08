from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from load_docs import extract_text_from_pdf

def get_vector_store(embedding,text_chunks,db_dir):
    vector_store = Chroma.from_texts(text_chunks, embedding=embedding,persist_directory=db_dir)

def load_vector_store(embedding,db_dir):
    db3 = Chroma(persist_directory=db_dir, embedding_function=embedding)
    return db3
