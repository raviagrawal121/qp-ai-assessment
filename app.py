import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from load_docs import extract_text_from_pdf
from embeddings  import get_vector_store,load_vector_store
from generate_prompt import generate_prompt_chain
from generate_response import generate_response_answer
from pathlib import Path

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
MODEL =os.getenv("MODEL")
db_dir = os.getenv("DB_DIR")
genai.configure(api_key=api_key)
embeddings = GoogleGenerativeAIEmbeddings(model = f"models/{MODEL}")

def main():
    st.set_page_config("DOCUBOT")
    st.header("Chat with PDF,TXT or DOCX using GenAI")

    user_question = st.text_input("Ask a Question from the File")

    if user_question:
        vector_db = load_vector_store(embedding=embeddings,db_dir=db_dir)
        chain = generate_prompt_chain()
        response = generate_response_answer(vector_db=vector_db,user_question=user_question,chain=chain)
        st.text_area("Answer", value=response["output_text"], height=200)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload a Single Document:", accept_multiple_files=False, type=['pdf', 'txt','docx'])
        if st.button("Submit & Process"):
            if pdf_docs is not None:
                filetype = Path(pdf_docs.name).suffix
            with st.spinner("Processing..."):
                text = extract_text_from_pdf(pdf_docs,filetype)
                get_vector_store(text_chunks=text,embedding=embeddings,db_dir=db_dir)
                st.success("Done")
    

if __name__ == "__main__":
    main()