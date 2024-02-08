from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(filepath, filetype='pdf'):
    if filetype == ".pdf":
        docs=""
        pdf_reader= PdfReader(filepath)
        for page in pdf_reader.pages:
            docs+= page.extract_text()
    elif filetype == ".txt":
        content = filepath.getvalue()
        docs = content.decode("utf-8")
    elif filetype == ".docx":
        docs = ""
        doc = Document(filepath)
        for paragraph in doc.paragraphs:
            docs += paragraph.text + "\n"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    doc_chunk=text_splitter.split_text(docs)
    return doc_chunk