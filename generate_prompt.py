from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

def generate_prompt_chain():

    prompt_template = """
    Your task is to provide a detailed answer based on the given context. Ensure that your response includes all relevant information. 
    If the answer cannot be determined from the provided context, please respond with "answer is not available in the context" instead of guessing.Context:\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain
