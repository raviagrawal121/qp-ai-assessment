
def generate_response_answer(vector_db,user_question,chain):
    docs = vector_db.similarity_search(user_question)
    response = chain({"input_documents":docs, "question": user_question}, return_only_outputs=True)
    return response