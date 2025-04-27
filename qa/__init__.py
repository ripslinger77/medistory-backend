# qa/__init__.py
import azure.functions as func
import json
import logging
from modules.llm import (
    initialize_llm, 
    initialize_embeddings, 
    create_vector_store, 
    create_qa_chain, 
    answer_medical_question
)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('QA function processed a request.')
    
    try:
        req_body = req.get_json()
        patient_data = req_body.get('patient_data')
        question = req_body.get('question')
        chat_history = req_body.get('chat_history', [])
        
        if not patient_data or not question:
            return func.HttpResponse(
                json.dumps({"error": "Please provide patient_data and question"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Initialize LLM and embeddings
        llm = initialize_llm()
        embeddings = initialize_embeddings()
        
        # Create vector store and QA chain
        vectorstore = create_vector_store(json.dumps(patient_data), embeddings)
        qa_chain = create_qa_chain(llm, vectorstore)
        
        # Answer the question
        answer = answer_medical_question(question, qa_chain, chat_history)
        
        return func.HttpResponse(
            json.dumps({"answer": answer}),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in QA function: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
