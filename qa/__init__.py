# qa/__init__.py
import azure.functions as func
import json
import logging
import os
from modules.llm import (
    create_vector_store, 
    create_qa_chain, 
    answer_medical_question,
    retrieve_soap_note_from_blob
)

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('QA function processed a request.')
    
    try:
        req_body = req.get_json()
        question = req_body.get('question')
        chat_history = req_body.get('chat_history', [])
        
        patient_id = req.params.get('patient_id')
        
        if not patient_id:
            req_body = req.get_json()
            patient_id = req_body.get('patient_id')
        
        if not patient_id or not question:
            return func.HttpResponse(
                json.dumps({"error": "Please provide a patient_id and/or question"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Get Azure Storage connection details from environment variables
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.environ.get("SOAP_NOTES_CONTAINER", "patient-history")
        
        if not connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Azure Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500
            )
        
        # Retrieve SOAP note from Blob Storage
        soap_note_json, error = retrieve_soap_note_from_blob(patient_id, connection_string, container_name)

        if error:
            return func.HttpResponse(
                json.dumps({"error": error}),
                mimetype="application/json",
                status_code=404 if "No SOAP note found" in error else 500
            )
        
        # Initialize LLM and embeddings
        # llm = initialize_llm()
        # embeddings = initialize_embeddings()
        
        # Create vector store and QA chain
        vectorstore = create_vector_store(text_content= json.dumps(soap_note_json))
        qa_chain = create_qa_chain(vectorstore= vectorstore)
        
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
