# summary/__init__.py
import azure.functions as func
import json
import logging
import os
from modules.llm import retrieve_soap_note_from_blob, summarize_patient_data

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Summary function processed a request.')
    
    try:
        # Get patient_id from the request
        patient_id = req.params.get('patient_id')
        
        if not patient_id:
            req_body = req.get_json()
            patient_id = req_body.get('patient_id')
        
        if not patient_id:
            return func.HttpResponse(
                json.dumps({"error": "Please provide a patient_id"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Get Azure Storage connection details from environment variables
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.environ.get("SOAP_NOTES_CONTAINER", "soap-notes")
        
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
        
        # Generate summary
        summary = summarize_patient_data(soap_note_json)
        
        return func.HttpResponse(
            json.dumps({
                "patient_id": patient_id,
                "summary": summary
            }),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in Summary function: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
