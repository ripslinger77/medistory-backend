# summary/__init__.py
import azure.functions as func
import json
import logging
from modules.llm import initialize_llm, summarize_patient_data

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Summary function processed a request.')
    
    try:
        req_body = req.get_json()
        soap_note = req_body.get('soap_note')
        
        if not soap_note:
            return func.HttpResponse(
                json.dumps({"error": "Please provide a SOAP note"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Initialize LLM
        llm = initialize_llm()
        
        # Generate summary
        summary = summarize_patient_data(llm, soap_note)
        
        return func.HttpResponse(
            json.dumps({"summary": summary}),
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error in Summary function: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
