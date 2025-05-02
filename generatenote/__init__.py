# audio2notes/__init__.py
import azure.functions as func
import json
import logging
from modules.llm import generate_soap_from_transcript, retrieve_transcript_from_blob

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Audio2Notes function (Blob Transcript to SOAP) processed a request.')
    
    try:
        # Parse JSON body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                json.dumps({"error": "Invalid JSON body."}),
                mimetype="application/json",
                status_code=400
            )

        # Required inputs
        patient_id = req_body.get("patient_id")
        connection_string = req_body.get("connection_string")
        container_name = req_body.get("container_name")

        if not all([patient_id, connection_string, container_name]):
            return func.HttpResponse(
                json.dumps({"error": "Missing one or more required fields: 'patient_id', 'connection_string', 'container_name'."}),
                mimetype="application/json",
                status_code=400
            )

        # Retrieve transcript from blob
        transcript, error = retrieve_transcript_from_blob(patient_id, connection_string, container_name)
        if error:
            return func.HttpResponse(
                json.dumps({"error": error}),
                mimetype="application/json",
                status_code=404
            )

        # Generate SOAP note
        soap_note = generate_soap_from_transcript(transcript)

        # Return results
        return func.HttpResponse(
            json.dumps({
                "patient_id": patient_id,
                "transcript": transcript,
                "soap_note": soap_note
            }),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error in Audio2Notes function: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
