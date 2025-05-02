# generatenote/__init__.py
import azure.functions as func
import json
import logging
import os
from modules.llm import generate_soap_from_transcript, retrieve_transcript_from_blob, save_soap_note_to_blob

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

        # Required input
        patient_id = req_body.get("patient_id")
        if not patient_id:
            return func.HttpResponse(
                json.dumps({"error": "Missing required field: 'patient_id'."}),
                mimetype="application/json",
                status_code=400
            )

        # Get Azure Storage connection details from environment variables
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.environ.get("TRANSCRIPT_CONTAINER", "conversation-transcripts")

        if not connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Missing environment variable: AZURE_STORAGE_CONNECTION_STRING"}),
                mimetype="application/json",
                status_code=500
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

        # Save generated SOAP note
        success, save_error = save_soap_note_to_blob(patient_id, soap_note, connection_string, container_name)
        if not success:
            logging.warning(f"SOAP note was generated but failed to save: {save_error}")

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
