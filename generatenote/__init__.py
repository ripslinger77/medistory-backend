# audio2notes/__init__.py
import azure.functions as func
import json
import logging
import os
import tempfile
from modules.llm import transcribe_audio, generate_soap_from_transcript

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Audio2Notes function processed a request.')
    
    try:
        # Check if there's an audio file in the request
        audio_file = req.files.get('audio')
        
        if not audio_file:
            return func.HttpResponse(
                json.dumps({"error": "Please upload an audio file"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Create a temporary directory to store files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded audio file to the temp directory
            audio_path = os.path.join(temp_dir, "audio_file.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_file.read())
            
            # Process audio to transcript
            transcript = transcribe_audio(audio_path, device="cpu")
            
            if transcript is None:
                return func.HttpResponse(
                    json.dumps({"error": "Failed to transcribe audio file"}),
                    mimetype="application/json",
                    status_code=500
                )
            
            # Generate SOAP note from transcript
            soap_note = generate_soap_from_transcript(transcript)
            
            # Return the results
            return func.HttpResponse(
                json.dumps({
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
