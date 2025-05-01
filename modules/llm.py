# modules/llm.py
import json
import whisperx
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from azure.storage.blob import BlobServiceClient
from modules.config import get_azure_openai_client

# Import clients from config
from modules.config import get_azure_openai_client, get_azure_embedding_client

def create_vector_store(text_content, embeddings=None):
    """Create a vector store from text content"""
    if embeddings is None:
        embeddings = get_azure_embedding_client()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_text(text_content)
    return FAISS.from_texts(documents, embeddings)

def create_qa_chain(vectorstore=None, llm=None, memory=None):
    """Create a QA chain for conversational retrieval"""
    if llm is None:
        llm = get_azure_openai_client()
        
    if memory is None:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            return_messages=True
        )
    
    prompt_template = PromptTemplate.from_template("""
    You are a helpful assistant. Use the context below to answer the question.
    If the answer is not explicitly mentioned, say "I don't know."

    Context:
    {context}

    Question: {question}
    Answer:""")
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

def retrieve_soap_note_from_blob(patient_id, connection_string, container_name):
    """
    Retrieve a SOAP note JSON file from Azure Blob Storage
    
    Args:
        patient_id: ID of the patient
        connection_string: Azure Storage connection string
        container_name: Name of the blob container
        
    Returns:
        Tuple of (soap_note_json, error_message)
    """
    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Define blob name based on patient ID
        blob_name = f"soap_note_{patient_id}.json"
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Check if blob exists
        if not blob_client.exists():
            return None, f"No SOAP note found for patient ID {patient_id}"
        
        # Download blob content
        download_stream = blob_client.download_blob()
        soap_note_json = json.loads(download_stream.readall().decode('utf-8'))
        
        return soap_note_json, None
    except Exception as e:
        return None, f"Error retrieving SOAP note: {str(e)}"

def summarize_patient_data(soap_note_json, llm=None):
    """
    Summarize patient data from a SOAP note
    
    Args:
        soap_note_json: SOAP note in JSON format
        llm: LLM client (optional)
        
    Returns:
        Summarized SOAP note
    """
    if llm is None:
        llm = get_azure_openai_client(temperature=0.6, max_tokens=512)
    
    json_str = json.dumps(soap_note_json, indent=2)
    prompt = f"""
    You are a highly skilled and reliable medical professional assistant. Your task is to summarize the following SOAP note, provided in JSON format, using a standardized bullet-point format.

    Instructions:
    - For each section (Subjective, Objective, Assessment, Plan), write **one concise bullet point** summarizing the most important, **clinically relevant** information.
    - **Label each bullet clearly** as S (Subjective), O (Objective), A (Assessment), or P (Plan).
    - Use clear, professional language and **complete sentences**.
    - **Do not infer or assume any information** not explicitly mentioned in the SOAP note. If you are uncertain, say "Not specified."
    - If a section is missing, empty, or lacks meaningful content, **omit that bullet point**.
    - Maintain the order: S, O, A, P.
    - Keep each bullet under 40 words for clarity.

    Strict Output Format (no extra text or explanations):
    S: [Subjective summary]
    O: [Objective summary]
    A: [Assessment summary]
    P: [Plan summary]

    SOAP NOTE (in JSON format):
    {json_str}
    """
    
    response = llm.predict(prompt)
    return response.strip()

def answer_medical_question(question, qa_chain, chat_history=None):
    """Answer a medical question using the QA chain"""
    if chat_history is None:
        chat_history = []
        
    response = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    return response["answer"].strip()

# Audio processing functions adapted from AUDIO1.py
def transcribe_audio(audio_path, device="cpu"):
    """
    Transcribe audio file and assign speaker roles
    
    Args:
        audio_path: Path to the audio file
        device: Device to use for processing (cpu or cuda)
        
    Returns:
        Full transcript with speaker roles
    """
    try:
        print("Transcribing audio...")
        model = whisperx.load_model("medium.en", device=device, compute_type="float32")
        result = model.transcribe(audio_path)
        
        # Align for word-level accuracy
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], model_a, metadata, audio_path, device=device)
        
        # Assign speaker roles based on speaking order
        speaker_ids = list({seg.get("speaker", "speaker_0") for seg in aligned["segments"]})
        speaker_map = {
            speaker_ids[0]: "Doctor",
            speaker_ids[1] if len(speaker_ids) > 1 else speaker_ids[0]: "Patient"
        }
        
        # Apply role mapping
        for seg in aligned["segments"]:
            speaker = seg.get("speaker", "speaker_0")
            seg["role"] = speaker_map.get(speaker, "Unknown")
        
        # Combine into a readable transcript
        full_transcript = "\n".join([f"{seg['role']}: {seg['text']}" for seg in aligned["segments"]])
        
        return full_transcript
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return None

def generate_soap_from_transcript(transcript, llm=None):
    """
    Generate a SOAP note from a transcript using LLM
    
    Args:
        transcript: The transcript text
        llm: LLM client (optional)
        
    Returns:
        Generated SOAP note
    """
    if llm is None:
        llm = get_azure_openai_client(temperature=0.6, max_tokens=1000)
    
    print("Generating SOAP note from transcript...")
    
    prompt = ChatPromptTemplate.from_template("""
    You are a medical documentation specialist. Based on the following transcript of a doctor-patient conversation, generate a comprehensive and detailed SOAP note.

    Requirements:
    Format the output using the standard SOAP structure:
    Subjective (S), Objective (O), Assessment (A), Plan (P).
    Each section (S, O, A, P) must be descriptive, cover all relevant aspects of the dialogue, and contain at least 100 words.
    Expand clinical reasoning, differential diagnoses, diagnostic considerations, and treatment plans as reflected in the conversation.
    Use professional and medically accurate language. Avoid summarizing. Instead, extract every possible clinical detail from the transcript and expand upon it thoughtfully.
    The final SOAP note should be thorough and clearly structured for ease of documentation and clinical interpretation.

    Transcript:
    {transcript}

    Output format:
    - Subjective:
    
    - Objective:
      1. Appearance:
      2. Behavior:
      3. Mood/Affect:
      4. Thought process:
      5. Interaction:
    
    - Assessment:
    
    - Plan:
    """)
    
    chain = LLMChain(llm=llm, prompt=prompt)
    soap_note = chain.run({"transcript": transcript})
    
    return soap_note

def process_audio_to_soap(audio_path, output_dir=None, device="cpu"):
    """
    Complete pipeline to process audio to SOAP note
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save output files (optional)
        device: Device to use for processing (cpu or cuda)
        
    Returns:
        Tuple of (transcript, soap_note)
    """
    # Get transcript
    transcript = transcribe_audio(audio_path, device=device)
    
    if transcript is None:
        return None, None
    
    # Save
