# modules/llm.py
import json
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
    You are a clinical documentation specialist. Given the following SOAP note, generate a concise summary of the patient's history using clear bullet points. Organize the summary under the following consistent headings:

    Chief Complaint

    History of Present Illness

    Past Medical History

    Current Medications

    Allergies

    Family History

    Social History

    Review of Systems (if available)

For each section, extract only relevant and significant details from the SOAP note. Use professional medical terminology and keep each bullet point brief and factual. Do not include information from the Objective, Assessment, or Plan sections unless it directly pertains to the patient’s history.

Format your output as shown below:

text
- Chief Complaint:
  - [Bullet point(s) summarizing the main reason for the visit]

- History of Present Illness:
  - [Bullet point(s) summarizing symptom onset, duration, characteristics, aggravating/relieving factors, etc.]

- Past Medical History:
  - [Bullet point(s) listing relevant medical, surgical, or psychiatric history]

- Current Medications:
  - [Bullet point(s) listing current medications and dosages]

- Allergies:
  - [Bullet point(s) listing medication or other allergies]

- Family History:
  - [Bullet point(s) on relevant family health conditions]

- Social History:
  - [Bullet point(s) on lifestyle factors such as smoking, alcohol, drug use, occupation, living situation]

- Review of Systems:
  - [Bullet point(s) summarizing pertinent positives and negatives from the review of systems]

If any section is not mentioned in the SOAP note, include the heading and write "Not reported."
Do not include any fake information.

    SOAP NOTE (in JSON format):
    {json_str}
    """
    
    response = llm.predict(prompt)
    return response.strip()

def answer_medical_question(question, qa_chain, chat_history=None):
    """
    Answer a medical question using the QA chain and return in a structured JSON format
    
    Args:
        question: The medical question to answer
        qa_chain: The QA chain to use for answering
        chat_history: Optional chat history (default: None)
        
    Returns:
        A dictionary in the format {"answer": "...", "chat_history": [...]}
    """
    if chat_history is None:
        chat_history = []
       
    # Get response from QA chain
    response = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    # Extract answer from response based on its type
    answer = ""
    if isinstance(response, dict):
        # Try to find the answer in common response keys
        for key in ["answer", "result", "content", "text", "response"]:
            if key in response:
                answer = response[key].strip()
                break
        # If no known keys found, convert the whole response to string
        if not answer:
            answer = str(response).strip()
    else:
        # If response is a string or other simple type
        answer = str(response).strip()
    
    # Update chat history with the new Q&A pair
    updated_chat_history = chat_history.copy()
    updated_chat_history.append((question, answer))
    
    # Return in the requested JSON format
    return {
        "answer": answer,
        "chat_history": updated_chat_history
    }

def retrieve_transcript_from_blob(patient_id, connection_string, container_name):
    """
    Retrieve a transcript TXT file from Azure Blob Storage

    Args:
        patient_id: ID of the patient
        connection_string: Azure Storage connection string
        container_name: Name of the blob container

    Returns:
        Tuple of (transcript_text, error_message)
    """
    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Get container client
        container_client = blob_service_client.get_container_client(container_name)
        
        # Define blob name based on patient ID
        blob_name = f"transcript_{patient_id}.txt"
        
        # Get blob client
        blob_client = container_client.get_blob_client(blob_name)
        
        # Check if blob exists
        if not blob_client.exists():
            return None, f"No transcript found for patient ID {patient_id}"
        
        # Download blob content
        download_stream = blob_client.download_blob()
        transcript_text = download_stream.readall().decode('utf-8')
        
        return transcript_text, None
    except Exception as e:
        return None, f"Error retrieving transcript: {str(e)}"


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

    # Clean output
    if soap_note.startswith("```json"):
        soap_note = soap_note[len("```json"):].lstrip()
    elif soap_note.startswith("```"):
        soap_note = soap_note[len("```"):].lstrip()

    if soap_note.endswith("```"):
        soap_note = soap_note[:-3].rstrip()
    
    return soap_note


def save_soap_note_to_blob(patient_id, soap_note, connection_string, container_name):
    """
    Save generated SOAP note to Azure Blob Storage in a 'generated-note' folder.
    
    Args:
        patient_id: ID of the patient
        soap_note: SOAP note text (string or dict)
        connection_string: Azure Storage connection string
        container_name: Name of the blob container
        
    Returns:
        Tuple (True, None) if successful, or (False, error_message)
    """
    try:
        # Convert SOAP note to JSON string if it's a dictionary
        if isinstance(soap_note, dict):
            soap_note_data = json.dumps(soap_note, indent=2)
        else:
            soap_note_data = soap_note
        
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Define blob name in virtual folder
        blob_name = f"generated_soap_note_{patient_id}.json"
        blob_client = container_client.get_blob_client(blob_name)

        # Upload the SOAP note
        blob_client.upload_blob(soap_note_data, overwrite=True)

        return True, None
    except Exception as e:
        return False, f"Error saving SOAP note: {str(e)}"
    