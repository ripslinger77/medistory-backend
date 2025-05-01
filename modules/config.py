# modules/config.py
import os
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings

# Environment variables for API keys and endpoints
# These can be set in your deployment environment
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
EMBEDDING_DEPLOYMENT_NAME = os.environ.get("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small")
EMBEDDING_API_VERSION = os.environ.get("EMBEDDING_API_VERSION", "2024-02-01")

# Fallback values if environment variables are not set
# Note: In production, you should always use environment variables
if not AZURE_OPENAI_API_KEY:
    AZURE_OPENAI_API_KEY = "your-default-key-for-development"
    print("Warning: Using default API key. Set AZURE_OPENAI_API_KEY environment variable in production.")

if not AZURE_OPENAI_ENDPOINT:
    AZURE_OPENAI_ENDPOINT = "https://your-default-endpoint.cognitiveservices.azure.com/"
    print("Warning: Using default endpoint. Set AZURE_OPENAI_ENDPOINT environment variable in production.")

# Client initialization functions
def get_azure_openai_client(temperature=0.7, max_tokens=512):
    """
    Returns an initialized Azure OpenAI client
   
    Args:
        temperature: Controls randomness in responses (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
       
    Returns:
        An initialized AzureChatOpenAI client
    """
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=temperature,
        max_tokens=max_tokens
    )

def get_azure_embedding_client():
    """
    Returns an initialized Azure OpenAI embeddings client
    
    Returns:
        An initialized AzureOpenAIEmbeddings client
    """
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        deployment=EMBEDDING_DEPLOYMENT_NAME,
        openai_api_key=AZURE_OPENAI_EMBEDDING_KEY,
        openai_api_version=EMBEDDING_API_VERSION,
        chunk_size=1000
    )

# Singleton instances for common use cases
default_llm_client = get_azure_openai_client()
default_embedding_client = get_azure_embedding_client()
