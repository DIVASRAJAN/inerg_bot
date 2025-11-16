from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

from sentence_transformers import CrossEncoder

import os

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL","http://localhost:6333")
QDRANT_BUCKET_NAME = os.getenv("QDRANT_BUCKET_NAME","default_bucket")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_gemini():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        google_api_key=GEMINI_KEY,
    )
    return llm

def get_qdrant():
    client = QdrantClient(url=QDRANT_URL,timeout=60)
    return client

def get_embeddings():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY,model="text-embedding-ada-002")
    return embeddings

def get_reranker():
    reranker_model = CrossEncoder("./bge-reranker-v2-m3")
    return reranker_model
