
# inerg_bot

Domain-specific information retriever using Qdrant, LangChain wrappers and a Google Gemini LLM.

## usage

```
git clone https://github.com/DIVASRAJAN/inerg_bot.git

git checkout dev
```

download model from the drive and place it inside the folder (bge-reranker-v2-m3)

model_link :
```
https://drive.google.com/file/d/1Eju09sIHoYt0CEBrR1xEmFzV5SzMvv0g/view?usp=sharing
```


## Features
- Ingests JSONL corpus, splits text into chunks and stores embeddings in Qdrant.
- FastAPI service exposing a `/chat` endpoint that queries the agent pipeline.
- Uses Google Gemini (via langchain_google_genai), OpenAI embeddings, and a Cross-Encoder reranker.

## Requirements
- Python 3.10+ (project tested on 3.13)
- Qdrant running and reachable
- GPU/CPU resources for reranker model (sentence-transformers)
- A virtualenv or similar
- uv as package manager (optional)

## Environment variables
Create a `.env` file or export these variables in your environment:

- GEMINI_API_KEY — Google Gemini API key
- OPENAI_API_KEY — OpenAI API key for embeddings
- QDRANT_URL — Qdrant endpoint, default `http://localhost:6333`
- QDRANT_BUCKET_NAME — collection name

## Setup

1. Create and activate a virtual environment:
   uv sync  (optional)

   source .venv/bin/activate

2. Populate Qdrant (ingest corpus)
   - Place your corpus at `data_ingestion/oil_and_gas_corpus.jsonl` (JSONL with fields: id, title, tags, content).
   - Run the ingestion script:
     python data_ingestion/data_ingestion.py

   The script will create the Qdrant collection (if missing), split documents and upload vectors.

## Running the API

Start the FastAPI app (development):uvicorn api.endpoints:app --reload --host 0.0.0.0 --port 8000

Endpoints:
- GET /health — service health check
- POST /chat — body: {"query": "your question"}  
  Response: {"query": "...", "answer":"...", "title":"..."}

Notes:
- The app initializes heavy clients (LLM, embeddings, Qdrant, reranker) at startup using a lifespan handler. Ensure the server process has access to environment variables.

- 
```
uvicorn api.endpoints:app 
```
you can access the swagger using 
```
http://127.0.0.1:8000/docs
```


## project structure

```
inerg_bot/
├── .env
├── README.md
├── requirements.txt
├── .venv/  
├── docker-compose.yml
├── pyproject.toml                  
├── data_ingestion/
│   ├── data_ingestion.py
│   └── oil_and_gas_corpus.jsonl
├── bge-reranker-v2-m3/
└── api/
    ├── tools.py
    ├── app_state.py
    ├── clients.py
    ├── endpoints.py
    ├── main.py
    └── prompt_manager.yaml                
```
## main.py
Agent creation logic 

## tools.py
Tools for agent creation, data retrieval and re-ranking logic

## clients.py
Instances are creating here like embeddings, re-rankings etc

## bge-reranker-v2-m3
re-ranking model

## data_ingestion
Data ingestion pipeline to vector DB

## docker-comopose.yml
Qdrant vector DB initiation


## TO_DO

UI creation
