import json

from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api.clients import get_qdrant,QDRANT_BUCKET_NAME,get_embeddings

import logging
logging.basicConfig(level=logging.INFO)

client = get_qdrant()
embeddings = get_embeddings()

try:
    existing = client.get_collection(collection_name=QDRANT_BUCKET_NAME)
    logging.info("Qdrant collection '%s' already exists.", QDRANT_BUCKET_NAME)
except Exception:
    logging.info("Creating Qdrant collection '%s'.", QDRANT_BUCKET_NAME)
    client.create_collection(
        collection_name=QDRANT_BUCKET_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_BUCKET_NAME,
    embedding=embeddings,
)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=100,
    separators ="\n\n"
)


# with open("oil_and_gas_corpus.jsonl", "r") as f:
#     all_data = json.load(f)

all_data = []
with open("data_ingestion/oil_and_gas_corpus.jsonl", "r") as f:
    for line in f:
        all_data.append(json.loads(line))

for item in all_data:   
    text = item.get("content")
    tags = item.get("tags")
    id = item.get("id")
    title = item.get("title")
    info = {
        "tags": tags,
        "id": id,
        "title" : title
    }
    

    texts = text_splitter.split_text(text)
    docs=text_splitter.create_documents(texts=texts,metadatas=[info]*len(texts))
    vector_store.add_documents(docs)
