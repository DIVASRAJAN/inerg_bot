import json

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333",timeout=60)

from langchain_openai import OpenAIEmbeddings,OpenAI
OPENAI_API_KEY="sk -xxxxx"
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY,model="text-embedding-ada-002")

client.create_collection(
    collection_name="name",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="name",
    embedding=embeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load example document

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=2000,
    chunk_overlap=100,
    separators ="/n/n"
)


with open("tesst.json", "r") as f:
    all_data = json.load(f)

for item in all_data:   
    pdf_text = item.get("page_content")
    key_words = item.get("key_words")
    page_number = item.get("page_number")
    title = item.get("title")
    pdf_info = {
        "key_words": key_words,
        "page_number": page_number,
        "title" : title
    }
    

# print("pdf_info is :",pdf_info)
    texts = text_splitter.split_text(pdf_text)
    # for text in texts:
    docs=text_splitter.create_documents(texts=texts,metadatas=[pdf_info]*len(texts))
# for doc in docs:
    vector_store.add_documents(docs)




def get_response(user_prompt,context):
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        # "response_mime_type": "application/json",
        # "response_schema": list[Response]
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    response = model.generate_content([f"{user_prompt} >>>>> {context}"])
    return(response.text)

client = ChatOpenAI(model=model_clean, **config)