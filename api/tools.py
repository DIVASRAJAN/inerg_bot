import re
import json,yaml

from langchain_qdrant import QdrantVectorStore
from langchain_core.messages import HumanMessage
from api import app_state
from api.clients import QDRANT_BUCKET_NAME

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data

data = load_yaml("api/prompt_manager.yaml")
parameter_prompt = data["parameter_prompt"]["v1"]


def clean_json_response(response):
    # Remove markdown-like code fences (``` or ''') and optional json label
    response = re.sub(r"^[`']{3}json\s*|[`']{3}$", "", response.strip(), flags=re.IGNORECASE)

    # Attempt to fix missing quotes on string values (basic case)
    response = re.sub(r'("post"\s*:\s*)([a-zA-Z0-9_]+)', r'\1"\2"', response)

    return response

def build_or_filter(tags=None, id_=None, title=None):
    conditions = []

    if tags:
        conditions.append({"key": "tags", "match": {"value": tags}})
    
    if id:
        conditions.append({"key": "id", "match": {"value": id_}})
    
    if title:
        conditions.append({"key": "title", "match": {"value": title}})

    return {
        "should": conditions,
        "minimum_should_match": 1
    }


def parameter_extraction(user_query):
    """this tool should be the first,Extract parameters from the query for better filtering the data"""
    print("inside parameter")
    response = app_state.llm.invoke([
        HumanMessage(content=parameter_prompt + " user query is " + user_query)
        ])
    # result = response["messages"][-1].content
    result = response.content
    print("parameter response",result)
    new_result = clean_json_response(result)
    new_result = json.loads(new_result)
    return new_result


def data_retrieval(user_query, filter_keys):
    """Retrieve + rerank documents based on user query and the filter keys.
    Args :
     user_query : user input
     filter_keys : response of parameter_extraction tool. used to add filtering on the DB

    """

    print("inside data retrieval")

    tags = filter_keys.get("tags")
    id_ = filter_keys.get("id")
    title = filter_keys.get("title")
    vs = QdrantVectorStore(
        client=app_state.qdrant,
        collection_name=QDRANT_BUCKET_NAME,
        embedding=app_state.embeddings,
    )
    filter = build_or_filter(tags,id_,title)
    try:
        similar = vs.similarity_search_with_score(user_query, k=10, filter=filter)
    except:
        similar = vs.similarity_search_with_score(user_query, k=10)

    documents = [doc for doc, score in similar]
    print("qdrant number ",len(documents))

    pairs = [(user_query, doc.page_content) for doc in documents]
    scores = app_state.reranker.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    top_k = 3
    final_docs = [doc for doc, score in ranked[:top_k]]
    print("final docs ",len(final_docs))

    return final_docs
