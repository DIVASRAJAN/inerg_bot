from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import json,inspect
from contextlib import asynccontextmanager
from api import app_state

from api.main import agent_creation
from api import clients

@asynccontextmanager
async def lifespan(app: FastAPI):
    try :
        app_state.embeddings = clients.get_embeddings()
        app_state.reranker = clients.get_reranker()
        app_state.qdrant = clients.get_qdrant()
        app_state.llm = clients.get_gemini()
        logging.info("App state initialized")
    except Exception:
        logging.exception("Failed to initialize app state")
        raise
    yield


app = FastAPI(
    title="inergBot",
    description="Domain specific information retriever",
    version="1.0.0",
    lifespan=lifespan
)

# === Input and Output Schemas ===

class TopicRequest(BaseModel):
    query: str

class PostResponse(BaseModel):
    query: str
    answer: str
    title: str


# === FastAPI Endpoint ===

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=PostResponse)
async def chat(request: TopicRequest):
    try:
        user_input = request.query

        if inspect.iscoroutinefunction(agent_creation):
            result = await agent_creation(user_input)
        else:
            result = agent_creation(user_input)

        if isinstance(result, str):
            try:
                result_parsed = json.loads(result)
            except json.JSONDecodeError:
                result_parsed = {"answer": result, "title": ""}
        elif isinstance(result, dict):
            result_parsed = result
        # print("userrrr",user_input)
        return PostResponse(
            query=user_input,
            answer=result_parsed.get("answer", "No content fetched."),
            title=result_parsed.get("title", "No title fetched.")
        )

    except Exception as e:
        logging.exception("Error in generating post")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)