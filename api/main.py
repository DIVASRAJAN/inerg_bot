from langgraph.prebuilt import create_react_agent
import logging
import json

import yaml
import logging, json
from langgraph.checkpoint.memory import MemorySaver

from api import app_state
from api.tools import clean_json_response,parameter_extraction,data_retrieval,load_yaml



data = load_yaml("api/prompt_manager.yaml")
system_prompt = data["system_prompt"]["v1"]

memory =MemorySaver ()

def agent_creation(user_query):
    
    agent = create_react_agent(
        model=app_state.llm,
        tools=[parameter_extraction,data_retrieval],
        prompt=system_prompt,
        checkpointer = memory
    )
    app = agent
    logging.info("Agent created successfully.")
    response = app.invoke({"messages": [{"role": "user", "content": user_query}]},
                          config = {
                              "recursion_limit":8,
                              "configurable":{"thread_id": "default"}})
    # result = response["messages"][-1].content
    # print("responseeeee is",response)
    result = response["messages"][-1].content[0]["text"]
    print("result is ",result)
    new_result = clean_json_response(result)
    new_result = json.loads(new_result)
    return new_result