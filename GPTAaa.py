import logging
import os
from llm_config import config_list

from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from openai import OpenAI

llm_config = {
    "config_list": config_list,
}


assistant_id = None
#os.environ("ASSISTANT_ID", None)
#config_list = config_list_from_json("OAI_CONFIG_LIST")
llm_config = {
    "config_list": config_list,
}
assistant_config = {
    # define the openai assistant behavior as you need
    'You are reading restaurant reviews from a text file'
}
oai_agent = GPTAssistantAgent(
    name="oai_agent",
    instructions="I'm an openai assistant running in autogen",
    llm_config=llm_config,
    #assistant_config=assistant_config,
)


assistant_config = {
    "tools": [
        {"type": "file_search"},
    ],
    "tool_resources": {
        "file_search": {
            "vector_store_ids": ["$vector_store.id"]
        }
    }
}

client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)

# Step 1: Create a Vector Store
vector_store = client.beta.vector_stores.create(name="Restaurant_reviews")
print("Vector Store created:", vector_store.id)  # This is your vector_store.id

# Step 2: Prepare Files for Upload
file_paths = ["D:/Witold/Documents/Computing/LLMAgentsOfficial/restaurant-data.txt"]
file_streams = [open(path, "rb") for path in file_paths]

# Step 3: Upload Files and Add to Vector Store (with status polling)
file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
    vector_store_id=vector_store.id, files=file_streams
)

# Step 4: Verify Completion (Optional)
print("File batch status:", file_batch.status)
print("Uploaded file count:", file_batch.file_counts)
