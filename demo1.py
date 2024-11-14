## AutoGen + RAG
## RetrieveUserProxyAgent + AssistantAgent

import chromadb
import autogen
#from autogen import config_list_from_json,UserProxyAgent,AssistantAgent
from autogen import AssistantAgent
#from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

def termination_msg(x):
    return isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()


llm_config = {
        "config_list": [
                {
                        "model": "gpt-4",
                        "api_type": "azure",
                        "api_key": "xxxxxxxxxxx",
                        "base_url":"xxxxxxxxxxx",
                        "api_version":"2024-08-01-preview"
                }
        ]
}


URL = "https://github.com/enrique-ochoa/delete/blob/main/explanations.md"

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="NEVER",
    default_auto_reply="Reply `TERMINATE` if the task is done.",
    max_consecutive_auto_reply=3,
    retrieve_config={
        "task": "code",
        "docs_path": URL,
#        "docs_path": [
#            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
#            "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Research.md",
#        ],
        "chunk_token_size": 1000,
        "model": "gpt-4",
        "client": chromadb.PersistentClient(path="/tmp/db/chromadb"),
        "collection_name": "groupchat",
        "get_or_create": True,
    },
    code_execution_config=False,  # we don't want to execute code in this case.
    system_message="Assistant who has extra content retrieval power for solving difficult problems.",
)

assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    # description="Assistant who can create Groovy business firewall rules or can describe content of Groovy scripts",
    # llm_config={
    #     "timeout": 600,
    #     "cache_seed": 42,
    #     "config_list": config_list,
    # },
)

# RAG AUTOGEN
assistant.reset()

PROBLEM = "Can you create a warranty JSON rule for Contoso Quantum Comfort?"
qa_problem = "Who is the author of FLAML?"
chat_result = boss_aid.initiate_chat(assistant, message=boss_aid.message_generator, problem=PROBLEM)

