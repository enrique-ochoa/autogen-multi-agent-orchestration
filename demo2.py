## AutoGen + Group Chat + RAG
## RetrieveUserProxyAgent + AssistantAgent1 + AssistantAgent2 

import chromadb
from typing_extensions import Annotated

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


#URL = "https://github.com/enrique-ochoa/delete/blob/main/explanations.md"
URL = "explanations.md"

boss_aid = RetrieveUserProxyAgent(
    name="Boss_Assistant",
    is_termination_msg=termination_msg,
    human_input_mode="ALWAYS",
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
    is_termination_msg=termination_msg,
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
    # description="Assistant who can create Groovy business firewall rules or can describe content of Groovy scripts",
)

BFR1_assistant = AssistantAgent(
    name="BFR1assistant",
    is_termination_msg=termination_msg,
    system_message="As a BFR1.0 assistant, you are responsible for creating and explaining business firewall rules in GROOVY format.",
    llm_config=llm_config,
    description="Assistant who can create Groovy business firewall rules or can describe content of Groovy scripts",
)

BFR2_assistant = AssistantAgent(
    name="BFR2assistant",
    is_termination_msg=termination_msg,
    system_message="As a BFR2.0 assistant, you are responsible for creating and explaining business firewall rules in JSON format.",
    llm_config=llm_config,
    description="Assistant who can create JSON business firewall rules or can describe content of JSON payloads",
)

BFR3_assistant = AssistantAgent(
    name="BFR3assistant",
    is_termination_msg=termination_msg,
    system_message="As a BFR3.0 assistant, you are responsible for creating and explaining business firewall rules in YAML format.",
    llm_config=llm_config,
    description="Assistant who can create YAML business firewall rules or can describe content of YAML scripts",
)

def _reset_agents():
    boss_aid.reset()
    BFR1_assistant.reset()
    BFR2_assistant.reset()
    BFR3_assistant.reset()

# RAG AUTOGEN
assistant.reset()

# MULTI-AGENT ORCHESTRATION
    
#_reset_agents()
groupchat = autogen.GroupChat(
    agents=[boss_aid, BFR1_assistant, BFR2_assistant, BFR3_assistant], 
    messages=[], 
    max_round=10, 
    speaker_selection_method="auto",
    allow_repeat_speaker=False,
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# To check if the rag agent is loading file 
#boss_aid.retrieve_docs(problem="What are the key features of Contoso Quantum Comfort?", n_results=3,search_string="")

#qa_problem = "Who is the author of FLAML?"
#PROBLEM = "Can you create a warranty JSON rule for Contoso Quantum Comfort?"
#PROBLEM = "Can you create a warranty Groovy rule for Contoso Quantum Comfort?"
PROBLEM = "Can you create a warranty YAML rule for Contoso Quantum Comfort?"

# Start chatting with boss_aid as this is the user proxy agent.
boss_aid.initiate_chat(
    manager,
    message=boss_aid.message_generator,
    problem=PROBLEM,
    n_results=3,
)

