from autogen import UserProxyAgent,AssistantAgent,GroupChat,GroupChatManager
import os, autogen
from mistralai_azure import UserMessage
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


llm_config = {
"model": "gpt-4",
"api_type": "azure",
"api_version": "2025-01-01-preview",
"base_url": "https://autogenpoc.openai.azure.com/",
"api_key":""
}
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="TERMINATE",
)
# coder = autogen.AssistantAgent(
#     name="Coder",
#     llm_config=llm_config,
#     system_message= "You are a highly knowledgeable and resourceful travel planning assistant")
# pm = autogen.AssistantAgent(
#     name="Product_manager",
#     system_message="You are an AI assistant helpful in reviewing the travel plan and optimizing for the best cost.",
#     llm_config=llm_config,
# )
# groupchat = autogen.GroupChat(agents=[user_proxy, coder, pm], messages=[], max_round=12)
# manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# chat_result = user_proxy.initiate_chat(
#     manager, message="Create a travel plan to go to spain this summer", 
#     summary_method="reflection_with_llm",
#     summary_prompt = "Please provide a concise, high-level summary of the problem discussed, the solution proposed, and the final outcome of the multi-agent collaboration and decisions. Start with 'Overall, the team collaborated to..."
# )

prompt_compressor = autogen.AssistantAgent(
    name="prompt compressor",
    llm_config=llm_config,
    system_message="Create a summary from the data received",
)

# Wrap the input in a UserMessage if required by your framework
input_text = (
"In recent years, urban vertical farming has emerged as a revolutionary approach to food production, particularly in densely populated cities where traditional agriculture is impractical. This method involves growing crops in stacked layers, often integrated into buildings or specially designed towers."
)

chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
            }
        ]
    }
]


def fetch_data_ai_search(searchkey):
    # Replace with your actual values
    endpoint = "https://autogenpoc-search.search.windows.net"
    index_name = "rag-1750323268963"
    api_key = ""

    # Create a SearchClient
    search_client = SearchClient(endpoint=endpoint,
                             index_name=index_name,
                             credential=AzureKeyCredential(api_key))

    #Perform a search
    results = search_client.search(searchkey)

    searchmessages = []

    #Print results
    for result in results:
        searchmessages.append(result['chunk'])
    
    searchmessagesstr = str(searchmessages)
   
    return searchmessagesstr



groupchat = autogen.GroupChat(agents=[user_proxy, prompt_compressor],
                               speaker_selection_method="round_robin",
                               max_round=2,
                               messages=[])
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

messages = str(fetch_data_ai_search("What is indirect tax?"))

print(messages)

chat_result = user_proxy.initiate_chat(
    manager, message= messages ,
    summary_method= 'last_msg',
    summary_prompt = "Please provide a concise, high-level summary of the problem discussed, the solution proposed, and the final outcome of the multi-agent collaboration and decisions. Start with 'Overall, the team collaborated to..."
)
    

chat_result1 = user_proxy.initiate_chat(
    manager, message= chat_result.summary.join("What is indirect tax?"),
    summary_method= 'last_msg',
    summary_prompt = "Please provide a concise, high-level summary of the problem discussed, the solution proposed, and the final outcome of the multi-agent collaboration and decisions. Start with 'Overall, the team collaborated to..."
)
    

#print(chat_result1.summary)
