# Import necessary autogen modules
import autogen
import requests # Used for making HTTP requests to Azure AI Search
import json # Used for handling JSON responses

# Define the configuration for the Language Model
# Replace "YOUR_GEMINI_API_KEY" with your actual Gemini API key if you want to run this locally.
# For Canvas environment, leave it as an empty string, Canvas will provide it.
config_list = [
    {
        "model": "gemini-1.5-pro", # You can use other models like "gemini-1.0-pro"
        "api_key": "" # IMPORTANT: Leave this empty for Canvas, it will be auto-populated.
                      # If running locally, replace with your Gemini API key.
    }
]

# --- Azure AI Search Configuration ---
# IMPORTANT: Replace these with your actual Azure AI Search service details.
AZURE_AI_SEARCH_SERVICE_NAME = "YOUR_AZURE_AI_SEARCH_SERVICE_NAME"
AZURE_AI_SEARCH_INDEX_NAME = "YOUR_AZURE_AI_SEARCH_INDEX_NAME"
AZURE_AI_SEARCH_API_KEY = "YOUR_AZURE_AI_SEARCH_API_KEY"
AZURE_AI_SEARCH_API_VERSION = "2023-11-01" # Or the latest version you prefer

# Create an AssistantAgent. This is the AI agent that will process requests.
# It's configured to use the specified LLM.
llm_config = {
    "config_list": config_list,
    "temperature": 0.7, # Adjust temperature for creativity (higher) vs. precision (lower)
    "timeout": 120, # Timeout for API calls in seconds
}
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config,
    system_message="""You are a helpful AI assistant. Your primary goal is to provide accurate and concise information.
    When a user asks for information that requires searching, you should suggest using the 'azure_ai_search' tool.
    Once the search results are provided by the user_proxy, synthesize the information and present it clearly.
    If you need to perform a search, clearly state what you want to search for.
    """,
)

# Create a UserProxyAgent. This agent acts on behalf of the user and can execute code.
# It's configured to automatically reply and use the azure_ai_search tool.
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Set to "ALWAYS" if you want to provide manual input
    max_consecutive_auto_reply=10, # Max number of consecutive auto-replies
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={
        "work_dir": "coding", # Directory for code execution
        "use_docker": False,  # Set to True if you want to use Docker for isolation
    },
    llm_config=llm_config, # User proxy can also use LLM for generating replies
    system_message="""You are a user proxy agent. You can execute code and tools on behalf of the user.
    You have access to the 'azure_ai_search' tool.
    When the assistant asks for a search, you will use the 'azure_ai_search' tool to fetch the content.
    After executing a tool, you must provide the output back to the assistant.
    Always end your response with 'TERMINATE' when the task is complete.
    """,
)

# Register the azure_ai_search tool with the user_proxy agent.
# This makes the 'azure_ai_search' function available for the agent to call.
@user_proxy.register_for_execution()
@assistant.register_for_llm(description="A tool for performing searches using Azure AI Search to fetch relevant content.")
def azure_ai_search(query: str) -> str:
    """
    Performs a search using Azure AI Search for the given query and returns the relevant results.
    """
    if not all([AZURE_AI_SEARCH_SERVICE_NAME, AZURE_AI_SEARCH_INDEX_NAME, AZURE_AI_SEARCH_API_KEY]):
        return "Azure AI Search credentials are not configured. Please update AZURE_AI_SEARCH_SERVICE_NAME, AZURE_AI_SEARCH_INDEX_NAME, and AZURE_AI_SEARCH_API_KEY."

    search_url = (
        f"https://{AZURE_AI_SEARCH_SERVICE_NAME}.search.windows.net/indexes/"
        f"{AZURE_AI_SEARCH_INDEX_NAME}/docs/search?api-version={AZURE_AI_SEARCH_API_VERSION}"
    )
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_AI_SEARCH_API_KEY
    }
    payload = {
        "search": query,
        "queryType": "semantic", # Or "simple" depending on your index configuration
        "semanticConfiguration": "default", # If using semantic search, specify your config name
        "queryLanguage": "en-us",
        "captions": "extractive|highlight-pre-post",
        "answers": "extractive|highlight-pre-post"
    }

    print(f"\n--- User Proxy is performing Azure AI Search for: '{query}' ---")
    try:
        response = requests.post(search_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        search_results = response.json()

        formatted_results = []
        if search_results and "value" in search_results:
            for i, result in enumerate(search_results["value"]):
                # Customize this part based on the fields in your Azure AI Search index
                doc_id = result.get("id", f"Doc {i+1}")
                content = result.get("content", "No content snippet available.")
                title = result.get("title", "No title available.")
                url = result.get("url", "#") # Assuming your index might have a 'url' field

                formatted_results.append(
                    f"Result {i+1} (ID: {doc_id}):\n"
                    f"Title: {title}\n"
                    f"Content: {content}\n"
                    f"URL: {url}\n"
                    f"---"
                )
        else:
            return "No Azure AI Search results found."
        return "\n".join(formatted_results)
    except requests.exceptions.RequestException as e:
        return f"Error during Azure AI Search API call: {e}"
    except json.JSONDecodeError as e:
        return f"Error decoding JSON response from Azure AI Search: {e}"
    except Exception as e:
        return f"An unexpected error occurred during Azure AI Search: {e}"

# Initiate a chat between the user_proxy and the assistant.
# The user_proxy will start by asking a question that requires searching.
print("\n--- Starting Autogen Chat ---")
chat_result = user_proxy.initiate_chat(
    assistant,
    message="What are the key benefits of using Azure AI Search for enterprise data?",
    # message="What is the capital of France?", # Example of a simple question
)

print("\n--- Autogen Chat Finished ---")
