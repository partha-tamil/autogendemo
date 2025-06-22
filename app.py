# app.py
import json
import os,autogen
import asyncio
from flask import Flask, request, jsonify
from flask_cors import CORS # Needed to allow requests from your web app (different origin)
from autogenstudio import WorkflowManager
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing


# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://autogenpoc.openai.azure.com/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4") # e.g., "gpt-4" or "gpt-35-turbo"
AZURE_OPENAI_API_VERSION = "2024-02-01" # Or your specific API version

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://autogenpoc-search.search.windows.net")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "")


# --- Initialize Clients ---
try:
    # Initialize Azure OpenAI Client
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )
    print("Azure OpenAI client initialized successfully.")

    # Initialize Azure AI Search Client
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )
    print("Azure AI Search client initialized successfully.")

except Exception as e:
    print(f"Error initializing clients: {e}")
    print("Please ensure your environment variables/configurations are correct.")
    exit()

#LLM Configuration
llm_config = {
"model": "gpt-4",
"api_type": "azure",
"api_version": "2025-01-01-preview",
"base_url": "https://autogenpoc.openai.azure.com/",
"api_key":""
}

#agent definitions
user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="You are an helpful AI assistant.",
    code_execution_config={
        "last_n_messages": 2,
        "work_dir": "groupchat",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    human_input_mode="TERMINATE",
)

data_accumulator_compliance_rules = autogen.AssistantAgent(
    name="data_accumulator_compliance_rules",
    llm_config=llm_config,
    system_message="You are a data retrieval agent to fetch only tax rules. Your role is to query an Azure AI Search index and fetch the relevant chunk of data based on a given user query.",
)

rules_comparator = autogen.AssistantAgent(
    name="rules_comparator",
    llm_config=llm_config,
    system_message="You are a document comparison agent of tax rules. Retrieve the tax rules stored in shared memory and compare it with the document received from user proxy or input. Your task is to analyze two input documents and identify any differences between them. These may include changes in wording, structure, formatting, data values, or semantic meaning. Highlight both exact (verbatim) differences and subtle contextual shifts. Your output should be structured to indicate the type of difference, location within the document, and a clear explanation of the change. Be concise and accurate, and do not make assumptions beyond the provided content. Generate a PDF report.",
)

data_accumulator_judgements = autogen.AssistantAgent(
    name="data_accumulator_judgements",
    llm_config=llm_config,
    system_message="You are a data retrieval agent to fetch only court orders. Your role is to query an Azure AI Search index and fetch the relevant chunk of data based on a given user query.",
)

judgement_analyzer = autogen.AssistantAgent(
    name="judgement_analyzer",
    llm_config=llm_config,
    system_message="You are a document comparison agent of court orders and prepare a report on deviation of tax compliance. Retrieve the court orders stored in shared memory and compare it with the document received from user proxy or input. Your task is to analyze two input documents and identify any differences between them. These may include changes in wording, structure, formatting, data values, or semantic meaning. Highlight both exact (verbatim) differences and subtle contextual shifts. Your output should be structured to indicate the type of difference, location within the document, and a clear explanation of the change. Be concise and accurate, and do not make assumptions beyond the provided content. Generate a PDF report.",
)

prompt_compressor = autogen.AssistantAgent(
    name="prompt compressor",
    llm_config=llm_config,
    system_message="You are AI assitant to compress the data received from AI search to generate input less than 16000 tokens",
)

processor = autogen.GroupChat(agents=[
                                    user_proxy,
                                    data_accumulator_compliance_rules,
                                    rules_comparator,
                                    data_accumulator_judgements,
                                    judgement_analyzer], messages=[], max_round=12)
orchestrator = autogen.GroupChatManager(groupchat=processor, llm_config=llm_config)

# agent configuration
prompt_compressor = autogen.AssistantAgent(
    name="prompt compressor",
    llm_config=llm_config,
    system_message="You are AI assitant to compress the data received from AI search to generate input less than 16000 tokens",
)


groupchat1 = autogen.GroupChat(agents=[user_proxy, prompt_compressor],
                               speaker_selection_method="round_robin",
                               max_round=2,
                               messages=[])
manager1 = autogen.GroupChatManager(groupchat=groupchat1, llm_config=llm_config)

def retrieve_documents_from_search(query_text: str, top_n: int = 3):
    """
    Retrieves relevant documents from Azure AI Search based on the query text.
    """
    print(f"\nSearching Azure AI Search for: '{query_text}'...")
    try:
        # Perform a simple search. For more advanced scenarios, consider semantic or vector search.
        search_results = search_client.search(
            search_text=query_text,
            top=top_n,
            include_total_count=True,
            query_type="simple" # Can be "semantic", "vector", "full", etc. depending on your index
            # For semantic search: query_type="semantic", semantic_configuration_name="my-semantic-config"
            # For vector search: vector_queries=[VectorQuery(vector=embedding_of_query, k_nearest_neighbors=5, fields="contentVector")]
        )

        documents = []
        for result in search_results:
            # --- DEBUGGING STEP: Print the entire result object ---
            print(f"  --- Full search result document: {result}")
            # --- END DEBUGGING STEP ---

            # Assuming your documents have a 'content' field and a 'title' or 'id' field
            # Adjust field names based on your Azure AI Search index schema
            # IMPORTANT: Check the printed 'result' object above to find the correct field for your document's main content.
            # It might be 'text', 'main_content', 'description', etc., instead of 'content'.
            doc_content = result.get('chunk', 'No content found') # <--- **UPDATE THIS FIELD NAME IF NECESSARY**
            doc_title = result.get('title', result.get('id', 'Untitled Document')) # <--- UPDATE THIS FIELD NAME IF NECESSARY

            documents.append({
                "title": doc_title,
                "content": doc_content,
                "score": result['@search.score']
            })
            print(f"  - Found document: '{doc_title}' (Score: {result['@search.score']:.2f})")
            if doc_content == 'No content found':
                print(f"    WARNING: Content for '{doc_title}' was not found. Check your index schema for the correct content field name.")
        return documents

    except Exception as e:
        print(f"Error during Azure AI Search retrieval: {e}")
        return []





#fetch data from knowledge store
def fetch_data_ai_search(searchkey):
    # Replace with your actual values
    endpoint = "https://autogenpoc-search.search.windows.net"
    index_name = "rag-1750581699787"
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

    groupchat = autogen.GroupChat(agents=[user_proxy, prompt_compressor],
                               speaker_selection_method="round_robin",
                               max_round=2,
                               messages=[])
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    chat_result = user_proxy.initiate_chat(
    manager, message= searchmessagesstr ,
    summary_method= 'last_msg',
    summary_prompt = "Please provide a concise, high-level summary of the problem discussed, the solution proposed, and the final outcome of the multi-agent collaboration and decisions. Start with 'Overall, the team collaborated to..."
    )
      
    return chat_result.summary

@app.route('/reverse_string', methods=['POST'])
def reverse_string():
    # Get JSON data from the request
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = data['text']
    reversed_text = input_text[::-1] # Pythonic way to reverse a string

    return jsonify({"original": input_text, "reversed": reversed_text})

@app.route('/run_workflow',methods=['POST'])
def run_workflow():

 
    workflow_path = os.path.join(os.path.dirname(__file__), "workflow.json")
    workflow_manager = WorkflowManager(workflow=workflow_path)
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = data['text']
    
    retrieved_docs = retrieve_documents_from_search(input_text)
    #fetch_data_ai_search(input_text)

    system_message = (
        "You are a helpful AI assistant that answers questions based ONLY on the provided context. "
        "If the answer cannot be found in the context, politely state that you don't have enough information. "
        "Cite the document titles you used to answer the question, if applicable."
    )
   

    #agent_input = f"""Based on the following documents:\n{responsefromaisearch}\n\nAnswer the question: {input_text}"""
    
# Format the retrieved documents as context for the LLM
    context_text = ""
    if retrieved_docs:
        context_text = "Context Documents:\n"
        for i, doc in enumerate(retrieved_docs):
            context_text += f"Document {i+1} (Title: {doc['title']}):\n{doc['content']}\n\n"
    else:
        context_text = "No relevant documents were found to provide context.\n\n"

    # Combine context and user query
    full_prompt = f"{context_text}User Question: {input_text}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": full_prompt}
    ]



    # run the workflow on a task
    
    chat_result = user_proxy.initiate_chat(
    orchestrator, message = full_prompt, 
    summary_method="reflection_with_llm",
    summary_prompt = system_message
    )
    
    return jsonify({'message':chat_result.summary})

if __name__ == '__main__':
    # Run the Flask app on port 5000 (or any other available port)
 app.run(debug=True, port=5000)
