import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

# --- Configuration ---
# IMPORTANT: Replace with your actual Azure OpenAI and Azure AI Search details.
# It is highly recommended to use environment variables or Azure Key Vault for production.

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "YOUR_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "YOUR_GPT_MODEL_DEPLOYMENT_NAME") # e.g., "gpt-4" or "gpt-35-turbo"
AZURE_OPENAI_API_VERSION = "2024-02-01" # Or your specific API version

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "YOUR_AZURE_AI_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "YOUR_AZURE_AI_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "YOUR_AZURE_AI_SEARCH_INDEX_NAME")

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

# --- RAG Functions ---

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
            # Assuming your documents have a 'content' field and a 'title' or 'id' field
            # Adjust field names based on your Azure AI Search index schema
            doc_content = result.get('content', 'No content found')
            doc_title = result.get('title', result.get('id', 'Untitled Document'))
            documents.append({
                "title": doc_title,
                "content": doc_content,
                "score": result['@search.score']
            })
            print(f"  - Found document: '{doc_title}' (Score: {result['@search.score']:.2f})")
        return documents

    except Exception as e:
        print(f"Error during Azure AI Search retrieval: {e}")
        return []

def generate_response_with_context(user_query: str, retrieved_docs: list):
    """
    Generates a response using Azure OpenAI, augmented with retrieved documents.
    """
    print("\nGenerating response with Azure OpenAI...")

    # Construct the system message to guide the LLM
    system_message = (
        "You are a helpful AI assistant that answers questions based ONLY on the provided context. "
        "If the answer cannot be found in the context, politely state that you don't have enough information. "
        "Cite the document titles you used to answer the question, if applicable."
    )

    # Format the retrieved documents as context for the LLM
    context_text = ""
    if retrieved_docs:
        context_text = "Context Documents:\n"
        for i, doc in enumerate(retrieved_docs):
            context_text += f"Document {i+1} (Title: {doc['title']}):\n{doc['content']}\n\n"
    else:
        context_text = "No relevant documents were found to provide context.\n\n"

    # Combine context and user query
    full_prompt = f"{context_text}User Question: {user_query}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": full_prompt}
    ]

    try:
        chat_completion = openai_client.chat.completions.create(
            messages=messages,
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            temperature=0.7, # Adjust for creativity vs. factualness
            max_tokens=800
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        print(f"Error during Azure OpenAI chat completion: {e}")
        return "An error occurred while generating the response."

# --- Main Chat Simulation ---

def main():
    print("--- Azure AI Chat Playground Simulation ---")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye!")
            break

        # Step 1: Retrieve documents from Azure AI Search
        retrieved_documents = retrieve_documents_from_search(user_input)

        # Step 2: Generate response using Azure OpenAI with retrieved context
        ai_response = generate_response_with_context(user_input, retrieved_documents)

        print("\nAI Chatbot:")
        print(ai_response)
        print("-" * 50) # Separator for clarity

if __name__ == "__main__":
    main()
