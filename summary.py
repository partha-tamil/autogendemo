import autogen
from typing_extensions import Annotated
import json

# --- IMPORTANT: Replace this with your actual AI Search Integration ---
def ai_search_knowledge_store(query: Annotated[str, "The search query to send to the knowledge store."]) -> str:
    """
    Performs a search on the external AI knowledge store and returns the relevant results.
    This is a MOCKED function. In a real application, connect to your actual AI Search service.
    Returns results in a structured format (e.g., Markdown table, JSON) if possible.
    """
    print(f"\n--- Agent calling AI Search with query: '{query}' ---")

    knowledge_base = {
        "sales_q1_2025": {
            "title": "Sales Report Q1 2025",
            "content": "Total sales for Q1 2025 were $1.5 million. Key contributors were Product X ($800K), Product Y ($400K), and Service Z ($300K). This represents a 15% increase year-over-year. Region APAC showed significant growth.",
            "source": "Internal Sales Database"
        },
        "product_x_features": {
            "title": "Product X Features and Benefits",
            "content": "Product X is our flagship AI-powered analytics platform. Key features include: real-time data processing, predictive modeling, customizable dashboards, and seamless integration with existing CRM systems. Benefits: improved decision-making, operational efficiency, and competitive advantage.",
            "source": "Product Documentation"
        },
        "product_y_features": {
            "title": "Product Y Features",
            "content": "Product Y is our new cloud-based collaboration tool. Features: secure file sharing, video conferencing, task management, and mobile accessibility. Benefits: enhanced team productivity and remote work capabilities.",
            "source": "Product Documentation"
        },
        "market_trends_ai_assistants": {
            "title": "Current Market Trends in AI Assistants",
            "content": "The AI assistant market is experiencing rapid growth driven by advancements in natural language processing and increased demand for automation. Key trends include: hyper-personalization, multimodal capabilities, edge AI integration, and ethical AI considerations.",
            "source": "Industry Research Report"
        },
        "competitors_ai_space": {
            "title": "Top Competitors in AI Assistant Space",
            "content": "Major competitors include companies like Google (Duet AI), Microsoft (Copilot), and OpenAI (ChatGPT Enterprise). They focus on enterprise solutions, integration with existing software ecosystems, and highly specialized vertical applications.",
            "source": "Competitor Analysis Report"
        },
        "default_response": {
            "title": "General Business Information",
            "content": "Our company is a leading technology provider specializing in AI solutions for enterprise clients. We focus on innovation, customer satisfaction, and delivering measurable business value. Specific information might require a more focused query.",
            "source": "Company Profile"
        }
    }

    relevant_results = []
    for key, data in knowledge_base.items():
        if query.lower() in data["title"].lower() or query.lower() in data["content"].lower():
            relevant_results.append(data)
            break

    if not relevant_results:
        if "sales" in query.lower() or "figure" in query.lower():
            if "Q1 2025" in query:
                 relevant_results.append(knowledge_base["sales_q1_2025"])
            else:
                 relevant_results.append(knowledge_base["sales_q1_2025"])
        elif "product" in query.lower() and ("feature" in query.lower() or "spec" in query.lower()):
            relevant_results.append(knowledge_base["product_x_features"])
        elif "market" in query.lower() or "trend" in query.lower():
             relevant_results.append(knowledge_base["market_trends_ai_assistants"])
        elif "competitor" in query.lower() or "rival" in query.lower():
             relevant_results.append(knowledge_base["competitors_ai_space"])
        else:
            relevant_results.append(knowledge_base["default_response"])

    formatted_output = ""
    if relevant_results:
        for res in relevant_results:
            formatted_output += f"### {res['title']}\n"
            formatted_output += f"Source: {res['source']}\n"
            formatted_output += f"{res['content']}\n\n"
    else:
        formatted_output = "No highly relevant information found for your specific query in the knowledge store. Please try a different query or be more specific."

    return formatted_output

# --- 2. Initialize Agents ---
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
    },
)

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. You initiate tasks, review intermediate steps, and decide when the final summary is acceptable. You can approve or reject the final summary. You have access to tools for execution.",
    code_execution_config={"last_n_messages": 1, "work_dir": "coding"},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: "summary complete" in x.get("content", "").lower(),
)

researcher = autogen.AssistantAgent(
    name="Researcher",
    llm_config={"config_list": config_list},
    system_message=(
        "You are a diligent and resourceful researcher. Your primary goal is to gather all necessary "
        "information from the AI knowledge store to answer the user's questions comprehensively. "
        "Use the 'ai_search_knowledge_store' tool to find relevant data. "
        "If you don't find enough information with one query, try rephrasing or using related terms. "
        "Once you believe you have all the required information, inform the Summarizer agent."
    ),
)

summarizer = autogen.AssistantAgent(
    name="Summarizer",
    llm_config={"config_list": config_list},
    system_message=(
        "You are an expert summarizer and report generator. Your task is to take all the information "
        "provided by the Researcher and synthesize it into a clear, concise, and comprehensive report or summary. "
        "Ensure the summary directly addresses the original query from the Admin. "
        "Format the summary professionally using Markdown. "
        "Conclude your summary with the phrase 'SUMMARY COMPLETE' to signal the end of the report."
    ),
)

# --- 3. Register the search function ---
autogen.register_function(
    ai_search_knowledge_store,
    caller=researcher,
    executor=user_proxy,
    name="ai_search_knowledge_store",
    description="Tool to perform a search on the external AI knowledge store. Input is the search query string.",
)

# --- 4. Create Group Chat ---
groupchat = autogen.GroupChat(
    agents=[user_proxy, researcher, summarizer],
    messages=[], # The messages list is initialized here
    max_round=20,
    speaker_selection_method="auto",
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# --- 5. Initiate the Conversation ---
initial_prompt = (
    "Please provide a comprehensive summary report on the current market trends in AI assistants "
    "and identify our top competitors in this space, including their key features. "
    "Researcher, begin by searching the knowledge store for this information."
)

print(f"\n--- Initiating Chat with Prompt: ---\n{initial_prompt}\n")

user_proxy.initiate_chat(
    manager,
    message=initial_prompt,
)

# --- Retrieve all messages from the group chat ---
print("\n--- Retrieving All Messages from Group Chat ---")
all_messages = groupchat.messages

if all_messages:
    for i, message in enumerate(all_messages):
        sender_name = message.get("name", "Unknown Sender")
        message_content = message.get("content", "[No Content]")
        message_role = message.get("role", "N/A") # Role can be 'user', 'assistant', 'tool'

        print(f"--- Message {i+1} ---")
        print(f"Sender: {sender_name} (Role: {message_role})")
        print(f"Content:\n{message_content}")
        print("-" * 40)
else:
    print("No messages found in the group chat history.")


# --- 6. Retrieve the Final Summary Report (as before) ---
print("\n--- Final Summary Report (extracted from last message) ---")
final_summary = None
for msg in reversed(groupchat.messages):
    if msg.get("sender") == summarizer and "summary complete" in msg.get("content", "").lower():
        final_summary = msg["content"].replace("SUMMARY COMPLETE", "").strip()
        break

if final_summary:
    print(final_summary)
else:
    print("Could not find a clear final summary. Review the full chat log for details.")

print("\n--- End of Conversation ---")
