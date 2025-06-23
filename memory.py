import autogen
import json
import os

# --- Configuration ---
# Define the path for the chat history file
CHAT_HISTORY_FILE = "autogen_chat_history.json"

# Autogen configuration for LLM
# Replace with your actual LLM configuration
# For example, using OpenAI API key from environment variable
# If you don't have an API key, you can set 'model' to a local LLM or mock it for testing.
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-3.5-turbo"], # Specify the models you want to use
    },
)

llm_config = {
    "config_list": config_list,
    "temperature": 0.7, # Adjust temperature for creativity/determinism
}

# --- Persistence Functions ---

def save_chat_history(messages, filename=CHAT_HISTORY_FILE):
    """
    Saves the chat history to a JSON file.
    Args:
        messages (list): A list of message dictionaries from the chat.
        filename (str): The name of the file to save the history to.
    """
    print(f"Saving chat history to {filename}...")
    with open(filename, 'w') as f:
        json.dump(messages, f, indent=4)
    print("Chat history saved.")

def load_chat_history(filename=CHAT_HISTORY_FILE):
    """
    Loads chat history from a JSON file.
    Args:
        filename (str): The name of the file to load the history from.
    Returns:
        list: A list of message dictionaries, or an empty list if the file doesn't exist.
    """
    if os.path.exists(filename):
        print(f"Loading chat history from {filename}...")
        with open(filename, 'r') as f:
            messages = json.load(f)
        print(f"Loaded {len(messages)} messages.")
        return messages
    else:
        print(f"No existing chat history found at {filename}. Starting fresh.")
        return []

# --- AutoGen Agents Setup ---

def create_agents(llm_config):
    """
    Creates and returns AutoGen agents.
    Args:
        llm_config (dict): LLM configuration for the agents.
    Returns:
        tuple: (user_proxy_agent, assistant_agent)
    """
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="ALWAYS", # Allows human input to continue the conversation
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
        code_execution_config={
            "work_dir": "coding", # Directory for code execution
            "use_docker": False, # Set to True if you have Docker installed and want isolated execution
        },
    )

    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
        system_message="You are a helpful AI assistant. Please respond to user queries concisely. "
                        "Say 'TERMINATE' when the task is done and you have nothing more to add.",
    )
    return user_proxy, assistant

# --- Simulation of Sessions ---

def run_session(session_name, initial_message=None):
    """
    Simulates a single chat session.
    Loads history, runs chat, and saves history.
    Args:
        session_name (str): Name of the current session (for logging).
        initial_message (str, optional): The first message to start the chat.
    """
    print(f"\n--- Starting {session_name} ---")

    # Load previous chat history
    loaded_history = load_chat_history()

    # Create agents
    user_proxy, assistant = create_agents(llm_config)

    # Initialize chat with loaded history
    # The 'message' parameter is used for the very first message of the chat.
    # The 'chat_history' parameter ensures previous messages are loaded.
    # 'clear_history=False' is important to append to the existing history.
    chat_result = user_proxy.initiate_chat(
        assistant,
        message=initial_message,
        clear_history=False, # Keep previous history
        chat_history=loaded_history # Provide the loaded history
    )

    # The chat_history attribute of the chat_result object contains all messages from this conversation
    # (including the loaded history and the new exchanges).
    all_messages = chat_result.chat_history

    # Save the updated chat history for the next session
    save_chat_history(all_messages)

    print(f"--- {session_name} Finished ---")

# --- Main Execution ---
if __name__ == "__main__":
    # --- Session 1 ---
    # User starts a conversation.
    # The agents will not have any prior context unless loaded from the file.
    run_session(
        "Session 1: Initial Conversation",
        initial_message="Hello, Assistant! My name is Alice. Can you tell me a fun fact about pandas?"
    )

    # --- Session 2 ---
    # User returns for a new conversation.
    # The agents should now remember "Alice" and the previous context.
    input("\nPress Enter to start Session 2 (should remember Alice and pandas fact)...")
    run_session(
        "Session 2: Continuing Conversation",
        initial_message="Hi again! What was that fact you told me about? And what else do you know about their diet?"
    )

    print("\nDemonstration complete. Check 'autogen_chat_history.json' for the saved messages.")
    print("To run again from scratch, delete the 'autogen_chat_history.json' file.")
