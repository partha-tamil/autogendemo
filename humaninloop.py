import autogen
import json
import os

# --- Configuration ---
# Define the path for the chat history file
CHAT_HISTORY_FILE = "autogen_human_in_loop_chat_history.json"

# Autogen configuration for LLM
# Replace with your actual LLM configuration in OAI_CONFIG_LIST file.
# For example, using OpenAI API key from environment variable
# If you don't have an API key, you can set 'model' to a local LLM or mock it for testing.
try:
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4", "gpt-3.5-turbo"], # Specify the models you want to use
        },
    )
except FileNotFoundError:
    print(f"Error: OAI_CONFIG_LIST not found. Please create this file with your LLM configuration.")
    print("Example OAI_CONFIG_LIST content:")
    print('''
[
    {
        "model": "gpt-4",
        "api_key": "YOUR_OPENAI_API_KEY"
    },
    {
        "model": "gpt-3.5-turbo",
        "api_key": "YOUR_OPENAI_API_KEY"
    }
]
    ''')
    exit() # Exit if config file is missing

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
    print(f"\n--- Saving chat history to {filename} ---")
    try:
        with open(filename, 'w') as f:
            json.dump(messages, f, indent=4)
        print("Chat history saved successfully.")
    except IOError as e:
        print(f"Error saving chat history: {e}")

def load_chat_history(filename=CHAT_HISTORY_FILE):
    """
    Loads chat history from a JSON file.
    Args:
        filename (str): The name of the file to load the history from.
    Returns:
        list: A list of message dictionaries, or an empty list if the file doesn't exist.
    """
    if os.path.exists(filename):
        print(f"\n--- Loading chat history from {filename} ---")
        try:
            with open(filename, 'r') as f:
                messages = json.load(f)
            print(f"Loaded {len(messages)} messages.")
            return messages
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from history file: {e}. Starting fresh.")
            return []
        except IOError as e:
            print(f"Error loading chat history: {e}. Starting fresh.")
            return []
    else:
        print(f"No existing chat history found at {filename}. Starting fresh.")
        return []

# --- AutoGen Agents Setup for Multi-Agent Conversation ---

def create_multi_agents(llm_config):
    """
    Creates and returns AutoGen agents for a multi-agent conversation.
    Includes UserProxyAgent with human_input_mode='ALWAYS' for human-in-loop.
    Args:
        llm_config (dict): LLM configuration for the agents.
    Returns:
        tuple: (user_proxy_agent, groupchat_manager)
    """
    # The UserProxyAgent acts as the human and can provide input
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="ALWAYS", # IMPORTANT: This enables human in the loop
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
        code_execution_config={
            "work_dir": "coding", # Directory for code execution
            "use_docker": False, # Set to True if you have Docker installed and want isolated execution
            # Ensure the 'coding' directory exists or is created by the agent
        },
        system_message="You are the human user. You can provide feedback, approve code, "
                        "or tell the agents to TERMINATE. "
                        "Type 'exit' or 'TERMINATE' to end the conversation, or just press Enter "
                        "to let the agents continue their work.",
    )

    # An assistant agent to help with planning and general tasks
    assistant = autogen.AssistantAgent(
        name="Assistant",
        llm_config=llm_config,
        system_message="You are a helpful AI assistant. You can assist with planning, "
                        "answering questions, and general problem-solving.",
    )

    # A coder agent specialized in writing and debugging code
    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
        system_message="You are a Python programmer. You write and debug Python code. "
                        "Provide code in markdown blocks. If a task requires code execution, "
                        "ask the User to run it and provide output. "
                        "Confirm when a task is completed with the code.",
    )

    # A critic agent to review solutions and provide constructive feedback
    critic = autogen.AssistantAgent(
        name="Critic",
        llm_config=llm_config,
        system_message="You are a critic. Your role is to review the proposed solutions, "
                        "especially code, and identify potential issues, improvements, or errors. "
                        "Provide constructive feedback.",
    )

    # Create a GroupChat to manage the multi-agent conversation
    groupchat = autogen.GroupChat(
        agents=[user_proxy, assistant, coder, critic],
        messages=[], # Messages will be populated by the chat
        max_round=15, # Maximum rounds in the group chat
        # speaker_selection_method="auto", # Auto-selects the next speaker
        # For human-in-loop, often 'round_robin' or a custom function might be useful
        # to ensure the human gets a chance. 'auto' works well with `human_input_mode=ALWAYS`.
        allow_repeat_speaker=False, # Avoid agents talking twice in a row unnecessarily
    )

    # Create a GroupChatManager to orchestrate the group chat
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    return user_proxy, manager

# --- Simulation of Sessions with Human in the Loop ---

def run_multi_agent_session(session_name, initial_message=None):
    """
    Simulates a single multi-agent chat session with human in the loop.
    Loads history, runs chat, and saves history.
    Args:
        session_name (str): Name of the current session (for logging).
        initial_message (str, optional): The first message to start the chat.
    """
    print(f"\n=============================================")
    print(f"--- Starting {session_name} with Human-in-Loop ---")
    print(f"=============================================\n")

    # Load previous chat history
    loaded_history = load_chat_history()

    # Create multi-agents and manager
    user_proxy, manager = create_multi_agents(llm_config)

    # Initialize chat with loaded history
    # The 'message' parameter is used for the very first message of the chat.
    # The 'chat_history' parameter ensures previous messages are loaded.
    # 'clear_history=False' is important to append to the existing history.
    chat_result = user_proxy.initiate_chat(
        manager, # The user_proxy initiates chat with the manager
        message=initial_message,
        clear_history=False, # Keep previous history
        chat_history=loaded_history # Provide the loaded history
    )

    # The chat_history attribute of the chat_result object contains all messages from this conversation
    # (including the loaded history and the new exchanges).
    all_messages = chat_result.chat_history

    # Save the updated chat history for the next session
    save_chat_history(all_messages)

    print(f"\n=============================================")
    print(f"--- {session_name} Finished ---")
    print(f"=============================================\n")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Session 1 ---
    # User starts a complex conversation that might require multiple agents.
    run_multi_agent_session(
        "Session 1: Initial Complex Task",
        initial_message="Hello team! I need help with a Python script. "
                        "First, write a Python function to calculate the Nth Fibonacci number "
                        "using recursion. Then, the Coder should provide the code, and the Critic should review it. "
                        "Make sure to explain any potential issues."
    )

    # --- Session 2 ---
    # User returns for a new conversation.
    # The agents should now remember the previous context and be ready to continue.
    input("\nPress Enter to start Session 2 (should remember previous context and allow intervention)...")
    run_multi_agent_session(
        "Session 2: Continuing Task or New Inquiry",
        initial_message="Hi team! Do you remember the Fibonacci function? Can you now optimize it "
                        "using memoization? Critic, please review the optimized code."
    )

    print("\nDemonstration complete.")
    print(f"Check '{CHAT_HISTORY_FILE}' for the saved messages.")
    print(f"To run again from scratch, delete the '{CHAT_HISTORY_FILE}' file.")
    print("Ensure you have a valid OAI_CONFIG_LIST file with your LLM API keys.")

