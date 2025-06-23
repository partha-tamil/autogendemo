import autogen
import json
import os
from typing import Dict, List, Union

# --- Configuration ---
# Replace with your actual OpenAI API key or other LLM configuration
llm_config = {
    "config_list": [
        {
            "model": "gpt-4", # Or "gpt-3.5-turbo", etc.
            "api_key": os.environ.get("OPENAI_API_KEY"), # Ensure OPENAI_API_KEY is set in your environment variables
        }
    ]
}

# --- File Paths for Storing History ---
CHAT_HISTORY_FILE = "autogen_group_chat_history.json"

# --- Helper Functions ---

def save_chat_history(chat_messages: List[Dict], filename: str):
    """Saves the chat history (list of message dicts) to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(chat_messages, f, indent=4)
        print(f"\n--- Chat history saved to {filename} ---")
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_chat_history(filename: str) -> List[Dict]:
    """Loads chat history (list of message dicts) from a JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                history = json.load(f)
            print(f"\n--- Chat history loaded from {filename} ---")
            return history
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filename}: {e}")
            return []
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return []
    else:
        print(f"No chat history file found at {filename}. Starting fresh.")
        return []

# --- Main Demonstration ---

def run_demonstration():
    print("--- Starting AutoGen Group Chat Memory Persistence Demonstration ---")

    # --- PART 1: Initial Group Chat Conversation ---
    print("\n\n--- PART 1: Initial Group Chat Conversation ---")
    print("Agents will discuss a user preference and a task.")

    # Define agents for the initial group chat
    user_proxy_group = autogen.UserProxyAgent(
        name="User_Proxy",
        system_message="A human user proxy. You can ask questions and give instructions.",
        llm_config=llm_config,
        human_input_mode="NEVER", # Set to "ALWAYS" for interactive input
        code_execution_config=False,
    )

    product_manager = autogen.AssistantAgent(
        name="Product_Manager",
        system_message="You are a product manager. You collect requirements and assign tasks. You remember user preferences.",
        llm_config=llm_config,
        code_execution_config=False,
    )

    engineer = autogen.AssistantAgent(
        name="Engineer",
        system_message="You are an engineer. You provide technical solutions and estimates.",
        llm_config=llm_config,
        code_execution_config=False,
    )

    # Create the GroupChat
    groupchat_initial = autogen.GroupChat(
        agents=[user_proxy_group, product_manager, engineer],
        messages=[], # Start with an empty message list for the group
        max_round=10,
        speaker_selection_method="auto"
    )

    # Create the GroupChatManager to orchestrate the initial chat
    manager_initial = autogen.GroupChatManager(
        groupchat=groupchat_initial,
        llm_config=llm_config
    )

    print("\nUser_Proxy: Our new project codename is 'Phoenix'. Also, I prefer all technical documentation to be concise.")
    initial_group_chat_result = user_proxy_group.initiate_chat(
        manager_initial,
        message="Our new project codename is 'Phoenix'. Also, I prefer all technical documentation to be concise.",
        clear_history=False # Important for group chats, don't clear manager's history
    )

    # Let the agents have a small discussion
    print("\nUser_Proxy: Product_Manager, can you outline the initial steps for Phoenix?")
    user_proxy_group.send(
        "Product_Manager, can you outline the initial steps for Phoenix?",
        recipient=manager_initial,
        request_reply=True
    )
    # The agents should discuss and include "Phoenix" and "concise documentation" in their context.

    # --- Save the group chat history ---
    # The GroupChatManager's internal groupchat.messages holds the conversation history
    save_chat_history(manager_initial.groupchat.messages, CHAT_HISTORY_FILE)


    # --- PART 2: New Group Chat Session - Load and Use History ---
    print("\n\n--- PART 2: New Group Chat Session - Load and Use History ---")
    print("Simulating a new session. Agents should recall 'Phoenix' and documentation preference.")

    # Load the previously saved history
    loaded_chat_history = load_chat_history(CHAT_HISTORY_FILE)

    # Initialize new agents for the "new session"
    # Important: Agents in a group chat also need to receive the history
    user_proxy_new_session = autogen.UserProxyAgent(
        name="User_Proxy", # Keep name consistent for agent identity across sessions
        system_message="A human user proxy for a new session.",
        llm_config=llm_config,
        human_input_mode="NEVER",
        code_execution_config=False,
        chat_messages={manager_initial.name: loaded_chat_history} # Inject loaded history for user_proxy in relation to manager
    )

    product_manager_new_session = autogen.AssistantAgent(
        name="Product_Manager",
        system_message="You are a product manager. You collect requirements and assign tasks. You remember user preferences.",
        llm_config=llm_config,
        code_execution_config=False,
        chat_messages={user_proxy_new_session.name: loaded_chat_history} # Inject loaded history for product_manager in relation to user_proxy
    )

    engineer_new_session = autogen.AssistantAgent(
        name="Engineer",
        system_message="You are an engineer. You provide technical solutions and estimates.",
        llm_config=llm_config,
        code_execution_config=False,
        chat_messages={user_proxy_new_session.name: loaded_chat_history} # Inject loaded history for engineer in relation to user_proxy
    )
    # Note: For simplicity, we are injecting the full history into each agent's chat_messages related to the user_proxy.
    # In more complex scenarios, you might need to filter or summarize the history per agent.

    # Create a new GroupChat and Manager for the new session
    groupchat_new_session = autogen.GroupChat(
        agents=[user_proxy_new_session, product_manager_new_session, engineer_new_session],
        messages=loaded_chat_history, # Pass the loaded history to the groupchat itself
        max_round=10,
        speaker_selection_method="auto"
    )

    manager_new_session = autogen.GroupChatManager(
        groupchat=groupchat_new_session,
        llm_config=llm_config
    )

    # Now, initiate a new chat, asking a question that relies on the loaded history
    print("\nUser_Proxy: Engineer, what's our current understanding of the 'Phoenix' project scope, and remember my documentation preference.")
    final_group_chat_result = user_proxy_new_session.initiate_chat(
        manager_new_session,
        message="Engineer, what's our current understanding of the 'Phoenix' project scope, and remember my documentation preference.",
        clear_history=False # Do not clear the loaded history
    )

    # Observe the responses to see if they recall "Phoenix" and "concise" documentation.
    # You might need to check the full chat history of the result to see how agents interact.
    print(f"\n--- Final Group Chat Result (last message): ---\n{final_group_chat_result.last_message['content']}")

    print("\n--- Demonstration Complete ---")

if __name__ == "__main__":
    run_demonstration()

    # Optional: Clean up the history file after demonstration
    # if os.path.exists(CHAT_HISTORY_FILE):
    #     os.remove(CHAT_HISTORY_FILE)
    #     print(f"Cleaned up {CHAT_HISTORY_FILE}")
