import autogen
import json
import os
from typing import Dict, List

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
CHAT_HISTORY_FILE = "autogen_chat_history.json"

# --- Helper Functions ---

def save_chat_history(agent: autogen.ConversableAgent, filename: str):
    """Saves the chat history of a given agent to a JSON file."""
    try:
        # AutoGen's chat_messages attribute stores the conversation.
        # It's a list of dictionaries, which is easily serializable to JSON.
        history = agent.chat_messages[autogen.ALL_DIR] # Access the full history for this agent
        with open(filename, "w") as f:
            json.dump(history, f, indent=4)
        print(f"\n--- Chat history saved to {filename} ---")
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_chat_history(filename: str) -> Dict[str, List[Dict]]:
    """Loads chat history from a JSON file."""
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                history = json.load(f)
            print(f"\n--- Chat history loaded from {filename} ---")
            # The loaded history needs to be in the format expected by agent.chat_messages
            # which is typically {recipient_name: [messages]} or {autogen.ALL_DIR: [messages]}
            return {autogen.ALL_DIR: history} # Assuming we saved the ALL_DIR history
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filename}: {e}")
            return {}
        except Exception as e:
            print(f"Error loading chat history: {e}")
            return {}
    else:
        print(f"No chat history file found at {filename}. Starting fresh.")
        return {}

# --- Main Demonstration ---

def run_demonstration():
    print("--- Starting AutoGen Memory Persistence Demonstration ---")

    # --- PART 1: Initial Conversation ---
    print("\n\n--- PART 1: Initial Conversation ---")
    print("Agent will learn your favorite color.")

    user_proxy_initial = autogen.UserProxyAgent(
        name="User_Proxy_Initial",
        system_message="A human user proxy.",
        llm_config=llm_config,
        human_input_mode="NEVER", # Set to "ALWAYS" for interactive input
        code_execution_config=False,
    )

    assistant_initial = autogen.AssistantAgent(
        name="Assistant_Initial",
        system_message="You are a helpful assistant. You remember facts about the user.",
        llm_config=llm_config,
        code_execution_config=False,
    )

    # First turn: User states a fact
    print("\nUser_Proxy_Initial: My favorite color is blue.")
    initial_chat_result = user_proxy_initial.initiate_chat(
        assistant_initial,
        message="My favorite color is blue.",
        max_turns=1 # Just one turn to establish the fact
    )

    # Second turn in the same conversation: ask a question relying on the fact
    print("\nUser_Proxy_Initial: What did I say was my favorite color?")
    follow_up_result = user_proxy_initial.send(
        "What did I say was my favorite color?",
        recipient=assistant_initial,
        request_reply=True
    )
    # The assistant should recall "blue" from its current session's chat history.
    print(f"\nAssistant_Initial (response): {follow_up_result.last_message['content']}")

    # --- Save the chat history ---
    # We save the history from the user_proxy_initial, which contains the full conversation.
    save_chat_history(user_proxy_initial, CHAT_HISTORY_FILE)


    # --- PART 2: New Session - Load and Use History ---
    print("\n\n--- PART 2: New Session - Load and Use History ---")
    print("Simulating a new session. Agent should recall the favorite color.")

    # Load the previously saved history
    loaded_history = load_chat_history(CHAT_HISTORY_FILE)

    # Initialize new agents for the "new session"
    user_proxy_new = autogen.UserProxyAgent(
        name="User_Proxy_New",
        system_message="A human user proxy for a new session.",
        llm_config=llm_config,
        human_input_mode="NEVER", # Set to "ALWAYS" for interactive input
        code_execution_config=False,
        chat_messages=loaded_history # Inject the loaded history here
    )

    assistant_new = autogen.AssistantAgent(
        name="Assistant_New",
        system_message="You are a helpful assistant. You remember facts about the user from previous sessions.",
        llm_config=llm_config,
        code_execution_config=False,
        chat_messages=loaded_history # Inject the loaded history here
    )

    # Start a new conversation, but with the loaded history already in place
    # The 'message' here is the *new* message to start this session.
    print("\nUser_Proxy_New: Remind me, what is my favorite color from our last chat?")
    final_result = user_proxy_new.initiate_chat(
        assistant_new,
        message="Remind me, what is my favorite color from our last chat?",
        max_turns=1,
        clear_history=False # Important: Do NOT clear history, as we just loaded it
    )

    # The assistant should still recall "blue" because of the loaded history.
    print(f"\nAssistant_New (response from loaded history): {final_result.last_message['content']}")

    print("\n--- Demonstration Complete ---")

if __name__ == "__main__":
    run_demonstration()

    # Optional: Clean up the history file after demonstration
    # if os.path.exists(CHAT_HISTORY_FILE):
    #     os.remove(CHAT_HISTORY_FILE)
    #     print(f"Cleaned up {CHAT_HISTORY_FILE}")

```
