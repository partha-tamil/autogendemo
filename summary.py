import autogen

# --- 1. Define Agent Configuration ---
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4", "gpt-3.5-turbo"], # Or whatever models you have configured
    },
)

# --- 2. Initialize Agents ---

# User Proxy Agent: Acts as the human user, initiates tasks, and receives final output.
user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. You will provide the initial task and review the final summary.",
    code_execution_config={"last_n_messages": 1, "work_dir": "coding"},
    human_input_mode="NEVER",  # Set to "ALWAYS" or "TERMINATE" for human interaction
)

# Agent A: Focuses on gathering specific information (e.g., market trends)
agent_a = autogen.AssistantAgent(
    name="MarketResearcher",
    llm_config={"config_list": config_list},
    system_message="You are a market research expert. Your task is to analyze the current market trends for AI assistants and provide key findings.",
)

# Agent B: Focuses on gathering different information (e.g., competitor analysis)
agent_b = autogen.AssistantAgent(
    name="CompetitorAnalyst",
    llm_config={"config_list": config_list},
    system_message="You are a competitor analyst. Your task is to identify top competitors in the AI assistant space and summarize their key features.",
)

# Summarizer Agent: Combines inputs from other agents and generates a final summary
summarizer_agent = autogen.AssistantAgent(
    name="SummaryGenerator",
    llm_config={"config_list": config_list},
    system_message="You are a professional summarizer. Your task is to take the findings from the MarketResearcher and CompetitorAnalyst, synthesize them, and produce a concise, final report summarizing the overall landscape for AI assistants. Focus on actionable insights.",
)

# --- 3. Create a Group Chat (Optional but recommended for complex interactions) ---
# This allows agents to communicate and collaborate more naturally.
groupchat = autogen.GroupChat(
    agents=[user_proxy, agent_a, agent_b, summarizer_agent],
    messages=[],
    max_round=15,  # Limit the number of communication rounds
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

# --- 4. Initiate the Conversation ---
# The user_proxy starts the conversation with the manager of the group chat.
user_proxy.initiate_chat(
    manager,
    message="I need a comprehensive summary of the current landscape for AI assistants. "
            "MarketResearcher, please provide key market trends. "
            "CompetitorAnalyst, please identify top competitors and their features. "
            "Finally, SummaryGenerator, please synthesize these findings into a final report.",
)

# --- 5. Retrieve the Final Summary ---
# The final message in the conversation (from the summarizer agent) will be your summary.
# You might need to inspect the chat history to pinpoint the exact message.

# You can access the last message of the conversation through user_proxy.chat_messages
# or by defining a termination condition for the summarizer.

# A more robust way to get the final summary:
# In a real-world scenario, you might have the summarizer agent explicitly
# send its final output to the user_proxy with a specific keyword or a termination message.
# For simplicity in this example, we'll assume the last message from the summarizer is the summary.

print("\n--- Conversation History ---")
for i, msg in enumerate(groupchat.messages):
    print(f"Round {i+1} - From {msg['name']} ({msg['role']}):")
    print(msg['content'])
    print("-" * 30)

# To specifically extract the summary:
# You'd typically design the summarizer to respond to the user_proxy,
# or have a specific message ending.
# For this example, let's assume the last message from the SummaryGenerator is the final summary.

final_summary_message = None
for msg in reversed(groupchat.messages):
    if msg['name'] == "SummaryGenerator":
        final_summary_message = msg['content']
        break

if final_summary_message:
    print("\n--- Final Summary from SummaryGenerator ---")
    print(final_summary_message)
else:
    print("\nCould not find a final summary from the SummaryGenerator.")
