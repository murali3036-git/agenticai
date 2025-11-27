import requests
import json
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

# --- 1. Define the API Tool ---

class CryptoPriceInput(BaseModel):
    """Input schema for the get_crypto_price tool."""
    coin_id: str = Field(description="The ID of the cryptocurrency, e.g., 'bitcoin' or 'ethereum'.")
    currency: str = Field(description="The fiat currency to check the price in, e.g., 'usd' or 'eur'.")

@tool("get_crypto_price", args_schema=CryptoPriceInput)
def get_crypto_price(coin_id: str, currency: str) -> str:
    """
    Returns the current price of a cryptocurrency from the CoinGecko API.
    Use this tool only when the user asks for a price, e.g., 'What is the price of Bitcoin?'.
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": coin_id.lower(),
            "vs_currencies": currency.lower()
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Check if the data is valid and extract the price
        price_data = data.get(coin_id.lower(), {}).get(currency.lower())
        
        if price_data is not None:
            print(f"✅ [TOOL EXECUTION SUCCESS]: Fetched price for {coin_id}.")
            return f"The current price of {coin_id.capitalize()} is {price_data} {currency.upper()}."
        else:
            return f"Error: Could not find price data for {coin_id} in {currency}."

    except requests.exceptions.RequestException as e:
        return f"An error occurred while fetching the price: The API is unavailable or request failed."

# Define the list of tools and the map for execution
tools = [get_crypto_price]
tools_map: Dict[str, Any] = {tool.name: tool for tool in tools}


# --- 2. Setup LLM and Chain ---
try:
    # Use ChatOllama for models like Llama 3 that support tool calling natively.
    llm = ChatOllama(model="mistral", temperature=0) 
except Exception as e:
    print(f"❌ Error initializing Ollama: {e}")
    print("Please ensure Ollama is running and the 'llama3' model is pulled.")
    exit()

# Bind the tools to the LLM; this adds the tool schemas to the prompt context.
llm_with_tools = llm.bind_tools(tools)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent financial assistant. Use the 'get_crypto_price' tool ONLY when asked for a specific coin price. If the tool is used, provide a clear, final answer based on the tool's result."),
        ("placeholder", "{messages}"), 
    ]
)

# Create the runnable chain (Prompt -> LLM with Tools)
runnable = prompt | llm_with_tools

# --- 3. Implement the Correct Manual Agent Loop ---

def run_manual_agent(user_input: str) -> str:
    """
    Manually runs the Agent/Tool loop by inspecting the AIMessage's tool_calls attribute.
    """
    print(f"\n{'='*20} RUNNING QUERY {'='*20}")
    print(f"[USER]: {user_input}")
    
    # Start the messages history with the user's initial message
    messages: List[HumanMessage | AIMessage | ToolMessage] = [HumanMessage(content=user_input)]
    
    # 1. LLM decides (Invoke the chain)
    ai_message: AIMessage = runnable.invoke({"messages": messages})
    
    # --- Check for Tool Call ---
    if not ai_message.tool_calls:
        # If no tool call, the LLM answered directly
        print("💡 LLM answered directly (no tool call requested).")
        return ai_message.content
    
    # --- Tool Calling Execution ---
    
    print(f"⚙️ LLM requested {len(ai_message.tool_calls)} tool call(s)...")
    messages.append(ai_message) # Add the AI's tool call message to history
    
    # 2. Execute the tool(s) and prepare results
    for tool_call in ai_message.tool_calls:
        tool_name = tool_call["name"]
        # The 'args' are often returned as a dictionary; we ensure it's a dict for invocation
        tool_args = tool_call.get("args", {}) 
        tool_id = tool_call["id"] # Required to link the result back to the tool call
        
        if tool_name not in tools_map:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Call the actual Python function (the tool)
        tool_output_string = tools_map[tool_name].invoke(tool_args)
        
        # 3. Create a ToolMessage with the result and the linked tool_call_id
        tool_message = ToolMessage(
            content=tool_output_string,
            tool_call_id=tool_id,
        )
        messages.append(tool_message) # Add the tool's output to the history
        
    # 4. Final invocation: LLM uses the tool output to generate the final answer
    print("🔄 Sending tool results back to LLM for final answer...")
    final_ai_message = runnable.invoke({"messages": messages})

    return final_ai_message.content

# --- 4. Run the Agent Queries ---

if __name__ == "__main__":
    print("\n" + "="*70)
    print("💰 OLLAMA MANUAL AGENT DEMO: Live Crypto Price Checker")
    print("="*70)

    # Query 1: Requires the API tool
    api_query = "What's the current price of Cardano (JP) in USD?"
    final_answer_1 = run_manual_agent(api_query)
    print(f"\n**[FINAL AI RESPONSE]:** {final_answer_1}")

    # Query 2: Does NOT require the tool
    direct_query = "What is the primary difference between a cryptocurrency and a fiat currency?"
    final_answer_2 = run_manual_agent(direct_query)
    print(f"\n**[FINAL AI RESPONSE]:** {final_answer_2}")

"""
Initial Message: The user's input is wrapped in a HumanMessage and sent to the LLM.

LLM Decision: The chain is invoked. The LLM either returns a simple text answer OR returns an AIMessage containing a structured object in the tool_calls attribute.

Tool Call Check: The code checks if ai_message.tool_calls is populated.

If empty: The LLM's text content is returned directly.

If populated (Tool is called):

The loop extracts the tool name, arguments (args), and the unique tool ID.

It executes the corresponding function from the tools_map using the extracted arguments.

Send Result Back: The output from the executed tool (e.g., "The price is $68,000 USD") is wrapped in a ToolMessage (linked by the tool_call_id) and appended to the message history.

Final Response: The entire history (User Prompt, LLM Tool Call, Tool Result) is sent back to the LLM. The LLM then reads the tool's result and generates the final, natural-language answer for the user.
"""