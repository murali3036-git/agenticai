import ollama
from duckduckgo_search import DDGS
import json

# --- 1. Define the External Tool (Web Search) ---
def search_web(query: str) -> str:
    """A tool to perform a web search for up-to-date information."""
    with DDGS() as ddgs:
        # Get the top 3 results and return them as a formatted string
        results = [r for r in ddgs.text(query, max_results=3)]
        
        if not results:
            return "No web results found."
            
        # Format results for the LLM to read easily
        formatted_results = "\n".join([f"Title: {r['title']}, Snippet: {r['snippet']}" for r in results])
        return f"Web Search Results for '{query}':\n{formatted_results}"

# Ollama requires the tool to be described in a JSON dictionary
TOOL_SEARCH_WEB = {
    'type': 'function',
    'function': {
        'name': 'search_web',
        'description': 'Use this tool to search the internet for current or external information.',
        'parameters': {
            'type': 'object',
            'required': ['query'],
            'properties': {
                'query': {'type': 'string', 'description': 'The exact search query.'}
            }
        }
    }
}
# --- 2. Define the Agentic Loop ---
def run_agent(model_name="mistral"):
    print(f"Agent running with Ollama model: {model_name}")
    
    # 1. Initial Prompt for Agent Role (System Message)
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to a 'search_web' tool. You MUST decide whether to use the tool to answer user messages, especially for current events or external knowledge. Respond directly to simple questions."}
    ]

    while True:
        try:
            user_input = input("\n[USER] > ")
        except EOFError:
            break
        
        if user_input.lower() == 'quit':
            break

        messages.append({"role": "user", "content": user_input})

        # 2. Call the LLM with Tools
        agent_response = ollama.chat(
            model=model_name,
            messages=messages,
            tools=[TOOL_SEARCH_WEB] # Pass the tool definition
        )

        # 3. Check for Tool Call (The Agentic Decision)
        if agent_response.get('tool_calls'):
            tool_call = agent_response['tool_calls'][0]['function']
            tool_name = tool_call['name']
            tool_args = json.loads(tool_call['arguments'])
            
            print(f"\n[AGENT ACTION]: Calling Tool: {tool_name} with args: {tool_args}")
            
            # Execute the tool function
            if tool_name == 'search_web':
                tool_output = search_web(**tool_args)
            
            # 4. Pass Tool Output back to the LLM
            messages.append(agent_response['tool_calls'][0])
            messages.append({
                "role": "tool",
                "content": tool_output,
            })
            
            # Second LLM call to synthesize the final answer
            final_response = ollama.chat(
                model=model_name,
                messages=messages,
            )
            
            print(f"\n[AGENT RESPONSE]: {final_response['message']['content']}")
            
        else:
            # 3b. No Tool Call, just respond
            final_content = agent_response['message']['content']
            print(f"\n[AGENT RESPONSE]: {final_content}")
            messages.append(agent_response['message'])
            

if __name__ == "__main__":
    run_agent()
