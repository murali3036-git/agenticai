import asyncio
import os
import sys
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 1. Path to your server
    server_script = os.path.abspath("file_server.py")
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        # Forces Python to talk to the agent immediately
        env={**os.environ, "PYTHONUNBUFFERED": "1"} 
    )

    print(f"--- Attempting to connect to: {server_script} ---")

    try:
        async with stdio_client(server_params) as (read, write):
            # 2. Removed the 'init_timeout_seconds' argument that caused the error
            async with ClientSession(read, write) as session:
                
                # 3. Perform the handshake
                await session.initialize()
                
                # 4. Discovery
                tools = await session.list_tools()
                print(f"✅ Success! Tools found: {[t.name for t in tools.tools]}")

                # 5. Agentic Step (Ollama)
                # We wrap this in a small sleep to let the Windows process stabilize
                await asyncio.sleep(0.5)
                #Check if there is a file named 'mcp_server.log'. If there is, tell me what the third line says.
                messages = [{"role": "user", "content": "List the files in this folder."}]
                
                # Prepare tool definitions for Ollama
                ollama_tools = [{
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.inputSchema
                    }
                } for t in tools.tools]

                response = ollama.chat(
                    model="mistral",
                    messages=messages,
                    tools=ollama_tools
                )

                # 6. Handle Tool Call
                if response.message.tool_calls:
                    tool_call = response.message.tool_calls[0]
                    print(f"🛠️  Agent is running: {tool_call.function.name}")
                    
                    result = await session.call_tool(
                        tool_call.function.name, 
                        tool_call.function.arguments
                    )
                    print(f"📄 Server Result:\n{result.content[0].text}")
                else:
                    print(f"AI: {response.message.content}")

    except Exception as e:
        print(f"\n❌ CONNECTION ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())