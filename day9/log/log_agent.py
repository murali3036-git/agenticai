import asyncio
import os
import sys
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath("log_server.py")],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            ollama_tools = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in tools.tools]

            # The Sophisticated Multi-Step Prompt
            prompt = ("Analyze the 'auth' and 'system' logs. If you detect a security threat or "
                      "a hardware failure, generate a detailed incident report for each.")
            
            messages = [{"role": "user", "content": prompt}]
            print("🕵️  Agent is scanning production logs for anomalies...")

            response = ollama.chat(model="mistral", messages=messages, tools=ollama_tools)

            while response.message.tool_calls:
                messages.append(response.message)
                for call in response.message.tool_calls:
                    print(f"🛠️  Tool Call: {call.function.name}")
                    result = await session.call_tool(call.function.name, call.function.arguments)
                    messages.append({"role": "tool", "content": str(result.content[0].text), "name": call.function.name})
                
                response = ollama.chat(model="mistral", messages=messages, tools=ollama_tools)

            print(f"\n✨ AGENT'S INVESTIGATION SUMMARY:\n{response.message.content}")

if __name__ == "__main__":
    asyncio.run(main())