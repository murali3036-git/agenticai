import asyncio
import os
import sys
import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_enterprise_agent():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[os.path.abspath("mysql_server.py")],
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()
            
            # Prepare tools for the LLM
            tools_list = [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}} for t in mcp_tools.tools]

            # The Sophisticated Multi-Step Prompt
            prompt = (
                "Step 1: Use get_portfolio_from_db to see our assets. "
                "Step 2: If Bitcoin or Intel have a loss > 20%, you MUST call the "
                "log_compliance_action tool for each one. Do not just tell me you are doing it, "
                "actually execute the tool."
            )
            messages = [{"role": "user", "content": prompt}]
            
            print("🚀 Starting Enterprise Agent...")
            response = ollama.chat(model="mistral", messages=messages, tools=tools_list)
            print("🚀 next step Enterprise Agent...")

            while response.message.tool_calls:
                messages.append(response.message)
                for call in response.message.tool_calls:
                    print(f"🛠️  Database Action: {call.function.name}")
                    result = await session.call_tool(call.function.name, call.function.arguments)
                    messages.append({"role": "tool", "content": str(result.content[0].text), "name": call.function.name})
                
                response = ollama.chat(model="mistral", messages=messages, tools=tools_list)

            print(f"\n✨ EXECUTION SUMMARY:\n{response.message.content}")

if __name__ == "__main__":
    asyncio.run(run_enterprise_agent())