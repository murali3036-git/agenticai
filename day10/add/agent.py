import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent():
    # 1. Define how to connect to our server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    # 2. Establish the connection
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # 3. Discovery: The agent "asks" what the server can do
            tools = await session.list_tools()
            print(f"Agent discovered tools: {[t.name for t in tools.tools]}")

            # 4. Agentic Action: Let's manually trigger the tool 
            # (In a real scenario, an LLM would decide to call this)
            result = await session.call_tool("multiply", arguments={"a": 12, "b": 5})
            print(f"Result from agent call: {result.content[0].text}")

if __name__ == "__main__":
    asyncio.run(run_agent())