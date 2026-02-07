from mcp.server.fastmcp import FastMCP

# Create an MCP server named "MathServer"
mcp = FastMCP("MathServer")

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiplies two numbers together and returns the result."""
    print(f"DEBUG: Server is calculating {a} * {b}")
    return a * b

if __name__ == "__main__":
    # Start the server using the Stdio transport (standard for local agents)
    mcp.run()