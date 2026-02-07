import os
import logging
from mcp.server.fastmcp import FastMCP

# Setup logging to a file so it doesn't mess up the MCP communication
logging.basicConfig(filename="mcp_server.log", level=logging.DEBUG)

# Initialize FastMCP
mcp = FastMCP("FileSystem")

@mcp.tool()
def list_my_files(path: str = ".") -> str:
    """Lists files in the specified directory to help the user browse."""
    try:
        files = os.listdir(path)
        return "\n".join(files) if files else "The directory is empty."
    except Exception as e:
        return f"Error accessing directory: {str(e)}"
@mcp.tool()
def read_file_content(filename: str) -> str:
    """Reads the text content of a file. Use this to analyze the data inside a file."""
    try:
        # Security check: only allow reading from the current directory
        base_name = os.path.basename(filename)
        with open(base_name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"
if __name__ == "__main__":
    # Standard MCP run command
    mcp.run()