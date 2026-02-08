import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("LogAnalyzer")

LOG_FILES = {
    "auth": "auth.log",
    "system": "syslog.log"
}

@mcp.tool()
def get_log_summary(log_type: str) -> str:
    """Reads the specified log and returns its content."""
    filename = LOG_FILES.get(log_type)
    if not filename or not os.path.exists(filename):
        return f"Log file {log_type} not found."
    
    with open(filename, "r") as f:
        return f.read()

@mcp.tool()
def create_incident_report(incident_type: str, details: str) -> str:
    """Saves a formal incident report to a new file."""
    filename = f"INCIDENT_{incident_type.upper()}.txt"
    with open(filename, "w") as f:
        f.write(f"INCIDENT REPORT\nType: {incident_type}\nDetails: {details}")
    return f"Report saved as {filename}"

if __name__ == "__main__":
    mcp.run()