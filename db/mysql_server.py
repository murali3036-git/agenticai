import mysql.connector
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EnterpriseDB")

# Database configuration (adjust to your local MySQL settings)
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'finance_db'
}

@mcp.tool()
def get_portfolio_from_db() -> str:
    """Retrieves all owned assets and purchase prices from the MySQL database."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM assets")
    rows = cursor.fetchall()
    conn.close()
    return str(rows)

@mcp.tool()
def log_compliance_action(ticker: str, message: str) -> str:
    """Inserts a new audit log entry into the MySQL database."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    query = "INSERT INTO compliance_logs (asset_ticker, action_taken) VALUES (%s, %s)"
    cursor.execute(query, (ticker, message))
    conn.commit()
    conn.close()
    print("Logged here")
    return f"Logged action for {ticker} in MySQL."

if __name__ == "__main__":
    mcp.run()