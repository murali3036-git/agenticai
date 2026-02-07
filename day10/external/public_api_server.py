import httpx
import os
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("PublicAssistant")

# TOOL 1: Weather (Open-Meteo - No Key Needed)
@mcp.tool()
async def get_weather(city: str) -> str:
    """Fetches current weather for a city using Open-Meteo's public API."""
    # First, get coordinates for the city (Geocoding)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
    
    async with httpx.AsyncClient() as client:
        geo_res = await client.get(geo_url)
        geo_data = geo_res.json()
        
        if not geo_data.get("results"):
            return f"Could not find coordinates for {city}."
            
        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        
        # Second, get the weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = await client.get(weather_url)
        w_data = weather_res.json()
        
        temp = w_data["current_weather"]["temperature"]
        return f"The current temperature in {city} is {temp}°C."

# TOOL 2: Search (DuckDuckGo Lite - No Key Needed)
# Note: We use a simple scraping approach for DDG since they don't have a JSON API
@mcp.tool()
async def quick_search(query: str) -> str:
    """Performs a quick web search using DuckDuckGo."""
    # We use a public search redirector or a simple metadata API
    # For this example, we'll use the 'api.duckduckgo.com' for instant answers
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        
        abstract = data.get("AbstractText", "")
        if abstract:
            return f"Search Result: {abstract}"
        return "No instant answer found. Try a different query."

if __name__ == "__main__":
    mcp.run()