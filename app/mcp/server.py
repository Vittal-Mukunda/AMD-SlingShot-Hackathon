from fastmcp import FastMCP
from app.mcp.tools import register_tools
import sys

# Create MCP app
# Reference: https://github.com/mcp-sdk/python-sdk
mcp = FastMCP(name="Project Manager MCP")

# Register tools
# We pass the mcp instance to our tools module to register functions
register_tools(mcp)

# Expose ASGI app for FastAPI mounting
# FastMCP exposes .http_app() method for ASGI integration
mcp_app = mcp.http_app()

if __name__ == "__main__":
    # Standard entry point for stdio transport
    mcp.run()
