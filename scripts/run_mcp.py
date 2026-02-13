import sys
import os

# Add project root to python path to ensure imports work
# Current file is in scripts/, so root is one level up
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.mcp.server import mcp

if __name__ == "__main__":
    # Run the MCP server in stdio mode (default behavior of mcp.run())
    mcp.run()
