from fastapi import FastAPI
from app.api.routes import tasks
from app.mcp.server import mcp_app

app = FastAPI(title="MCP Project Manager")

# REST routes
app.include_router(tasks.router, prefix="/api")

# Mount MCP (important)
app.mount("/mcp", mcp_app)
