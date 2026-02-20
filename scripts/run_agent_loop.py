import asyncio
import sys
import os
import random
from typing import Dict, Any

# Add project root to sys.path so we can import 'app' and 'environment'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock MCP Client (In reality, use standard MCP Python Client here when agent is ready)
class DummyMCPClient:
    async def call_tool(self, name: str, args: Dict[str, Any] = None) -> Any:
        # Fallback to direct imports just for testing the loop without full MCP server setup
        from app.mcp.tools import task_service, simulation_service
        
        args = args or {}
        if name == "get_rl_observation":
            try:
                from environment.project_env import ProjectEnv
                env = ProjectEnv(num_workers=3, num_tasks=10)
                return env.reset()
            except:
                return [0] * 88 # Mock Observation
                
        elif name == "simulate_step":
            return simulation_service.step(**args)
            
        elif name == "assign_task":
            return task_service.assign_task(**args)
            
        elif name == "defer_task":
            return task_service.defer_task(**args)
            
        elif name == "escalate_task":
            return task_service.escalate_task(**args)
            
        elif name == "get_state":
            return simulation_service.get_state()
            
        print(f"[TCP Client Mock] Called tool: {name}({args})")
        return {}

async def run_agent_loop():
    print("--- Starting Agent Orchestration Loop ---")
    mcp_client = DummyMCPClient()
    
    # 1. Initialize
    print("\n[Env] Initializing State...")
    state = await mcp_client.call_tool("get_state")
    
    # Run 5 steps to verify loop logic
    for step in range(5):
        print(f"\n--- Timestep {step} ---")
        
        # 1. Observe
        obs = await mcp_client.call_tool("get_rl_observation")
        print(f"[Agent] Observed vector of length {len(obs)}")
        
        # 2. Decide (Random Agent Placeholder)
        # Assuming RL action space format:
        # 0-29: Assign Task x to Worker y
        # 30-39: Defer Task x
        # 40-49: Escalate Task x
        action = random.randint(0, 49) 
        
        # 3. Act
        if action < 30:
            t_id = action // 3
            w_id = action % 3
            print(f"[Agent] Decided to ASSIGN Task {t_id} to Worker {w_id}")
            await mcp_client.call_tool("assign_task", {"task_id": f"t{t_id}", "worker_id": f"w{w_id}"})
        elif action < 40:
            t_id = action - 30
            print(f"[Agent] Decided to DEFER Task {t_id}")
            await mcp_client.call_tool("defer_task", {"task_id": f"t{t_id}"})
        else:
            t_id = action - 40
            print(f"[Agent] Decided to ESCALATE Task {t_id}")
            await mcp_client.call_tool("escalate_task", {"task_id": f"t{t_id}"})
            
        # 4. Simulate
        print("[Env] Executing Physics Step...")
        sim_result = await mcp_client.call_tool("simulate_step", {"hours": 1.0})
        
        if isinstance(sim_result, dict) and "current_time" in sim_result:
             print(f"[Env] Simulation time is now: {sim_result['current_time']}")
             if sim_result.get("events"):
                 print(f"[Env] Events: {sim_result['events']}")

    print("\n--- Loop Complete. Waiting for trained DQN integration. ---")

if __name__ == "__main__":
    asyncio.run(run_agent_loop())
