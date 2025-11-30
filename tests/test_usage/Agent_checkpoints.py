from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Setup (as in Case 1)
external_repo = ExternalSessionRepository()
checkpoint_repo = CheckpointRepository()

external_session = external_repo.create(ExternalSession(
    user_id=1,
    session_name="Checkpoint Demo",
    created_at=datetime.now()
))

@tool
def process_order(order_id: str) -> str:
    """Process a customer order."""
    return f"Order {order_id} processed successfully"

# Create agent with automatic checkpoints
agent = RollbackAgent(
    external_session_id=external_session.id,
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[process_order],
    auto_checkpoint=True,  # Automatically checkpoint after each tool call
    checkpoint_repo=checkpoint_repo
)

# 1. Regular conversation (no checkpoint created - no tools used)
agent.run("Hello, I need help with an order")

# 2. Create a manual checkpoint before important operation
checkpoint_msg = agent.create_checkpoint_tool("Before order processing")
print(checkpoint_msg)
# Output: "✓ Checkpoint 'Before order processing' created successfully (ID: 1)"

# 3. Tool usage triggers automatic checkpoint
agent.run("Please process order #12345")
# Automatic checkpoint "After process_order" is created

# 4. Another manual checkpoint
agent.create_checkpoint_tool("After first order")

# 5. More operations with auto-checkpoints
agent.run("Now process order #67890")

# 6. List all checkpoints
checkpoints_list = agent.list_checkpoints_tool()
print(checkpoints_list)
# Output:
# Available checkpoints:
# • ID: 1 | Before order processing | Type: manual | Created: 2024-01-15 10:30:45
# • ID: 2 | After process_order | Type: auto | Created: 2024-01-15 10:31:12
# • ID: 3 | After first order | Type: manual | Created: 2024-01-15 10:31:45
# • ID: 4 | After process_order | Type: auto | Created: 2024-01-15 10:32:10

# 7. Get detailed info about a specific checkpoint
info = agent.get_checkpoint_info_tool(checkpoint_id=1)
print(info)

# 8. Clean up old automatic checkpoints (keep only last 2)
cleanup_result = agent.cleanup_auto_checkpoints_tool(keep_latest=2)
print(cleanup_result)
# Output: "✓ Cleaned up 2 old automatic checkpoints"

# 9. Access checkpoint data programmatically
all_checkpoints = checkpoint_repo.get_by_internal_session(
    agent.internal_session.id,
    auto_only=False
)
print(f"Total checkpoints: {len(all_checkpoints)}")