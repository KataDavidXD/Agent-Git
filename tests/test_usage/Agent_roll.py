from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.user_repository import UserRepository
from agentgit.auth.user import User
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Step 1: Create or get a user
user_repo = UserRepository()  # Auto-creates 'rootusr' with ID=1
user = user_repo.find_by_username("rootusr")

# Step 2: Setup repositories
external_repo = ExternalSessionRepository()
checkpoint_repo = CheckpointRepository()
internal_repo = InternalSessionRepository()

external_session = external_repo.create(ExternalSession(
    user_id=user.id,
    session_name="Rollback Demo",
    created_at=datetime.now()
))

# Define a tool with side effects
order_database = {}  # Simulated database

@tool
def create_order(order_id: str, amount: float) -> str:
    """Create a new order in the database."""
    order_database[order_id] = {"amount": amount, "status": "created"}
    return f"Order {order_id} created for ${amount}"

def reverse_create_order(args, result):
    """Reverse order creation by deleting it."""
    order_id = args['order_id']
    if order_id in order_database:
        del order_database[order_id]
    return f"Order {order_id} deleted"

# Create agent with rollback capability
agent = RollbackAgent(
    external_session_id=external_session.id,
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[create_order],
    reverse_tools={"create_order": reverse_create_order},
    auto_checkpoint=True,
    checkpoint_repo=checkpoint_repo,
    internal_session_repo=internal_repo
)

# Conversation flow
print("=== Original Timeline ===")
agent.run("Hello! I need to create some orders.")

# Create manual checkpoint before operations
checkpoint_msg = agent.create_checkpoint_tool("Before order creation")
print(checkpoint_msg)
# Extract checkpoint ID from message: "✓ Checkpoint 'Before order creation' created successfully (ID: 1)"
safe_checkpoint_id = 1  # In practice, parse this from checkpoint_msg

# Create some orders (auto-checkpoints created after each)
agent.run("Please create order #1001 for $250")
print(f"Database state: {order_database}")

agent.run("Now create order #1002 for $150")
print(f"Database state: {order_database}")

agent.run("Create one more order #1003 for $300")
print(f"Database state: {order_database}")
# Database now has: {1001: ..., 1002: ..., 1003: ...}

# Get conversation history length
original_history = agent.get_conversation_history()
print(f"Original history length: {len(original_history)} messages")

# Oops! We need to go back to before the orders were created
print("\n=== Performing Rollback ===")

# Rollback to the safe checkpoint - creates a new branch
branched_agent = RollbackAgent.from_checkpoint(
    checkpoint_id=safe_checkpoint_id,
    external_session_id=external_session.id,
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[create_order],
    reverse_tools={"create_order": reverse_create_order},
    checkpoint_repo=checkpoint_repo,
    internal_session_repo=internal_repo
)

# Check the branched agent's state
branched_history = branched_agent.get_conversation_history()
print(f"Branched history length: {len(branched_history)} messages")
print(f"Database state after branch: {order_database}")

# The branch has the state from before the checkpoint
# But the tool operations haven't been reversed yet
# To reverse tools, get the checkpoint and rollback from its track position
checkpoint = checkpoint_repo.get_by_id(safe_checkpoint_id)
if "tool_track_position" in checkpoint.metadata:
    track_position = checkpoint.metadata["tool_track_position"]
    reverse_results = branched_agent.rollback_tools_from_track_index(track_position)

    print("\nTool Reversal Results:")
    for result in reverse_results:
        if result.reversed_successfully:
            print(f"✓ Reversed {result.tool_name}")
        else:
            print(f"✗ Failed to reverse {result.tool_name}: {result.error_message}")

print(f"Database after tool reversal: {order_database}")

# Continue with different actions on the branch
print("\n=== New Branch Timeline ===")
branched_agent.run("Let's create just one order instead: #2001 for $500")
print(f"Database state: {order_database}")

# Original agent can still continue independently
print("\n=== Original Timeline Continues ===")
agent.run("Create another order #1004 for $200")
print(f"Database state: {order_database}")

# Verify both sessions exist
all_sessions = internal_repo.get_by_external_session(external_session.id)
print(f"\nTotal internal sessions (branches): {len(all_sessions)}")
print(f"Original session ID: {agent.internal_session.id}")
print(f"Branched session ID: {branched_agent.internal_session.id}")
print(f"Branch parent: {branched_agent.internal_session.parent_session_id}")
print(f"Is branch: {branched_agent.internal_session.is_branch()}")