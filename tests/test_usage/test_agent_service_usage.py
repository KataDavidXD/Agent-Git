from agentgit.agents.agent_service import AgentService
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.user_repository import UserRepository
from langchain_core.tools import tool
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Define tools
@tool
def process_payment(amount: float, order_id: str) -> str:
    """Process a payment for an order."""
    return f"Payment of ${amount} processed for order {order_id}"

def reverse_payment(args, result):
    """Reverse a payment."""
    print(f"  ✓ Reversed: Refunded ${args['amount']} for order {args['order_id']}")
    return f"Refunded ${args['amount']} for order {args['order_id']}"

# Step 1: Create user and external session (same as before)
user_repo = UserRepository()
user = user_repo.find_by_username("rootusr")

external_repo = ExternalSessionRepository()
external_session = external_repo.create(ExternalSession(
    user_id=user.id,
    session_name="Payment Processing",
    created_at=datetime.now()
))

# Step 2: Initialize service (auto-creates all repositories)
service = AgentService()
# ✓ No manual repository creation
# ✓ Model config loaded from environment
# ✓ All dependencies managed internally

# Step 3: Create agent with minimal code
agent = service.create_new_agent(
    external_session_id=external_session.id,
    tools=[process_payment],
    reverse_tools={"process_payment": reverse_payment},
    auto_checkpoint=True
)
# ✓ No model creation needed
# ✓ No repository passing needed
# ✓ Best-practice defaults applied

# Step 4: Use the agent
response1 = agent.run("Process a payment of $150 for order #1001")
print(response1)

# Create checkpoint
checkpoint_msg = agent.create_checkpoint_tool("Before second payment")
checkpoint_id = int(checkpoint_msg.split("ID: ")[1].split(")")[0])

response2 = agent.run("Process another payment of $250 for order #1002")
print(response2)

# Step 5: List checkpoints using service utility
checkpoints = service.list_checkpoints(agent.internal_session.id)
print(f"Total checkpoints: {len(checkpoints)}")

# Step 6: Resume session later
# Simulate app restart
service = AgentService()
resumed_agent = service.resume_agent(
    external_session_id=external_session.id,
    tools=[process_payment],
    reverse_tools={"process_payment": reverse_payment}
)
# ✓ Conversation history automatically restored
# ✓ Session state automatically loaded
# ✓ Ready to continue immediately

if resumed_agent:
    response3 = resumed_agent.run("What payments did we process?")
    print(response3)

# Step 7: Rollback with automatic tool reversal
rolled_back = service.rollback_to_checkpoint(
    external_session_id=external_session.id,
    checkpoint_id=checkpoint_id,
    rollback_tools=True,  # Automatically reverses payment operations!
    tools=[process_payment],
    reverse_tools={"process_payment": reverse_payment}
)
# ✓ Checkpoint retrieved automatically
# ✓ Tool operations reversed automatically
# ✓ New branch created automatically
# ✓ Ready to use immediately

if rolled_back:
    response4 = rolled_back.run("Let's try a different payment amount: $300 for order #1003")
    print(response4)
    print()