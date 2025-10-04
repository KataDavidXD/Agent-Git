from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime

# Step 1: Create an external session (conversation container)
external_repo = ExternalSessionRepository()
external_session = external_repo.create(ExternalSession(
    user_id=1,
    session_name="Customer Support Session",
    created_at=datetime.now()
))

# Step 2: Define your tools (optional)
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# Step 3: Create the agent
agent = RollbackAgent(
    external_session_id=external_session.id,
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[calculate_sum],
    auto_checkpoint=True  # Enable automatic checkpointing after tool calls
)

# Step 4: Chat with the agent
response1 = agent.run("Hello! How can you help me?")
print(response1)

response2 = agent.run("Please calculate 25 + 17 for me.")
print(response2)

response3 = agent.run("Thanks for the help!")
print(response3)