
<p align="center">
  <picture>
    <!-- Dark theme -->
    <source 
      media="(prefers-color-scheme: dark)" 
      srcset="https://github.com/user-attachments/assets/a22537bf-f800-46a0-b49f-35ef1cc55cff" />
    <!-- Light theme -->
    <source 
      media="(prefers-color-scheme: light)" 
      srcset="https://github.com/user-attachments/assets/34f95487-325d-4777-bf65-ceac5a160cb4" />
    <!-- Fallback -->
    <img
      alt="Agent Git"
      src="https://github.com/user-attachments/assets/a22537bf-f800-46a0-b49f-35ef1cc55cff"
      style="
        display: block;
        margin: auto;
        max-width: 90%;
        height: auto;
        border-radius: 10px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: box-shadow 0.3s ease-in-out;
      "
      onmouseover="this.style.boxShadow='0 0 25px rgba(100, 180, 255, 0.5)'"
      onmouseout="this.style.boxShadow='0 0 15px rgba(0, 0, 0, 0.15)'"
    />
  </picture>
</p>

<h1 align="center">Agent Git: Agent Version Control, Open-Branching, and Reinforcement Learning MDP for Agentic AI</h1>


<div align="center">

[![Docs](https://img.shields.io/badge/docs-latest-blue?logo=readthedocs&logoColor=auto)](https://katadavidxd.github.io/Agent-Git/)
[![GitHub stars](https://img.shields.io/github/stars/katadavidxd/agent-git?logo=github&logoColor=auto)](https://github.com/KataDavidXD/Agent-Git/stargazers)
[![License](https://img.shields.io/github/license/katadavidxd/agent-git?logoColor=auto)](https://github.com/KataDavidXD/Agent-Git/blob/main/LICENSE)
[![Pre-release Wheel](https://img.shields.io/badge/Download-Pre--release-blue?logo=python&logoColor=white)](https://github.com/KataDavidXD/Agent-Git/releases/tag/V0.1.0-alpha)

[![WeChat](https://img.shields.io/badge/WeChat-AgentGit-green?logo=wechat&logoColor=white)](https://github.com/user-attachments/assets/e0759fe0-45a9-40b1-9c5c-a9869aee5b63)
[![Reddit](https://img.shields.io/badge/Reddit-AgentGit-orange?logo=reddit&logoColor=white)](https://www.reddit.com/r/AgentGit/s/mstmirU8zo)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=auto)](https://discord.gg/C82Z9C6P)

</div>


<p align="center">
  <a href="https://katadavidxd.github.io/Agent-Git/">📖 <b>Documentation </b></a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://github.com/KataDavidXD/Agent-Git/blob/main/Agent_Git.pdf">🧾 <b>White Paper</b></a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://camo.hku.hk/research-labs/research-labs-lab-for-ai-agents-in-business-and-economics/">🏛️ <b>Our Lab</b></a>
</p>

##

**Agent Git** is the first self-contained package that extends the standard Agentic framework, such as
LangGraph and Agno by introducing Git-like version control for AI conversations. By enabling operators
such as State Commit, State Revert, and Branching, Agent Git provides durable and reproducible
checkpoints, allowing users to reverse actions and travel to previous states on a Markov Chain of
Agentic flow.

## Overview

Agent Git is a comprehensive version control framework for LangGraph agents that introduces
Git-like operations. Analogous to how Git lets developers commit, branch, and checkout code,
Agent Git enables equivalent operations via a three-layer architecture.

<img width="1769" height="1071" alt="image" src="https://github.com/user-attachments/assets/89ac981b-465f-410e-881b-0553c716d01c" />

### Core Concepts

* **Commit State** : A saved snapshot of an agent’s state (internal context + tool usage), persisted in the database.
* **External Session** : A logical container holding multiple Internal Sessions.
* **Internal Session** : A single agent instance with rollback ability, linked to an External Session.
* **Session History** : Message sequence within an Internal Session, including commits, branches, and LangGraph BaseMessages (AI, Human, System, Tool).
* **State Revert** : Restores the agent to an earlier Commit State.
* **Tool** : A stateless executable called by humans, LLMs, or scripts.
* **Tool Revert** : Reverses tool effects based on commit records (simple reversion for state-independent tools, compensating actions for path-dependent tools).
* **Branching** : Git-like branching from a Commit State, creating parallel exploration paths without affecting the original trajectory.


### Key Features

- **Checkpoint & Rollback**: Create restore points and travel back in conversation history
- **Non-Destructive Branching**: Rollbacks create new branches, preserving all timelines
- **Tool Reversal**: Undo side effects of tool operations (database writes, API calls, etc.)
- **Persistent State**: SQLite-backed storage survives restarts
- **Drop-in Integration**: Works seamlessly with existing LangGraph applications
- **Production Ready**: Comprehensive error handling, testing, and performance optimization

## 📦 Download Pre-release Wheel

You can download the **pre-release version** of Agent Git as a Python wheel from our GitHub Releases:

[![Pre-release Wheel](https://img.shields.io/badge/Download-Pre--release-blue?logo=python&logoColor=white)](https://github.com/KataDavidXD/Agent-Git/releases/tag/V0.1.0-alpha)

> ⚠️ This is a pre-release version. Please use with caution and report any issues.

## Installation

## Quick Start Video: Download & Prepare to Use

<p align="center">
  <a href="https://youtu.be/1bTJ0RTdjzU?si=vXsCibB5C8gTmu29" target="_blank">
    <img width="1163" height="814" alt="Agent-Git Demo Video" src="https://github.com/user-attachments/assets/09f6d4cf-578a-4784-a53b-c7d18bef0005" />
  </a>
</p>



### Using uv (Recommended)

```bash
# Install from wheel file
uv pip install agent_git_langchain-0.1.0-py3-none-any.whl

# Or install with dependencies
uv pip install agent_git_langchain-0.1.0-py3-none-any.whl --reinstall
```

### Using pip

```bash
pip install agent_git_langchain-0.1.0-py3-none-any.whl
```

### Required Dependencies

```bash
# Install LangChain and LangGraph
pip install langchain langchain-openai langgraph

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### Prerequisites: User Setup

Agent Git automatically creates a default user on first run:

- **Username**: `rootusr`
- **Password**: `1234`
- **User ID**: `1`

You can use this default user or create your own:

```python
from agentgit.database.repositories.user_repository import UserRepository
from agentgit.auth.user import User
from datetime import datetime

user_repo = UserRepository()  # Auto-creates rootusr

# Option A: Use default user
user = user_repo.find_by_username("rootusr")

# Option B: Create custom user
new_user = User(username="alice", created_at=datetime.now())
new_user.set_password("secure_password")
user = user_repo.save(new_user)
```

### Simple Example

```python
from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.user_repository import UserRepository
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime

# 1. Get user
user_repo = UserRepository()
user = user_repo.find_by_username("rootusr")

# 2. Create session
external_repo = ExternalSessionRepository()
session = external_repo.create(ExternalSession(
    user_id=user.id,
    session_name="My First Agent",
    created_at=datetime.now()
))

# 3. Define tools
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# 4. Create agent
agent = RollbackAgent(
    external_session_id=session.id,
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[calculate_sum],
    auto_checkpoint=True
)

# 5. Chat!
response = agent.run("What is 25 + 17?")
print(response)
```

## Usage Patterns

Agent Git offers two approaches: **Manual** (fine-grained control) and **AgentService** (simplified, production-recommended).

### Option 1: Manual Approach

Full control over all components—ideal for custom configurations.

```python
from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.user_repository import UserRepository
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from datetime import datetime

# Setup repositories
user_repo = UserRepository()
external_repo = ExternalSessionRepository()
checkpoint_repo = CheckpointRepository()
internal_repo = InternalSessionRepository()

# Get user and create session
user = user_repo.find_by_username("rootusr")
session = external_repo.create(ExternalSession(
    user_id=user.id,
    session_name="Manual Agent",
    created_at=datetime.now()
))

# Define tool with side effects
order_db = {}

@tool
def create_order(order_id: str, amount: float) -> str:
    """Create an order."""
    order_db[order_id] = {"amount": amount}
    return f"Order {order_id} created for ${amount}"

def reverse_create_order(args, result):
    """Undo order creation."""
    order_id = args['order_id']
    if order_id in order_db:
        del order_db[order_id]
    return f"Order {order_id} deleted"

# Create agent
agent = RollbackAgent(
    external_session_id=session.id,
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
    tools=[create_order],
    reverse_tools={"create_order": reverse_create_order},
    auto_checkpoint=True,
    checkpoint_repo=checkpoint_repo,
    internal_session_repo=internal_repo
)

# Use the agent
agent.run("Create order #1001 for $250")

# Create checkpoint
checkpoint_msg = agent.create_checkpoint_tool("Before changes")
checkpoint_id = int(checkpoint_msg.split("ID: ")[1].split(")")[0])

# Continue conversation
agent.run("Create order #1002 for $150")

# Rollback - creates new branch!
branched_agent = RollbackAgent.from_checkpoint(
    checkpoint_id=checkpoint_id,
    external_session_id=session.id,
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[create_order],
    reverse_tools={"create_order": reverse_create_order},
    checkpoint_repo=checkpoint_repo,
    internal_session_repo=internal_repo
)

# Reverse tools
checkpoint = checkpoint_repo.get_by_id(checkpoint_id)
if "tool_track_position" in checkpoint.metadata:
    track_pos = checkpoint.metadata["tool_track_position"]
    results = branched_agent.rollback_tools_from_track_index(track_pos)
    for r in results:
        print(f"Reversed {r.tool_name}: {r.reversed_successfully}")
```

### Option 2: AgentService (Recommended)

Simplified API with automatic dependency management—**55% less boilerplate code**.

```python
from agentgit.agents.agent_service import AgentService
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.user_repository import UserRepository
from langchain_core.tools import tool
from datetime import datetime

# Setup session (same as before)
user_repo = UserRepository()
user = user_repo.find_by_username("rootusr")

external_repo = ExternalSessionRepository()
session = external_repo.create(ExternalSession(
    user_id=user.id,
    session_name="Service Agent",
    created_at=datetime.now()
))

# Define tools
order_db = {}

@tool
def create_order(order_id: str, amount: float) -> str:
    """Create an order."""
    order_db[order_id] = {"amount": amount}
    return f"Order {order_id} created for ${amount}"

def reverse_create_order(args, result):
    """Undo order creation."""
    if args['order_id'] in order_db:
        del order_db[args['order_id']]
    return f"Order {args['order_id']} deleted"

# Initialize service (auto-creates repositories, loads model config)
service = AgentService()

# Create agent (model auto-configured from environment)
agent = service.create_new_agent(
    external_session_id=session.id,
    tools=[create_order],
    reverse_tools={"create_order": reverse_create_order},
    auto_checkpoint=True
)

# Use the agent
agent.run("Create order #1001 for $250")

# Create checkpoint
checkpoint_msg = agent.create_checkpoint_tool("Safe point")
checkpoint_id = int(checkpoint_msg.split("ID: ")[1].split(")")[0])

# Continue
agent.run("Create order #1002 for $150")

# One-line rollback with automatic tool reversal!
rolled_back = service.rollback_to_checkpoint(
    external_session_id=session.id,
    checkpoint_id=checkpoint_id,
    rollback_tools=True  # Automatically reverses tools!
)

# Continue on new branch
if rolled_back:
    rolled_back.run("Create order #2001 for $500 instead")

# One-line session resumption
service = AgentService()
resumed = service.resume_agent(external_session_id=session.id)
if resumed:
    resumed.run("What orders do we have?")
```

**Key Differences:**

| Feature                         | Manual          | AgentService       |
| ------------------------------- | --------------- | ------------------ |
| **Setup Code**            | ~45 lines       | ~8 lines           |
| **Repository Management** | Manual creation | Automatic          |
| **Model Config**          | Manual setup    | Auto from env vars |
| **Rollback**              | 25-30 lines     | 1 line             |
| **Session Resume**        | 15-20 lines     | 1 line             |
| **Best For**              | Custom configs  | Production apps    |

## Core Operations

### Creating Checkpoints

```python
# Automatic checkpoints (after tool calls)
agent = RollbackAgent(..., auto_checkpoint=True)

# Manual checkpoints
checkpoint_msg = agent.create_checkpoint_tool("Before risky operation")

# List all checkpoints
agent.list_checkpoints_tool()

# Get checkpoint details
agent.get_checkpoint_info_tool(checkpoint_id=1)
```

### Rolling Back

```python
# Rollback creates a new branch (preserves original timeline)
new_agent = RollbackAgent.from_checkpoint(
    checkpoint_id=checkpoint_id,
    external_session_id=session.id,
    model=model,
    tools=tools,
    reverse_tools=reverse_tools,
    checkpoint_repo=checkpoint_repo,
    internal_session_repo=internal_repo
)

# Or use AgentService for one-liner
rolled_back = service.rollback_to_checkpoint(
    external_session_id=session.id,
    checkpoint_id=checkpoint_id,
    rollback_tools=True
)
```

### Tool Reversal

```python
# Define tool
@tool
def save_record(record_id: str, data: str) -> str:
    """Save data to database."""
    db.write(record_id, data)
    return f"Saved {record_id}"

# Define reverse function
def reverse_save(args, result):
    """Delete the saved record."""
    db.delete(args['record_id'])
    return f"Deleted {args['record_id']}"

# Register when creating agent
agent = RollbackAgent(
    ...,
    tools=[save_record],
    reverse_tools={"save_record": reverse_save}
)
```

### Managing Branches

```python
# Check if session is a branch
agent.internal_session.is_branch()

# Get parent session ID
agent.internal_session.parent_session_id

# Get branch point checkpoint
agent.internal_session.branch_point_checkpoint_id

# List all branches for a session
branches = internal_repo.get_by_external_session(session.id)
print(f"Total branches: {len(branches)}")
```

## Common Use Cases

### 1. Customer Support Recovery

```python
# Agent helps customer, makes mistake
agent.run("Process refund of $500")

# Realize error, rollback to before refund
checkpoint_id = 5  # From before refund
corrected = service.rollback_to_checkpoint(
    external_session_id=session.id,
    checkpoint_id=checkpoint_id,
    rollback_tools=True  # Reverses the refund!
)

# Try again with correct amount
corrected.run("Process refund of $50")
```

### 2. A/B Testing

```python
# Create checkpoint after gathering context
checkpoint_msg = agent.create_checkpoint_tool("Context gathered")
checkpoint_id = 1

# Test approach A
branch_a = RollbackAgent.from_checkpoint(checkpoint_id, ...)
response_a = branch_a.run("Explain technically")

# Test approach B
branch_b = RollbackAgent.from_checkpoint(checkpoint_id, ...)
response_b = branch_b.run("Explain simply")

# Compare results
print(f"Technical: {len(response_a)} chars")
print(f"Simple: {len(response_b)} chars")
```

### 3. Safe Exploration

```python
# Create checkpoint before risky operation
agent.create_checkpoint_tool("Before deletion")

# Try risky operation
agent.run("Delete all test records")

# If something goes wrong, rollback
service.rollback_to_checkpoint(
    external_session_id=session.id,
    checkpoint_id=checkpoint_id,
    rollback_tools=True  # Undoes deletions!
)
```

## Environment Configuration

Agent Git uses environment variables for configuration:

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional (AgentService defaults)
export BASE_URL="https://api.openai.com/v1"  # Custom API endpoint
export DEFAULT_MODEL="gpt-4o-mini"            # Default model
export DEFAULT_TEMPERATURE="0.7"              # Default temperature
```

## API Reference

### RollbackAgent Methods

```python
# Conversation
agent.run(message: str) -> str

# Checkpoints
agent.create_checkpoint_tool(name: str) -> str
agent.list_checkpoints_tool() -> str
agent.get_checkpoint_info_tool(checkpoint_id: int) -> str
agent.cleanup_auto_checkpoints_tool(keep_latest: int) -> str

# State inspection
agent.get_conversation_history() -> List[dict]
agent.get_session_state() -> dict
agent.get_tool_track() -> List[ToolInvocationRecord]

# Rollback
RollbackAgent.from_checkpoint(checkpoint_id, ...) -> RollbackAgent
agent.rollback_tools_from_track_index(position: int) -> List[ReverseResult]
```

### AgentService Methods

```python
# Agent management
service.create_new_agent(external_session_id, tools, ...) -> RollbackAgent
service.resume_agent(external_session_id) -> RollbackAgent | None

# Rollback
service.rollback_to_checkpoint(
    external_session_id,
    checkpoint_id,
    rollback_tools=True
) -> RollbackAgent | None

# Utilities
service.list_checkpoints(internal_session_id) -> List[Checkpoint]
```

## Architecture

The session hierarchy follows a tree structure:

```
External Session (User's conversation container)
|
+-- Internal Session 1 (Main timeline)
|   |
|   +-- Checkpoint 1
|   +-- Checkpoint 2
|   +-- Checkpoint 3
|
+-- Internal Session 2 (Branch from Checkpoint 2)
|   |
|   +-- Checkpoint 4
|   +-- Checkpoint 5
|
+-- Internal Session 3 (Branch from Checkpoint 3)
    |
    +-- Checkpoint 6
```

**Key Concepts:**

- **External Session**: Container for related conversations (one per user/use-case)
- **Internal Session**: Actual agent instance (can have multiple branches)
- **Checkpoint**: Snapshot of conversation state at a specific point
- **Branch**: New internal session created from a checkpoint (preserves original timeline)

## Troubleshooting

### Common Issues

**Issue**: `sqlite3.IntegrityError: FOREIGN KEY constraint failed`

**Solution**: Ensure you're using a valid `user_id` when creating external sessions.

**Issue**: Rollback doesn't reverse tool effects

**Solution**: Register reverse functions in `reverse_tools` parameter.

**Issue**: Can't find previous conversation

**Solution**: Use `service.resume_agent(external_session_id)` to load the active session.

**Issue**: Model not configured

**Solution**: Set `OPENAI_API_KEY` environment variable or pass model explicitly.

## Documentation

For detailed information, see:

- `whitePaper.pdf` - Complete architecture and design documentation
- API documentation (coming soon)
- Deployment guide (coming soon)

## Contributing

We welcome contributions! Agent Git is open source and actively seeking:

- Bug reports and feature requests
- Tool implementations with reverse functions
- Documentation improvements
- Performance optimizations

Partners: **HKU Lab for AI Agents in Business and Economics**

👉 [Learn more](https://camo.hku.hk/research-labs/research-labs-lab-for-ai-agents-in-business-and-economics/)

## License
Agent Git is released under the Apache 2.0 License, ensuring compatibility with the LangGraph and Agno ecosystems while providing patent protection and commercial-friendly terms.

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See `whitePaper.pdf`


---

Built for the LangChain/LangGraph community
