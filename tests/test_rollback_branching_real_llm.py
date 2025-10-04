"""Test suite for agent rollback branching functionality with real OpenAI LLM.

Tests that rollback operations create new internal sessions while preserving original sessions,
and that both branches can continue conversations independently.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import warnings
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Suppress Pydantic V2 deprecation warnings from LangChain
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*__fields__.*")

from src.agents.rollback_agent import RollbackAgent
from src.database.repositories.checkpoint_repository import CheckpointRepository
from src.database.repositories.internal_session_repository import InternalSessionRepository
from src.database.repositories.external_session_repository import ExternalSessionRepository
from src.database.repositories.user_repository import UserRepository
from src.sessions.external_session import ExternalSession


# Simple tools for testing
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

@tool
def remember_fact(fact: str) -> str:
    """Remember a fact for later reference."""
    return f"I will remember: {fact}"


class TestRollbackBranchingRealLLM(unittest.TestCase):
    """Test cases for rollback branching functionality with real OpenAI LLM."""
    
    def setUp(self):
        """Set up test environment with OpenAI model and repositories."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize repositories
        self.user_repo = UserRepository(db_path=self.db_path)
        self.checkpoint_repo = CheckpointRepository(db_path=self.db_path)
        self.internal_repo = InternalSessionRepository(db_path=self.db_path)
        self.external_repo = ExternalSessionRepository(db_path=self.db_path)
        
        # Create external session
        self.external_session = ExternalSession(
            user_id=1,
            session_name="Rollback Branching Test Session",
            created_at=datetime.now()
        )
        self.external_session = self.external_repo.create(self.external_session)
        
        # Create OpenAI model with real API
        self.model = self._create_openai_model()
        
        # Define test tools
        self.tools = [add_numbers, multiply_numbers, remember_fact]
    
    def tearDown(self):
        """Clean up temporary database."""
        try:
            os.unlink(self.db_path)
        except:
            pass
    
    def _create_openai_model(self):
        """Create an OpenAI model for testing."""
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("BASE_URL")
        
        if not api_key:
            self.skipTest("OPENAI_API_KEY environment variable not set")
        
        # Sanitize base URL if provided
        if base_url:
            base_url = base_url.strip().rstrip("/")
            if not base_url.startswith(("http://", "https://")):
                base_url = "https://" + base_url
        
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for more predictable responses
            openai_api_key=api_key,
            openai_api_base=base_url
        )
    
    def test_rollback_creates_new_session_preserves_old(self):
        """Test that rollback creates new internal session while preserving the original."""
        # Create original agent
        original_agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        original_session_id = original_agent.internal_session.id
        original_langgraph_id = original_agent.langgraph_session_id
        
        print(f"\n--- Original Agent Created ---")
        print(f"Original internal session ID: {original_session_id}")
        print(f"Original LangGraph session ID: {original_langgraph_id}")
        
        # Have conversation with original agent
        response1 = original_agent.run("Hello, my name is Alice. Please remember this.")
        print(f"\nOriginal Agent Response 1: {response1}")
        
        # Create checkpoint after introduction
        checkpoint_result = original_agent.create_checkpoint_tool(name="After Introduction")
        print(f"Checkpoint created: {checkpoint_result}")
        
        # Extract checkpoint ID
        checkpoint_id_str = checkpoint_result.split("ID: ")[1].split(")")[0]
        checkpoint_id = int(checkpoint_id_str)
        
        # Continue original conversation
        response2 = original_agent.run("I live in Paris and love mathematics. Use the add_numbers tool to calculate 15 + 27.")
        print(f"\nOriginal Agent Response 2: {response2}")
        
        response3 = original_agent.run("Now please multiply 6 by 8 using the multiply_numbers tool.")
        print(f"Original Agent Response 3: {response3}")
        
        # Store original conversation state
        original_history_before_rollback = original_agent.internal_session.conversation_history.copy()
        original_history_length = len(original_history_before_rollback)
        
        print(f"\nOriginal conversation length before rollback: {original_history_length}")
        
        # Perform rollback - this should create a new internal session
        print(f"\n--- Performing Rollback to Checkpoint {checkpoint_id} ---")
        
        rolled_back_agent = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        rolled_back_session_id = rolled_back_agent.internal_session.id
        rolled_back_langgraph_id = rolled_back_agent.langgraph_session_id
        
        print(f"Rolled-back internal session ID: {rolled_back_session_id}")
        print(f"Rolled-back LangGraph session ID: {rolled_back_langgraph_id}")
        
        # === VERIFICATION TESTS ===
        
        # 1. Verify new internal session was created
        self.assertNotEqual(original_session_id, rolled_back_session_id)
        self.assertNotEqual(original_langgraph_id, rolled_back_langgraph_id)
        print("✓ New internal session created successfully")
        
        # 2. Verify original session still exists and is preserved
        original_session_from_db = self.internal_repo.get_by_id(original_session_id)
        self.assertIsNotNone(original_session_from_db)
        self.assertEqual(len(original_session_from_db.conversation_history), original_history_length)
        print("✓ Original session preserved in database")
        
        # 3. Verify rolled-back agent has correct history (only up to checkpoint)
        rolled_back_history = rolled_back_agent.internal_session.conversation_history
        self.assertLess(len(rolled_back_history), original_history_length)
        self.assertEqual(len(rolled_back_history), 2)  # Only "Hello, my name is Alice" exchange
        print(f"✓ Rolled-back agent has correct history length: {len(rolled_back_history)}")
        
        # 4. Verify branch relationship
        self.assertTrue(rolled_back_agent.internal_session.is_branch())
        self.assertEqual(rolled_back_agent.internal_session.parent_session_id, original_session_id)
        self.assertEqual(rolled_back_agent.internal_session.branch_point_checkpoint_id, checkpoint_id)
        print("✓ Branch relationship established correctly")
        
        # 5. Verify rolled-back agent remembers Alice but not Paris/math
        rolled_back_response = rolled_back_agent.run("Do you remember my name and where I live?")
        print(f"\nRolled-back agent response: {rolled_back_response}")
        
        # The response shows "your name is Alice" - this is correct behavior
        response_lower = rolled_back_response.lower()
        self.assertTrue("alice" in response_lower, f"Agent should remember Alice's name. Response: {rolled_back_response}")
        self.assertFalse("paris" in response_lower, f"Agent should not remember Paris (came after checkpoint). Response: {rolled_back_response}")
        print("✓ Rolled-back agent has correct memory state")
        
        # 6. Test that both agents can continue independently
        print(f"\n--- Testing Independent Continuation ---")
        
        # Continue original agent
        original_continue = original_agent.run("What's the capital of France?")
        print(f"Original agent continues: {original_continue}")
        
        # Continue rolled-back agent on different path
        rolled_back_continue = rolled_back_agent.run("I actually live in Tokyo. Please remember this new information.")
        print(f"Rolled-back agent continues: {rolled_back_continue}")
        
        # Verify they have different conversation states now
        final_original_history = original_agent.internal_session.conversation_history
        final_rolled_back_history = rolled_back_agent.internal_session.conversation_history
        
        self.assertGreater(len(final_original_history), len(final_rolled_back_history))
        print(f"✓ Agents continue independently - Original: {len(final_original_history)}, Rolled-back: {len(final_rolled_back_history)}")
        
        # 7. Verify both sessions exist in database
        all_internal_sessions = self.internal_repo.get_by_external_session(self.external_session.id)
        session_ids = [s.id for s in all_internal_sessions]
        
        self.assertIn(original_session_id, session_ids)
        self.assertIn(rolled_back_session_id, session_ids)
        self.assertEqual(len(all_internal_sessions), 2)
        print("✓ Both sessions exist in database")
        
        print(f"\n🎉 All rollback branching tests passed!")
        
    def test_multiple_rollbacks_create_multiple_branches(self):
        """Test that multiple rollbacks from same checkpoint create separate branches."""
        # Create original agent
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        original_session_id = agent.internal_session.id
        
        # Create conversation and checkpoint
        agent.run("Hello, I'm testing multiple branches.")
        checkpoint_result = agent.create_checkpoint_tool(name="Branch Point")
        checkpoint_id = int(checkpoint_result.split("ID: ")[1].split(")")[0])
        
        agent.run("This message will be lost in rollbacks.")
        
        print(f"\n--- Creating Multiple Branches ---")
        
        # Create first branch
        branch1 = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        # Create second branch
        branch2 = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        # Verify all sessions are different
        self.assertNotEqual(original_session_id, branch1.internal_session.id)
        self.assertNotEqual(original_session_id, branch2.internal_session.id)
        self.assertNotEqual(branch1.internal_session.id, branch2.internal_session.id)
        
        # Verify branch relationships
        self.assertEqual(branch1.internal_session.parent_session_id, original_session_id)
        self.assertEqual(branch2.internal_session.parent_session_id, original_session_id)
        
        # Continue each branch differently
        branch1.run("I'm in branch 1, going to calculate 5 + 3.")
        branch2.run("I'm in branch 2, going to calculate 10 * 4.")
        
        # Verify they have independent states
        branch1_history = branch1.internal_session.conversation_history
        branch2_history = branch2.internal_session.conversation_history
        
        branch1_text = str(branch1_history)
        branch2_text = str(branch2_history)
        
        self.assertIn("branch 1", branch1_text.lower())
        self.assertNotIn("branch 1", branch2_text.lower())
        self.assertIn("branch 2", branch2_text.lower())
        self.assertNotIn("branch 2", branch1_text.lower())
        
        print("✓ Multiple branches created successfully with independent states")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)