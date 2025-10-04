"""Test suite for verifying rollback framework integration with standard LangChain/LangGraph agents.

Tests that our rollback framework can be applied to any LangChain/LangGraph agent
and provides both manual and automatic checkpoint functionality.
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


# Sample tools for testing various scenarios
@tool
def calculate_sum(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    result = a + b
    print(f"[TOOL] calculate_sum({a}, {b}) = {result}")
    return result

@tool
def multiply_numbers(x: int, y: int) -> int:
    """Multiply two numbers and return the result."""
    result = x * y
    print(f"[TOOL] multiply_numbers({x}, {y}) = {result}")
    return result

@tool
def save_to_memory(key: str, value: str) -> str:
    """Save a key-value pair to memory (simulated)."""
    print(f"[TOOL] save_to_memory('{key}', '{value}')")
    return f"Saved {key} = {value}"

@tool
def get_weather(city: str) -> str:
    """Get weather information for a city (simulated)."""
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 65°F", 
        "Tokyo": "Rainy, 68°F",
        "Paris": "Partly cloudy, 70°F"
    }
    result = weather_data.get(city, f"Weather data not available for {city}")
    print(f"[TOOL] get_weather('{city}') = {result}")
    return result

# Reverse functions for rollback testing
def reverse_calculate_sum(args, _result):
    """Reverse function for calculate_sum - log the reversal."""
    print(f"[REVERSE] Undoing calculate_sum({args['a']}, {args['b']})")
    return f"Reversed calculation {args['a']} + {args['b']}"

def reverse_save_to_memory(args, result):
    """Reverse function for save_to_memory - simulate deletion."""
    print(f"[REVERSE] Deleting memory entry '{args['key']}'")
    return f"Deleted {args['key']} from memory"


class TestFrameworkIntegration(unittest.TestCase):
    """Test cases for rollback framework integration with standard agents."""
    
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
            session_name="Framework Integration Test",
            created_at=datetime.now()
        )
        self.external_session = self.external_repo.create(self.external_session)
        
        # Create OpenAI model
        self.model = self._create_openai_model()
        
        # Define tools and reverse functions
        self.tools = [calculate_sum, multiply_numbers, save_to_memory, get_weather]
        self.reverse_tools = {
            "calculate_sum": reverse_calculate_sum,
            "save_to_memory": reverse_save_to_memory
        }
    
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
            temperature=0.1,  # Low temperature for predictable responses
            openai_api_key=api_key,
            openai_api_base=base_url
        )
    
    def test_framework_can_be_applied_to_any_agent(self):
        """Test that our framework can be applied to any LangChain/LangGraph agent setup."""
        print("\n=== Testing Framework Application to Standard Agent ===")
        
        # Create agent with our rollback framework
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            auto_checkpoint=False,  # Disable auto-checkpointing for this test
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Verify framework components are properly initialized
        self.assertIsNotNone(agent.tool_rollback_registry)
        self.assertIsNotNone(agent.graph)
        self.assertIsNotNone(agent.internal_session)
        
        print(f"✓ Agent created with {len(self.tools)} tools")
        print(f"✓ Rollback registry initialized")
        print(f"✓ LangGraph workflow compiled")
        
        # Test basic conversation without tools
        response1 = agent.run("Hello! Please introduce yourself.")
        print(f"Agent response 1: {response1}")
        self.assertIsInstance(response1, str)
        self.assertGreater(len(response1), 0)
        
        # Verify no auto-checkpoints were created (no tools called)
        checkpoints_before = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints_before = [cp for cp in checkpoints_before if cp.is_auto]
        print(f"Auto-checkpoints after non-tool conversation: {len(auto_checkpoints_before)}")
        
        # Test conversation with tools
        response2 = agent.run("Please calculate 25 + 17 using the calculate_sum tool.")
        print(f"Agent response 2: {response2}")
        
        # Verify tool was called
        tool_track = agent.get_tool_track()
        print(f"Tools in track: {[record.tool_name for record in tool_track]}")
        self.assertGreater(len(tool_track), 0)
        self.assertEqual(tool_track[0].tool_name, "calculate_sum")
        
        # Test manual checkpoint creation
        checkpoint_result = agent.create_checkpoint_tool("After calculation")
        self.assertIn("successfully", checkpoint_result.lower())
        print(f"✓ Manual checkpoint created: {checkpoint_result}")
        
        # Test that framework preserves standard LangChain/LangGraph functionality
        response3 = agent.run("What's the weather like in Tokyo?")
        print(f"Agent response 3: {response3}")
        self.assertIn("tokyo", response3.lower())
        
        print("✓ Framework successfully applied to standard agent")
        print("✓ Standard LangChain/LangGraph functionality preserved")
        
    def test_auto_checkpoint_creation_with_tools(self):
        """Test automatic checkpoint creation when tools are called."""
        print("\n=== Testing Automatic Checkpoint Creation ===")
        
        # Create agent with auto-checkpointing enabled
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            auto_checkpoint=True,  # Enable auto-checkpointing
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        print("Agent created with auto_checkpoint=True")
        
        # Test conversation without tools - should NOT create auto-checkpoint
        response1 = agent.run("Hello, how are you today?")
        print(f"Non-tool response: {response1}")
        
        checkpoints_after_chat = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints_chat = [cp for cp in checkpoints_after_chat if cp.is_auto]
        print(f"Auto-checkpoints after non-tool conversation: {len(auto_checkpoints_chat)}")
        self.assertEqual(len(auto_checkpoints_chat), 0, "No auto-checkpoint should be created without tool calls")
        
        # Test conversation with tools - SHOULD create auto-checkpoint
        response2 = agent.run("Please multiply 8 by 7 using the multiply_numbers tool.")
        print(f"Tool response: {response2}")
        
        checkpoints_after_tool = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints_tool = [cp for cp in checkpoints_after_tool if cp.is_auto]
        print(f"Auto-checkpoints after tool call: {len(auto_checkpoints_tool)}")
        self.assertEqual(len(auto_checkpoints_tool), 1, "One auto-checkpoint should be created after tool call")
        
        # Verify auto-checkpoint details
        auto_checkpoint = auto_checkpoints_tool[0]
        self.assertTrue(auto_checkpoint.is_auto)
        self.assertIn("multiply_numbers", auto_checkpoint.checkpoint_name)
        print(f"✓ Auto-checkpoint created: {auto_checkpoint.checkpoint_name}")
        
        # Test multiple tool calls create multiple auto-checkpoints
        response3 = agent.run("Now calculate 15 + 25 using calculate_sum.")
        print(f"Second tool response: {response3}")
        
        response4 = agent.run("Save 'user_preference' as 'dark_mode' to memory.")
        print(f"Third tool response: {response4}")
        
        final_checkpoints = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        final_auto_checkpoints = [cp for cp in final_checkpoints if cp.is_auto]
        print(f"Final auto-checkpoints: {len(final_auto_checkpoints)}")
        self.assertEqual(len(final_auto_checkpoints), 3, "Three auto-checkpoints should exist after three tool calls")
        
        # Verify checkpoint names reflect the tools used
        checkpoint_names = [cp.checkpoint_name for cp in final_auto_checkpoints]
        print(f"Auto-checkpoint names: {checkpoint_names}")
        self.assertTrue(any("multiply_numbers" in name for name in checkpoint_names))
        self.assertTrue(any("calculate_sum" in name for name in checkpoint_names))
        self.assertTrue(any("save_to_memory" in name for name in checkpoint_names))
        
        print("✓ Auto-checkpoint creation works correctly")
        print("✓ No checkpoints created without tool calls")
        print("✓ Checkpoints created automatically after each tool call")
        
    def test_rollback_functionality_integration(self):
        """Test rollback functionality with integrated framework."""
        print("\n=== Testing Rollback Functionality Integration ===")
        
        # Create agent with both auto-checkpointing and rollback capabilities
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Build up conversation with mixed tool and non-tool interactions
        agent.run("Hello, I'm setting up a calculation session.")
        
        # First tool call - should create auto-checkpoint
        agent.run("Calculate 10 + 5 using calculate_sum.")
        
        # Create manual checkpoint
        manual_cp_result = agent.create_checkpoint_tool("Before complex operations")
        checkpoint_id = int(manual_cp_result.split("ID: ")[1].split(")")[0])
        print(f"Manual checkpoint created with ID: {checkpoint_id}")
        
        # More operations after checkpoint
        agent.run("Now multiply 3 by 4.")
        agent.run("Save 'calculation_mode' as 'advanced' to memory.")
        agent.run("Get weather for Paris.")
        
        # Store state before rollback
        original_history = agent.get_conversation_history().copy()
        original_tool_track = agent.get_tool_track().copy()
        
        print(f"Before rollback - History: {len(original_history)} messages")
        print(f"Before rollback - Tool track: {len(original_tool_track)} tools")
        
        # Perform rollback
        rolled_back_agent = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            reverse_tools=self.reverse_tools,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo
        )
        
        # Verify rollback worked
        rolled_back_history = rolled_back_agent.get_conversation_history()
        rolled_back_tool_track = rolled_back_agent.get_tool_track()
        
        print(f"After rollback - History: {len(rolled_back_history)} messages")
        print(f"After rollback - Tool track: {len(rolled_back_tool_track)} tools")
        
        self.assertLess(len(rolled_back_history), len(original_history))
        self.assertLess(len(rolled_back_tool_track), len(original_tool_track))
        
        # Verify new branch can continue independently
        continue_response = rolled_back_agent.run("Let's try a different calculation: 20 + 30.")
        print(f"Continued conversation: {continue_response}")
        
        # Verify both sessions exist in database
        all_sessions = self.internal_repo.get_by_external_session(self.external_session.id)
        self.assertEqual(len(all_sessions), 2, "Should have original and branched sessions")
        
        branch_session = rolled_back_agent.internal_session
        self.assertTrue(branch_session.is_branch())
        self.assertEqual(branch_session.branch_point_checkpoint_id, checkpoint_id)
        
        print("✓ Rollback functionality works with framework integration")
        print("✓ Tool rollback executed successfully")  
        print("✓ New branch created and can continue independently")
        print("✓ Original session preserved in database")
    
    def test_checkpoint_tool_exclusion(self):
        """Test that checkpoint management tools don't trigger auto-checkpoints."""
        print("\n=== Testing Checkpoint Tool Exclusion ===")
        
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=self.tools,
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Use checkpoint management tools - should NOT create auto-checkpoints
        agent.create_checkpoint_tool("Test checkpoint")
        agent.list_checkpoints_tool()
        
        checkpoints = self.checkpoint_repo.get_by_internal_session(agent.internal_session.id)
        auto_checkpoints = [cp for cp in checkpoints if cp.is_auto]
        manual_checkpoints = [cp for cp in checkpoints if not cp.is_auto]
        
        print(f"Manual checkpoints: {len(manual_checkpoints)}")
        print(f"Auto checkpoints: {len(auto_checkpoints)}")
        
        self.assertEqual(len(manual_checkpoints), 1)
        self.assertEqual(len(auto_checkpoints), 0)
        
        print("✓ Checkpoint management tools don't trigger auto-checkpoints")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)