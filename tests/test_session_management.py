"""Test suite for session management functionality.

Tests external/internal session relationships, branching, and session lifecycle.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import warnings
from datetime import datetime
from langchain_openai import ChatOpenAI

# Suppress Pydantic V2 deprecation warnings from LangChain
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*__fields__.*")

from src.sessions.external_session import ExternalSession
from src.sessions.internal_session import InternalSession
from src.database.repositories.external_session_repository import ExternalSessionRepository
from src.database.repositories.internal_session_repository import InternalSessionRepository
from src.database.repositories.checkpoint_repository import CheckpointRepository
from src.database.repositories.user_repository import UserRepository
from src.auth.user import User
from src.agents.agent_service import AgentService
from src.checkpoints.checkpoint import Checkpoint


class TestSessionManagement(unittest.TestCase):
    """Test cases for external and internal session management."""
    
    def setUp(self):
        """Set up test environment with repositories."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize repositories
        # Create user repository first to ensure users table exists
        self.user_repo = UserRepository(db_path=self.db_path)
        
        # Create a default test user (user_id=1 will be used by most tests)
        # The rootusr is created automatically by UserRepository
        # Most tests use user_id=1 which is rootusr
        
        self.external_repo = ExternalSessionRepository(db_path=self.db_path)
        self.internal_repo = InternalSessionRepository(db_path=self.db_path)
        self.checkpoint_repo = CheckpointRepository(db_path=self.db_path)
        
        # Create OpenAI model
        self.model = self._create_openai_model()
        
        # Initialize agent service with model config
        self.agent_service = AgentService(model_config={
            "id": "gpt-4o-mini",
            "temperature": 0.3,
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("BASE_URL")
        })
        self.agent_service.external_session_repo = self.external_repo
        self.agent_service.internal_session_repo = self.internal_repo
        self.agent_service.checkpoint_repo = self.checkpoint_repo
    
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
            temperature=0.3,
            openai_api_key=api_key,
            openai_api_base=base_url
        )
    
    def test_external_session_creation(self):
        """Test creating and managing external sessions."""
        # Create external session
        session = ExternalSession(
            user_id=1,
            session_name="My Chat Session",
            created_at=datetime.now()
        )
        
        # Save to repository
        saved_session = self.external_repo.create(session)
        
        self.assertIsNotNone(saved_session.id)
        self.assertEqual(saved_session.session_name, "My Chat Session")
        self.assertEqual(saved_session.user_id, 1)
        self.assertTrue(saved_session.is_active)
        
        # Retrieve session
        retrieved = self.external_repo.get_by_id(saved_session.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_name, "My Chat Session")
        
        # Update session
        retrieved.session_name = "Updated Chat Session"
        retrieved.metadata["description"] = "Test session"
        success = self.external_repo.update(retrieved)
        
        self.assertTrue(success)
        
        # Verify update
        updated = self.external_repo.get_by_id(saved_session.id)
        self.assertEqual(updated.session_name, "Updated Chat Session")
        self.assertEqual(updated.metadata["description"], "Test session")
    
    def test_internal_session_creation_and_linking(self):
        """Test creating internal sessions and linking to external sessions."""
        # Create external session
        external = ExternalSession(
            user_id=1,
            session_name="Parent Session"
        )
        external = self.external_repo.create(external)
        
        # Create internal session
        internal = InternalSession(
            external_session_id=external.id,
            langgraph_session_id="langgraph_abc123",
            session_state={"initial": True},
            created_at=datetime.now(),
            is_current=True
        )
        
        # Save internal session
        saved_internal = self.internal_repo.create(internal)
        
        self.assertIsNotNone(saved_internal.id)
        self.assertEqual(saved_internal.external_session_id, external.id)
        self.assertTrue(saved_internal.is_current)
        
        # Link internal session to external
        success = self.external_repo.add_internal_session(
            external.id,
            internal.langgraph_session_id
        )
        self.assertTrue(success)
        
        # Verify linking
        updated_external = self.external_repo.get_by_id(external.id)
        self.assertIn("langgraph_abc123", updated_external.internal_session_ids)
        self.assertEqual(updated_external.current_internal_session_id, "langgraph_abc123")
    
    def test_multiple_internal_sessions(self):
        """Test managing multiple internal sessions within one external session."""
        # Create external session
        external = ExternalSession(user_id=1, session_name="Multi-Internal")
        external = self.external_repo.create(external)
        
        # Create multiple internal sessions
        internal_ids = []
        for i in range(3):
            internal = InternalSession(
                external_session_id=external.id,
                langgraph_session_id=f"langgraph_{i:03d}",
                session_state={"session_num": i},
                is_current=(i == 2)  # Last one is current
            )
            saved = self.internal_repo.create(internal)
            internal_ids.append(saved.id)
            
            # Add to external session
            self.external_repo.add_internal_session(external.id, internal.langgraph_session_id)
        
        # Get all internal sessions
        all_internals = self.internal_repo.get_by_external_session(external.id)
        self.assertEqual(len(all_internals), 3)
        
        # Verify current session
        current = self.internal_repo.get_current_session(external.id)
        self.assertIsNotNone(current)
        self.assertEqual(current.langgraph_session_id, "langgraph_002")
        
        # Switch current session
        success = self.internal_repo.set_current_session(internal_ids[0])
        self.assertTrue(success)
        
        # Verify switch
        new_current = self.internal_repo.get_current_session(external.id)
        self.assertEqual(new_current.id, internal_ids[0])
    
    def test_branching_from_checkpoint(self):
        """Test creating branches from checkpoints."""
        # Create external session
        external = ExternalSession(user_id=1, session_name="Branching Test")
        external = self.external_repo.create(external)
        
        # Create original internal session
        original = InternalSession(
            external_session_id=external.id,
            langgraph_session_id="langgraph_original",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            session_state={"counter": 5}
        )
        original = self.internal_repo.create(original)
        
        # Create checkpoint from original session
        checkpoint = Checkpoint.from_internal_session(
            original,
            checkpoint_name="Branch Point"
        )
        checkpoint = self.checkpoint_repo.create(checkpoint)
        
        # Create branch from checkpoint
        branch = InternalSession.create_branch_from_checkpoint(
            checkpoint=checkpoint,
            external_session_id=external.id,
            parent_session_id=original.id
        )
        branch = self.internal_repo.create(branch)
        
        # Verify branch properties
        self.assertNotEqual(branch.id, original.id)
        self.assertEqual(branch.parent_session_id, original.id)
        self.assertEqual(branch.branch_point_checkpoint_id, checkpoint.id)
        self.assertTrue(branch.is_branch())
        
        # Verify branch inherited conversation history
        self.assertEqual(len(branch.conversation_history), 2)
        self.assertEqual(branch.conversation_history[0]["content"], "Hello")
        
        # Verify branch inherited session state
        self.assertEqual(branch.session_state["counter"], 5)
        
        # Verify branch metadata
        self.assertIn("branched_from", branch.metadata)
        self.assertEqual(branch.metadata["branched_from"], "Branch Point")
    
    def test_session_lifecycle_with_agent_service(self):
        """Test complete session lifecycle using AgentService."""
        # Create external session
        external = ExternalSession(user_id=1, session_name="Lifecycle Test")
        external = self.external_repo.create(external)
        
        # Create new agent (creates internal session automatically)
        agent = self.agent_service.create_new_agent(
            external_session_id=external.id,
            session_name="Initial Session"
        )
        
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.internal_session)
        self.assertEqual(agent.external_session_id, external.id)
        
        # Simulate some conversation
        agent.internal_session.add_message("user", "What's the weather?")
        agent.internal_session.add_message("assistant", "I can't check real weather.")
        agent.internal_session.session_state["interaction_count"] = 1
        self.internal_repo.update(agent.internal_session)
        
        # Create checkpoint
        checkpoint_result = agent.create_checkpoint_tool("Before rollback")
        self.assertIn("created successfully", checkpoint_result)
        
        # Continue conversation
        agent.internal_session.add_message("user", "Tell me a joke")
        agent.internal_session.add_message("assistant", "Why did the chicken cross the road?")
        agent.internal_session.session_state["interaction_count"] = 2
        self.internal_repo.update(agent.internal_session)
        
        original_session_id = agent.internal_session.id
        
        # Get checkpoint for rollback
        checkpoints = self.checkpoint_repo.get_by_internal_session(original_session_id)
        checkpoint = checkpoints[0]
        
        # Perform rollback
        rolled_back_agent = self.agent_service.rollback_to_checkpoint(
            external_session_id=external.id,
            checkpoint_id=checkpoint.id,
            rollback_tools=False  # No actual tools to rollback in test
        )
        
        self.assertIsNotNone(rolled_back_agent)
        
        # Verify rollback created new internal session
        self.assertNotEqual(rolled_back_agent.internal_session.id, original_session_id)
        
        # Verify rolled back state
        self.assertEqual(len(rolled_back_agent.internal_session.conversation_history), 2)
        self.assertEqual(rolled_back_agent.internal_session.session_state["interaction_count"], 1)
        
        # List all internal sessions
        all_sessions = self.agent_service.list_internal_sessions(external.id)
        self.assertEqual(len(all_sessions), 2)  # Original + branch
        
        # Resume from specific internal session
        resumed_agent = self.agent_service.resume_agent(
            external_session_id=external.id,
            internal_session_id=original_session_id
        )
        
        self.assertIsNotNone(resumed_agent)
        self.assertEqual(resumed_agent.internal_session.id, original_session_id)
        
        # Verify resumed agent has full history
        self.assertEqual(len(resumed_agent.internal_session.conversation_history), 4)
    
    def test_session_statistics(self):
        """Test session statistics and metadata tracking."""
        # Create sessions
        external = ExternalSession(user_id=1, session_name="Stats Test")
        external = self.external_repo.create(external)
        
        internal = InternalSession(
            external_session_id=external.id,
            langgraph_session_id="langgraph_stats"
        )
        
        # Add conversation
        for i in range(5):
            internal.add_message("user", f"Question {i}")
            internal.add_message("assistant", f"Answer {i}")
            internal.increment_tool_count()
        
        internal = self.internal_repo.create(internal)
        
        # Create checkpoints
        for i in range(3):
            checkpoint = Checkpoint.from_internal_session(
                internal,
                checkpoint_name=f"Checkpoint {i}"
            )
            self.checkpoint_repo.create(checkpoint)
            internal.checkpoint_count += 1
        
        self.internal_repo.update(internal)
        
        # Get statistics
        stats = internal.get_statistics()
        
        self.assertEqual(stats["total_messages"], 10)
        self.assertEqual(stats["user_messages"], 5)
        self.assertEqual(stats["assistant_messages"], 5)
        self.assertEqual(stats["checkpoints"], 3)
        self.assertEqual(stats["tool_invocations"], 5)
        self.assertTrue(stats["is_active"])
        self.assertFalse(stats["is_branch"])
        
        # Update external session statistics
        external.total_checkpoints = 3
        external.branch_count = 0
        self.external_repo.update(external)
        
        # Get branch info
        branch_info = external.get_branch_info()
        self.assertEqual(branch_info["total_branches"], 0)
        self.assertFalse(branch_info["is_branched"])
    
    def test_session_deletion_cascade(self):
        """Test that deleting external session cascades to internal sessions."""
        # Create external session
        external = ExternalSession(user_id=1, session_name="Delete Test")
        external = self.external_repo.create(external)
        
        # Create internal sessions
        for i in range(2):
            internal = InternalSession(
                external_session_id=external.id,
                langgraph_session_id=f"langgraph_del_{i}"
            )
            internal = self.internal_repo.create(internal)
            
            # Create checkpoints
            checkpoint = Checkpoint.from_internal_session(
                internal,
                checkpoint_name=f"Checkpoint {i}"
            )
            self.checkpoint_repo.create(checkpoint)
        
        # Verify sessions exist
        internals = self.internal_repo.get_by_external_session(external.id)
        self.assertEqual(len(internals), 2)
        
        # Delete external session
        success = self.external_repo.delete(external.id)
        self.assertTrue(success)
        
        # Verify cascade deletion
        internals = self.internal_repo.get_by_external_session(external.id)
        self.assertEqual(len(internals), 0)
        
        # Note: Checkpoint cascade deletion depends on foreign key constraints
        # in the actual SQLite database
    
    def test_session_ownership(self):
        """Test session ownership verification."""
        # Create a second test user
        user2 = User(username="testuser2")
        user2.set_password("password")
        user2 = self.user_repo.save(user2)
        
        # Create sessions for different users
        session1 = ExternalSession(user_id=1, session_name="User 1 Session")  # rootusr
        session1 = self.external_repo.create(session1)
        
        session2 = ExternalSession(user_id=user2.id, session_name="User 2 Session")
        session2 = self.external_repo.create(session2)
        
        # Test ownership
        owns1 = self.external_repo.check_ownership(session1.id, user_id=1)
        self.assertTrue(owns1)
        
        owns2 = self.external_repo.check_ownership(session1.id, user_id=user2.id)
        self.assertFalse(owns2)
        
        owns3 = self.external_repo.check_ownership(session2.id, user_id=user2.id)
        self.assertTrue(owns3)
        
        # Test user session retrieval
        user1_sessions = self.external_repo.get_user_sessions(user_id=1)
        self.assertEqual(len(user1_sessions), 1)
        self.assertEqual(user1_sessions[0].session_name, "User 1 Session")
        
        user2_sessions = self.external_repo.get_user_sessions(user_id=2)
        self.assertEqual(len(user2_sessions), 1)
        self.assertEqual(user2_sessions[0].session_name, "User 2 Session")
    
    def test_session_soft_delete(self):
        """Test soft deletion (deactivation) of sessions."""
        # Create session
        session = ExternalSession(user_id=1, session_name="Soft Delete Test")
        session = self.external_repo.create(session)
        
        self.assertTrue(session.is_active)
        
        # Deactivate session
        success = self.external_repo.deactivate(session.id)
        self.assertTrue(success)
        
        # Verify deactivation
        deactivated = self.external_repo.get_by_id(session.id)
        self.assertIsNotNone(deactivated)  # Still exists
        self.assertFalse(deactivated.is_active)
        
        # Test active_only filter
        active_sessions = self.external_repo.get_user_sessions(
            user_id=1,
            active_only=True
        )
        self.assertEqual(len(active_sessions), 0)
        
        all_sessions = self.external_repo.get_user_sessions(
            user_id=1,
            active_only=False
        )
        self.assertEqual(len(all_sessions), 1)


if __name__ == "__main__":
    unittest.main()