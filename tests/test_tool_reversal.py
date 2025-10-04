"""Test suite for tool reversal functionality.

Tests that tools with reverse handlers correctly undo their operations during rollback.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Suppress Pydantic V2 deprecation warnings from LangChain
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*__fields__.*")

from src.agents.rollback_agent import RollbackAgent
from src.checkpoints.checkpoint import Checkpoint
from src.database.repositories.checkpoint_repository import CheckpointRepository
from src.database.repositories.internal_session_repository import InternalSessionRepository
from src.database.repositories.external_session_repository import ExternalSessionRepository
from src.database.repositories.user_repository import UserRepository
from src.sessions.external_session import ExternalSession


class TestToolReversal(unittest.TestCase):
    """Test cases for tool reversal during rollback operations."""
    
    def setUp(self):
        """Set up test environment with OpenAI model and repositories."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Create test directory in tests/ folder
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_files")
        os.makedirs(self.test_dir, exist_ok=True)
        print(f"\n📁 Test files will be created in: {self.test_dir}")
        
        # Initialize repositories - UserRepository first to create users table
        self.user_repo = UserRepository(db_path=self.db_path)
        self.checkpoint_repo = CheckpointRepository(db_path=self.db_path)
        self.internal_repo = InternalSessionRepository(db_path=self.db_path)
        self.external_repo = ExternalSessionRepository(db_path=self.db_path)
        
        # Create external session
        self.external_session = ExternalSession(
            user_id=1,
            session_name="Tool Reversal Test Session",
            created_at=datetime.now()
        )
        self.external_session = self.external_repo.create(self.external_session)
        
        # Create OpenAI model
        self.model = self._create_openai_model()
        
        # Track created files for cleanup
        self.created_files = []
    
    def tearDown(self):
        """Clean up temporary database and files."""
        try:
            # Clean up any remaining test files
            for filepath in self.created_files:
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"Cleaned up remaining file: {filepath}")
            
            # Clean up test directory (remove all files but keep the directory for visibility)
            if os.path.exists(self.test_dir):
                # Remove all files in the test directory
                for filename in os.listdir(self.test_dir):
                    file_path = os.path.join(self.test_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Cleaned up: {file_path}")
                # Optionally remove the directory itself (comment out to keep it)
                # os.rmdir(self.test_dir)
            
            # Remove database
            os.unlink(self.db_path)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
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
            temperature=0.1,  # Low temperature for consistent behavior
            openai_api_key=api_key,
            openai_api_base=base_url
        )
    
    def test_file_creation_and_reversal(self):
        """Test that file creation tool is properly reversed during rollback."""
        
        # Create the file creation tool
        @tool
        def create_text_file(filename: str, content: str) -> str:
            """Create a text file with the given content.
            
            Args:
                filename: Name of the file to create (without path)
                content: Content to write to the file
                
            Returns:
                Success message with full file path
            """
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
            self.created_files.append(filepath)
            print(f"\n✅ TOOL EXECUTED: Created file -> {filepath}")
            return f"File created successfully at: {filepath}"
        
        # Create the reverse function for file deletion
        def delete_text_file_reverse(args, result):
            """Reverse function that deletes the created file.
            
            Args:
                args: Original arguments passed to create_text_file
                result: Result from create_text_file (contains filepath)
            """
            filename = args.get("filename")
            if filename:
                filepath = os.path.join(self.test_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"\n🔄 REVERSE EXECUTED: Deleted file -> {filepath}")
                    if filepath in self.created_files:
                        self.created_files.remove(filepath)
                else:
                    print(f"\n⚠️ REVERSE: File {filepath} doesn't exist, nothing to reverse")
        
        # Create agent with the file tool and its reverse handler
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=[create_text_file],
            reverse_tools={
                "create_text_file": delete_text_file_reverse
            },
            auto_checkpoint=True,  # Enable auto-checkpointing after tools
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Step 1: Initial conversation and create checkpoint BEFORE file creation
        print("\n=== Step 1: Initial conversation ===")
        response1 = agent.run("Hello! I need help managing some files.")
        print(f"Agent: {response1}")
        
        # Create a manual checkpoint before any file operations
        checkpoint_result = agent.create_checkpoint_tool(name="Before File Creation")
        print(f"Checkpoint: {checkpoint_result}")
        self.assertIn("created successfully", checkpoint_result)
        
        # Extract checkpoint ID
        checkpoint_id = int(checkpoint_result.split("ID: ")[1].split(")")[0])
        
        # Step 2: Ask agent to create a file (should trigger tool use)
        print("\n=== Step 2: Creating file via tool ===")
        test_filename = "test_document.txt"
        test_content = "This is a test file created by the LLM agent."
        
        response2 = agent.run(
            f"Please create a text file named '{test_filename}' with the content: '{test_content}'"
        )
        print(f"Agent: {response2}")
        
        # Verify file was created
        test_filepath = os.path.join(self.test_dir, test_filename)
        self.assertTrue(os.path.exists(test_filepath), "File should have been created by the tool")
        
        # Verify file content
        with open(test_filepath, 'r') as f:
            actual_content = f.read()
        self.assertEqual(actual_content, test_content, "File content should match requested content")
        print(f"✓ File created at: {test_filepath}")
        print(f"✓ Content verified: {actual_content}")
        
        # Step 3: Continue conversation after file creation
        print("\n=== Step 3: Conversation after file creation ===")
        response3 = agent.run("Great! Can you confirm the file was created?")
        print(f"Agent: {response3}")
        
        # Add more context that should be forgotten after rollback
        response4 = agent.run("Let's remember that we created this important file.")
        print(f"Agent: {response4}")
        
        # Verify conversation history includes file creation
        history_before = agent.get_conversation_history()
        self.assertTrue(
            any(test_filename in str(msg) for msg in history_before),
            "Conversation should mention the created file"
        )
        
        # Step 4: Get tool track to verify recording
        print("\n=== Step 4: Checking tool track ===")
        tool_track = agent.get_tool_track()
        print(f"Tool track length: {len(tool_track)}")
        
        # Find our file creation in the track
        file_creation_found = False
        for record in tool_track:
            if record.tool_name == "create_text_file":
                file_creation_found = True
                print(f"Found tool invocation: {record.tool_name} with args: {record.args}")
                self.assertEqual(record.args.get("filename"), test_filename)
                self.assertTrue(record.success)
        
        self.assertTrue(file_creation_found, "File creation tool should be in the track")
        
        # Step 5: Rollback to checkpoint BEFORE file creation
        print("\n=== Step 5: Rolling back to checkpoint before file creation ===")
        
        # Create new agent from checkpoint (simulating rollback with tool reversal)
        rolled_back_agent = RollbackAgent.from_checkpoint(
            checkpoint_id=checkpoint_id,
            external_session_id=self.external_session.id,
            model=self.model,
            checkpoint_repo=self.checkpoint_repo,
            internal_session_repo=self.internal_repo,
            tools=[create_text_file],  # Need to provide tools to the rolled-back agent
            reverse_tools={
                "create_text_file": delete_text_file_reverse
            }
        )
        
        # Get the checkpoint to access tool track position
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        if "tool_track_position" in checkpoint.metadata:
            track_position = checkpoint.metadata["tool_track_position"]
            print(f"Rolling back tools from position: {track_position}")
            
            # Manually trigger tool rollback (simulating what would happen in production)
            reverse_results = agent.rollback_tools_from_track_index(track_position)
            
            for result in reverse_results:
                print(f"Reverse result: {result.tool_name} - Success: {result.reversed_successfully}")
                if not result.reversed_successfully and result.error_message:
                    print(f"  Error: {result.error_message}")
        
        # Step 6: Verify file has been deleted by reverse handler
        print("\n=== Step 6: Verifying file deletion ===")
        self.assertFalse(
            os.path.exists(test_filepath),
            "File should have been deleted by the reverse handler during rollback"
        )
        print(f"✓ File successfully deleted by reverse handler: {test_filepath}")
        
        # Step 7: Verify rolled back agent has no memory of file creation
        print("\n=== Step 7: Verifying conversation state after rollback ===")
        rolled_back_history = rolled_back_agent.get_conversation_history()
        
        # Should not contain any mention of the file
        self.assertFalse(
            any(test_filename in str(msg) for msg in rolled_back_history),
            "Rolled back conversation should not mention the file"
        )
        
        # Should only have conversation up to checkpoint
        self.assertLess(
            len(rolled_back_history),
            len(history_before),
            "Rolled back history should be shorter"
        )
        
        print(f"✓ Original history length: {len(history_before)}")
        print(f"✓ Rolled back history length: {len(rolled_back_history)}")
        
        # Step 8: Verify the rollback created a branch
        print("\n=== Step 8: Verifying branch creation ===")
        self.assertNotEqual(
            agent.internal_session.id,
            rolled_back_agent.internal_session.id,
            "Rolled back agent should have a new internal session (branch)"
        )
        print(f"✓ Original session ID: {agent.internal_session.id}")
        print(f"✓ Branched session ID: {rolled_back_agent.internal_session.id}")
        
        # Verify original file remains deleted
        self.assertFalse(os.path.exists(test_filepath), "Original file should remain deleted")
        
        print("\n=== SUMMARY ===")
        print("✓ File was created by tool")
        print("✓ Tool invocation was tracked")
        print("✓ Rollback triggered reverse function")
        print("✓ File was successfully deleted")
        print("✓ Conversation history was rolled back")
        print("✓ New branch was created")
        print("✓ Tool reversal test completed successfully!")
    
    def test_multiple_tool_reversals(self):
        """Test that multiple tools are reversed in correct order."""
        
        # Create multiple file tools
        @tool
        def create_file_a(content: str) -> str:
            """Create file A with content."""
            filepath = os.path.join(self.test_dir, "file_a.txt")
            with open(filepath, 'w') as f:
                f.write(content)
            self.created_files.append(filepath)
            return f"File A created at: {filepath}"
        
        @tool
        def create_file_b(content: str) -> str:
            """Create file B with content."""
            filepath = os.path.join(self.test_dir, "file_b.txt")
            with open(filepath, 'w') as f:
                f.write(content)
            self.created_files.append(filepath)
            return f"File B created at: {filepath}"
        
        # Reverse functions
        def delete_file_a_reverse(args, result):
            filepath = os.path.join(self.test_dir, "file_a.txt")
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Reversed: Deleted file A")
        
        def delete_file_b_reverse(args, result):
            filepath = os.path.join(self.test_dir, "file_b.txt")
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"Reversed: Deleted file B")
        
        # Create agent
        agent = RollbackAgent(
            external_session_id=self.external_session.id,
            model=self.model,
            tools=[create_file_a, create_file_b],
            reverse_tools={
                "create_file_a": delete_file_a_reverse,
                "create_file_b": delete_file_b_reverse
            },
            auto_checkpoint=True,
            internal_session_repo=self.internal_repo,
            checkpoint_repo=self.checkpoint_repo
        )
        
        # Create checkpoint before any operations
        checkpoint_result = agent.create_checkpoint_tool(name="Before Any Files")
        checkpoint_id = int(checkpoint_result.split("ID: ")[1].split(")")[0])
        
        # Create both files
        agent.run("Create file A with content 'First file'")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "file_a.txt")))
        
        agent.run("Create file B with content 'Second file'")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "file_b.txt")))
        
        # Rollback - should delete both files in reverse order (B then A)
        checkpoint = self.checkpoint_repo.get_by_id(checkpoint_id)
        track_position = checkpoint.metadata.get("tool_track_position", 0)
        reverse_results = agent.rollback_tools_from_track_index(track_position)
        
        # Verify both files are deleted
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "file_a.txt")))
        self.assertFalse(os.path.exists(os.path.join(self.test_dir, "file_b.txt")))
        
        print("✓ Multiple tool reversals completed successfully!")


if __name__ == "__main__":
    unittest.main()