"""Comprehensive production-grade integration tests for Session and Checkpoint Repositories.

This test suite:
- Uses a real database connection (configured via DATABASE_URL)
- Validates data integrity and datetime field synchronization for Sessions and Checkpoints
- Specifically verifies the fix for 'checkpoint_data' field name in CheckpointRepository
- Ensures JSON fields are correctly updated with ISO formatted datetimes
"""

import os
import pytest
from datetime import datetime, timezone
from dotenv import load_dotenv

from agentgit.sessions.external_session import ExternalSession
from agentgit.checkpoints.checkpoint import Checkpoint
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.database.models import ExternalSession as ExternalSessionModel
from agentgit.database.models import Checkpoint as CheckpointModel
from agentgit.database.db_config import get_db_connection
from agentgit.auth.user import User
from agentgit.database.repositories.user_repository import UserRepository

load_dotenv()

# ===== CONFIGURATION =====
TEST_DB_URL = os.getenv("DATABASE_URL", "postgresql://agent:git@localhost:5432/agent_git")


@pytest.fixture
def prod_ext_repo():
    """ExternalSessionRepository connected to a real database."""
    return ExternalSessionRepository(db_path=TEST_DB_URL)


@pytest.fixture
def prod_checkpoint_repo():
    """CheckpointRepository connected to a real database."""
    return CheckpointRepository(db_path=TEST_DB_URL)


@pytest.fixture
def test_user():
    """Create a temporary user for linking sessions."""
    repo = UserRepository(db_path=TEST_DB_URL)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    user = User(username=f"sess_test_user_{timestamp}")
    user.set_password("pass")
    return repo.save(user)


class TestExternalSessionProduction:
    """Test ExternalSession repository against real DB."""

    def test_create_session_syncs_created_at(self, prod_ext_repo, test_user):
        """Test that created_at is synchronized to JSON data on creation."""
        # Create session
        session = ExternalSession(
            user_id=test_user.id,
            session_name="Prod Test Session"
        )
        saved_session = prod_ext_repo.create(session)

        assert saved_session.id is not None
        assert saved_session.created_at is not None

        # Verify persistence and data sync in a FRESH database session
        with get_db_connection(TEST_DB_URL) as db_session:
            db_sess = db_session.query(ExternalSessionModel).filter_by(id=saved_session.id).first()
            
            assert db_sess is not None
            assert db_sess.created_at is not None
            
            # CRITICAL: Verify JSON data has the synced created_at
            # This confirms flag_modified worked if it was used, or that the dict was updated correctly
            assert db_sess.data is not None
            assert 'created_at' in db_sess.data
            
            column_iso = db_sess.created_at.isoformat()
            data_iso = db_sess.data['created_at']
            
            # Allow for potential microsecond precision differences if DB truncates
            assert data_iso.startswith(column_iso[:19]), \
                f"JSON created_at ({data_iso}) should match DB column ({column_iso})"

    def test_update_session_persistence(self, prod_ext_repo, test_user):
        """Test that updates are persisted correctly."""
        # Create
        session = ExternalSession(user_id=test_user.id, session_name="Original Name")
        saved = prod_ext_repo.create(session)
        
        # Update
        saved.session_name = "Updated Name"
        saved.metadata = {"env": "prod"}
        prod_ext_repo.update(saved)
        
        # Verify in DB
        with get_db_connection(TEST_DB_URL) as db_session:
            db_sess = db_session.query(ExternalSessionModel).filter_by(id=saved.id).first()
            assert db_sess.session_name == "Updated Name"
            assert db_sess.session_metadata == {"env": "prod"}
            # Check data field sync (ExternalSessionRepo updates data=session_dict)
            assert db_sess.data['session_name'] == "Updated Name"


class TestCheckpointProduction:
    """Test Checkpoint repository against real DB."""

    def test_create_checkpoint_field_name_fix(self, prod_checkpoint_repo, test_user):
        """CRITICAL TEST: Verify the fix for 'checkpoint_data' field name.
        
        If the code still uses db_checkpoint.data['created_at'], this test will fail
        with an AttributeError or the data won't be in the right place.
        """
        # Setup: Need an internal session first (mocking ID is risky with FKs, but let's try 
        # creating a minimal one if needed, or just use a random ID if FK constraints allow.
        # Usually better to create real parent objects).
        
        # We need a real internal session ID usually due to FK constraints
        # Let's quickly create one using raw SQL or repositories to be safe
        from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
        from agentgit.sessions.internal_session import InternalSession
        
        ext_repo = ExternalSessionRepository(db_path=TEST_DB_URL)
        ext_sess = ext_repo.create(ExternalSession(user_id=test_user.id, session_name="CP Parent"))
        
        int_repo = InternalSessionRepository(db_path=TEST_DB_URL)
        int_sess = int_repo.create(InternalSession(
            external_session_id=ext_sess.id,
            langgraph_session_id=f"lg_{datetime.now().timestamp()}"
        ))

        # Create Checkpoint
        checkpoint = Checkpoint(
            internal_session_id=int_sess.id,
            checkpoint_name="Prod Test Checkpoint",
            user_id=test_user.id,
            session_state={"key": "value"}
        )
        
        # This call triggers the code path with the fix:
        # db_checkpoint.checkpoint_data['created_at'] = ...
        saved_cp = prod_checkpoint_repo.create(checkpoint)
        
        assert saved_cp.id is not None
        assert saved_cp.created_at is not None

        # Verify in DB
        with get_db_connection(TEST_DB_URL) as db_session:
            db_cp = db_session.query(CheckpointModel).filter_by(id=saved_cp.id).first()
            
            assert db_cp is not None
            # Verify the column
            assert db_cp.created_at is not None
            
            # Verify the JSON field (checkpoint_data)
            # If the code wrote to .data instead of .checkpoint_data, this might be empty 
            # or the code would have crashed.
            assert db_cp.checkpoint_data is not None
            assert 'created_at' in db_cp.checkpoint_data, \
                "created_at should be present in checkpoint_data JSON"
            
            column_iso = db_cp.created_at.isoformat()
            json_iso = db_cp.checkpoint_data['created_at']
            
            assert json_iso == column_iso, "JSON created_at must match DB column exactly"
