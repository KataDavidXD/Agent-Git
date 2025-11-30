"""Comprehensive production-grade integration tests for UserRepository.

This test suite:
- Uses a real database connection (configured via DATABASE_URL)
- Tests all UserRepository methods with production scenarios
- Validates data integrity and datetime field synchronization
- Uses dynamic usernames with timestamps to avoid conflicts
- Provides detailed assertions and error messages
"""

import os
import pytest
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from agentgit.auth.user import User
from agentgit.database.repositories.user_repository import UserRepository
from agentgit.database.models import User as UserModel
from agentgit.database.db_config import get_db_connection

load_dotenv()

# ===== CONFIGURATION =====
TEST_DB_URL = os.getenv("DATABASE_URL", "postgresql://agent:git@localhost:5432/agent_git")


@pytest.fixture
def prod_user_repo():
    """Repository connected to a real database for production testing."""
    return UserRepository(db_path=TEST_DB_URL)


@pytest.fixture
def unique_username():
    """Generate a unique username for each test."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"test_user_{timestamp}"


class TestUserRepositoryBasicOperations:
    """Test basic CRUD operations."""
    
    def test_init_creates_root_user(self, prod_user_repo):
        """Test that initialization creates the default root user."""
        root = prod_user_repo.find_by_username("rootusr")
        
        assert root is not None, "Root user should be created automatically"
        assert root.is_admin is True, "Root user should be admin"
        assert root.username == "rootusr"
        assert root.verify_password("1234"), "Root user default password should be '1234'"
    
    def test_save_new_user(self, prod_user_repo, unique_username):
        """Test creating and saving a new user."""
        # Create user
        new_user = User(
            username=unique_username,
            is_admin=False,
            session_limit=5
        )
        new_user.set_password("SecurePass123")
        
        # Save user
        saved_user = prod_user_repo.save(new_user)
        
        # Assertions
        assert saved_user.id is not None, "Saved user should have an ID"
        assert saved_user.username == unique_username
        assert saved_user.session_limit == 5
        assert saved_user.created_at is not None, "created_at should be populated"
        assert saved_user.last_login is None, "New user should not have last_login"
        
        # Verify password
        assert saved_user.verify_password("SecurePass123"), "Password should be correct"
        
        # Verify data field synchronization
        with get_db_connection(TEST_DB_URL) as db_session:
            db_user = db_session.query(UserModel).filter_by(id=saved_user.id).first()
            assert db_user.data.get('created_at') is not None, \
                "data['created_at'] should be synchronized with database column"
    
    def test_save_update_existing_user(self, prod_user_repo, unique_username):
        """Test updating an existing user."""
        # Create user
        user = User(username=unique_username, is_admin=False)
        user.set_password("OldPass")
        saved_user = prod_user_repo.save(user)
        original_id = saved_user.id
        
        # Update user
        saved_user.username = f"{unique_username}_updated"
        saved_user.is_admin = True
        saved_user.session_limit = 10
        saved_user.set_password("NewPass")
        
        updated_user = prod_user_repo.save(saved_user)
        
        # Assertions
        assert updated_user.id == original_id, "ID should not change on update"
        assert updated_user.username == f"{unique_username}_updated"
        assert updated_user.is_admin is True
        assert updated_user.session_limit == 10
        assert updated_user.verify_password("NewPass"), "Password should be updated"
        assert not updated_user.verify_password("OldPass"), "Old password should not work"


class TestUserRepositoryFindOperations:
    """Test various find/query operations."""
    
    def test_find_by_id(self, prod_user_repo, unique_username):
        """Test finding user by ID."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Find by ID
        found_user = prod_user_repo.find_by_id(saved_user.id)
        
        assert found_user is not None
        assert found_user.id == saved_user.id
        assert found_user.username == unique_username
    
    def test_find_by_id_nonexistent(self, prod_user_repo):
        """Test finding non-existent user by ID."""
        found_user = prod_user_repo.find_by_id(999999)
        assert found_user is None
    
    def test_find_by_username(self, prod_user_repo, unique_username):
        """Test finding user by username."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        prod_user_repo.save(user)
        
        # Find by username
        found_user = prod_user_repo.find_by_username(unique_username)
        
        assert found_user is not None
        assert found_user.username == unique_username
    
    def test_find_by_username_nonexistent(self, prod_user_repo):
        """Test finding non-existent user by username."""
        found_user = prod_user_repo.find_by_username("nonexistent_user_xyz")
        assert found_user is None
    
    def test_find_all(self, prod_user_repo, unique_username):
        """Test finding all users."""
        # Create multiple users
        user1 = User(username=f"{unique_username}_1")
        user1.set_password("pass1")
        user2 = User(username=f"{unique_username}_2")
        user2.set_password("pass2")
        
        prod_user_repo.save(user1)
        prod_user_repo.save(user2)
        
        # Find all
        all_users = prod_user_repo.find_all()
        
        assert len(all_users) >= 3, "Should have at least rootusr + 2 test users"
        usernames = {u.username for u in all_users}
        assert "rootusr" in usernames
        assert f"{unique_username}_1" in usernames
        assert f"{unique_username}_2" in usernames
    
    def test_find_by_api_key(self, prod_user_repo, unique_username):
        """Test finding user by API key."""
        # Create user with API key
        user = User(username=unique_username)
        user.set_password("pass")
        api_key = user.generate_api_key()
        saved_user = prod_user_repo.save(user)
        
        # Find by API key
        found_user = prod_user_repo.find_by_api_key(api_key)
        
        assert found_user is not None
        assert found_user.id == saved_user.id
        assert found_user.api_key == api_key
    
    def test_find_by_api_key_nonexistent(self, prod_user_repo):
        """Test finding user by non-existent API key."""
        found_user = prod_user_repo.find_by_api_key("sk-nonexistent-key")
        assert found_user is None


class TestUserRepositoryLastLogin:
    """Test last login functionality."""
    
    def test_update_last_login(self, prod_user_repo, unique_username):
        """Test updating last login timestamp."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        assert saved_user.last_login is None, "New user should not have last_login"
        
        # Update last login
        before_update = datetime.now(timezone.utc)
        success = prod_user_repo.update_last_login(saved_user.id)
        after_update = datetime.now(timezone.utc)
        
        assert success is True
        
        # Verify last_login was set
        updated_user = prod_user_repo.find_by_id(saved_user.id)
        assert updated_user.last_login is not None
        assert before_update <= updated_user.last_login <= after_update
    
    def test_update_last_login_nonexistent_user(self, prod_user_repo):
        """Test updating last login for non-existent user."""
        success = prod_user_repo.update_last_login(999999)
        assert success is False


class TestUserRepositoryAPIKey:
    """Test API key management."""
    
    def test_update_api_key(self, prod_user_repo, unique_username):
        """Test updating user's API key."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Update API key (use unique key to avoid conflicts)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        new_api_key = f"sk-test-api-key-{timestamp}"
        success = prod_user_repo.update_api_key(saved_user.id, new_api_key)
        
        assert success is True
        
        # Verify API key was updated
        updated_user = prod_user_repo.find_by_id(saved_user.id)
        assert updated_user.api_key == new_api_key
        
        # Verify can find by API key
        found_user = prod_user_repo.find_by_api_key(new_api_key)
        assert found_user.id == saved_user.id
    
    def test_remove_api_key(self, prod_user_repo, unique_username):
        """Test removing user's API key."""
        # Create user with API key
        user = User(username=unique_username)
        user.set_password("pass")
        api_key = user.generate_api_key()
        saved_user = prod_user_repo.save(user)
        
        # Remove API key
        success = prod_user_repo.update_api_key(saved_user.id, None)
        
        assert success is True
        
        # Verify API key was removed
        updated_user = prod_user_repo.find_by_id(saved_user.id)
        assert updated_user.api_key is None
        
        # Verify cannot find by old API key
        found_user = prod_user_repo.find_by_api_key(api_key)
        assert found_user is None
    
    def test_update_api_key_nonexistent_user(self, prod_user_repo):
        """Test updating API key for non-existent user."""
        success = prod_user_repo.update_api_key(999999, "sk-test-key")
        assert success is False


class TestUserRepositorySessions:
    """Test session management."""
    
    def test_update_user_sessions(self, prod_user_repo, unique_username):
        """Test updating user's active sessions."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Update sessions
        session_ids = [1001, 1002, 1003]
        success = prod_user_repo.update_user_sessions(saved_user.id, session_ids)
        
        assert success is True
        
        # Verify sessions were updated
        sessions = prod_user_repo.get_user_sessions(saved_user.id)
        assert sessions == session_ids
    
    def test_get_user_sessions(self, prod_user_repo, unique_username):
        """Test getting user's active sessions."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Initially should have no sessions
        sessions = prod_user_repo.get_user_sessions(saved_user.id)
        assert sessions == []
        
        # Add sessions
        session_ids = [2001, 2002]
        prod_user_repo.update_user_sessions(saved_user.id, session_ids)
        
        # Get sessions
        sessions = prod_user_repo.get_user_sessions(saved_user.id)
        assert sessions == session_ids
    
    def test_get_user_sessions_nonexistent_user(self, prod_user_repo):
        """Test getting sessions for non-existent user."""
        sessions = prod_user_repo.get_user_sessions(999999)
        assert sessions == []
    
    def test_cleanup_inactive_sessions(self, prod_user_repo, unique_username):
        """Test cleaning up inactive sessions."""
        # Create user with sessions
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Add sessions
        all_sessions = [3001, 3002, 3003, 3004]
        prod_user_repo.update_user_sessions(saved_user.id, all_sessions)
        
        # Keep only some sessions active
        active_sessions = [3002, 3004]
        success = prod_user_repo.cleanup_inactive_sessions(saved_user.id, active_sessions)
        
        assert success is True
        
        # Verify only active sessions remain
        remaining_sessions = prod_user_repo.get_user_sessions(saved_user.id)
        assert set(remaining_sessions) == set(active_sessions)
    
    def test_cleanup_inactive_sessions_nonexistent_user(self, prod_user_repo):
        """Test cleanup for non-existent user."""
        success = prod_user_repo.cleanup_inactive_sessions(999999, [1, 2, 3])
        assert success is False


class TestUserRepositoryPreferences:
    """Test user preferences management."""
    
    def test_update_user_preferences(self, prod_user_repo, unique_username):
        """Test updating user preferences."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Update preferences
        preferences = {
            "theme": "dark",
            "language": "en",
            "notifications": True
        }
        success = prod_user_repo.update_user_preferences(saved_user.id, preferences)
        
        assert success is True
        
        # Verify preferences were updated
        updated_user = prod_user_repo.find_by_id(saved_user.id)
        assert updated_user.preferences == preferences
        assert updated_user.get_preference("theme") == "dark"
        assert updated_user.get_preference("language") == "en"
        assert updated_user.get_preference("notifications") is True
    
    def test_update_user_preferences_merge(self, prod_user_repo, unique_username):
        """Test that preferences are merged, not replaced."""
        # Create user with initial preferences
        user = User(username=unique_username)
        user.set_password("pass")
        user.set_preference("initial_key", "initial_value")
        saved_user = prod_user_repo.save(user)
        
        # Update with new preferences
        new_preferences = {"new_key": "new_value"}
        prod_user_repo.update_user_preferences(saved_user.id, new_preferences)
        
        # Verify both old and new preferences exist
        updated_user = prod_user_repo.find_by_id(saved_user.id)
        assert updated_user.get_preference("initial_key") == "initial_value"
        assert updated_user.get_preference("new_key") == "new_value"
    
    def test_update_user_preferences_nonexistent_user(self, prod_user_repo):
        """Test updating preferences for non-existent user."""
        success = prod_user_repo.update_user_preferences(999999, {"key": "value"})
        assert success is False


class TestUserRepositoryDelete:
    """Test user deletion."""
    
    def test_delete_user(self, prod_user_repo, unique_username):
        """Test deleting a user."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        user_id = saved_user.id
        
        # Verify user exists
        assert prod_user_repo.find_by_id(user_id) is not None
        
        # Delete user
        success = prod_user_repo.delete(user_id)
        
        assert success is True
        
        # Verify user is deleted
        assert prod_user_repo.find_by_id(user_id) is None
        assert prod_user_repo.find_by_username(unique_username) is None
    
    def test_delete_user_with_sessions(self, prod_user_repo, unique_username):
        """Test deleting a user with active sessions."""
        # Create user with sessions
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        user_id = saved_user.id
        
        # Add sessions
        prod_user_repo.update_user_sessions(user_id, [4001, 4002])
        
        # Delete user
        success = prod_user_repo.delete(user_id)
        
        assert success is True
        
        # Verify user and sessions are deleted
        assert prod_user_repo.find_by_id(user_id) is None
        assert prod_user_repo.get_user_sessions(user_id) == []
    
    def test_delete_nonexistent_user(self, prod_user_repo):
        """Test deleting non-existent user."""
        success = prod_user_repo.delete(999999)
        assert success is False


class TestUserRepositoryDataIntegrity:
    """Test data integrity and datetime synchronization."""
    
    def test_created_at_synchronization(self, prod_user_repo, unique_username):
        """Test that created_at is synchronized between column and data field."""
        # Create user
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Verify created_at is set
        assert saved_user.created_at is not None
        
        # Query database directly
        with get_db_connection(TEST_DB_URL) as db_session:
            db_user = db_session.query(UserModel).filter_by(id=saved_user.id).first()
            
            # Verify column has value
            assert db_user.created_at is not None
            
            # Verify data field has value
            assert db_user.data.get('created_at') is not None
            
            # Verify they match (allowing for ISO format conversion)
            column_iso = db_user.created_at.isoformat()
            data_iso = db_user.data.get('created_at')
            assert column_iso == data_iso, \
                f"created_at mismatch: column={column_iso}, data={data_iso}"
    
    def test_password_hash_not_in_to_dict(self, prod_user_repo, unique_username):
        """Test that password_hash is not exposed in to_dict()."""
        # Create user
        user = User(username=unique_username)
        user.set_password("SecretPassword123")
        saved_user = prod_user_repo.save(user)
        
        # Get user dict
        user_dict = saved_user.to_dict()
        
        # Verify password_hash is not in dict
        assert 'password_hash' not in user_dict, \
            "password_hash should not be exposed in to_dict()"
        
        # But verify it's stored in database
        with get_db_connection(TEST_DB_URL) as db_session:
            db_user = db_session.query(UserModel).filter_by(id=saved_user.id).first()
            assert db_user.password_hash is not None
            assert len(db_user.password_hash) > 0
    
    def test_session_limit_default(self, prod_user_repo, unique_username):
        """Test that session_limit has correct default value."""
        # Create user without specifying session_limit
        user = User(username=unique_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        # Verify default session_limit
        assert saved_user.session_limit == 5, "Default session_limit should be 5"
        
        # Verify in database
        found_user = prod_user_repo.find_by_id(saved_user.id)
        assert found_user.session_limit == 5


class TestUserRepositoryEdgeCases:
    """Test edge cases and error handling."""
    
    def test_save_user_with_special_characters(self, prod_user_repo):
        """Test saving user with special characters in username."""
        special_username = f"user_特殊字符_{datetime.now().timestamp()}"
        
        user = User(username=special_username)
        user.set_password("pass")
        saved_user = prod_user_repo.save(user)
        
        assert saved_user.id is not None
        assert saved_user.username == special_username
        
        # Verify can find by username
        found_user = prod_user_repo.find_by_username(special_username)
        assert found_user is not None
        assert found_user.id == saved_user.id
    
    def test_save_user_with_long_password(self, prod_user_repo, unique_username):
        """Test saving user with very long password."""
        long_password = "A" * 1000
        
        user = User(username=unique_username)
        user.set_password(long_password)
        saved_user = prod_user_repo.save(user)
        
        assert saved_user.verify_password(long_password)
    
    def test_save_user_with_empty_preferences(self, prod_user_repo, unique_username):
        """Test saving user with empty preferences."""
        user = User(username=unique_username)
        user.set_password("pass")
        user.preferences = {}
        saved_user = prod_user_repo.save(user)
        
        assert saved_user.preferences == {}
        
        found_user = prod_user_repo.find_by_id(saved_user.id)
        assert found_user.preferences == {}
    
    def test_save_user_with_complex_metadata(self, prod_user_repo, unique_username):
        """Test saving user with complex nested metadata."""
        user = User(username=unique_username)
        user.set_password("pass")
        user.metadata = {
            "nested": {
                "level1": {
                    "level2": "value"
                }
            },
            "list": [1, 2, 3],
            "mixed": {"a": [1, 2], "b": {"c": "d"}}
        }
        saved_user = prod_user_repo.save(user)
        
        found_user = prod_user_repo.find_by_id(saved_user.id)
        assert found_user.metadata == user.metadata
