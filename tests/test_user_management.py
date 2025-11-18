"""Test suite for user creation and management functionality.

Tests user registration, authentication, API key management, and preferences.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import tempfile

from agentgit.auth.auth_service import AuthService
from agentgit.auth.user import User
from agentgit.database.repositories.user_repository import UserRepository
from dotenv import load_dotenv
load_dotenv()


class TestUserManagement(unittest.TestCase):
    """Test cases for user creation and management."""
    
    def setUp(self):
        """Set up test environment with temporary database."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        
        # Initialize services with test database
        self.user_repo = UserRepository(db_path=self.db_path)
        self.auth_service = AuthService(user_repository=self.user_repo)
    
    def tearDown(self):
        """Clean up temporary database."""
        try:
            os.unlink(self.db_path)
        except:
            pass
    
    def test_user_registration(self):
        """Test user registration process."""
        # Test successful registration
        success, user, msg = self.auth_service.register(
            username="alice",
            password="SecurePass123!",
            confirm_password="SecurePass123!"
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "alice")
        self.assertIn("registered successfully", msg)
        
        # Test duplicate username
        success, user, msg = self.auth_service.register(
            username="alice",
            password="AnotherPass456!",
            confirm_password="AnotherPass456!"
        )
        
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertIn("already taken", msg)
        
        # Test password mismatch
        success, user, msg = self.auth_service.register(
            username="bob",
            password="Pass123!",
            confirm_password="Pass456!"
        )
        
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertIn("do not match", msg.lower())
    
    def test_user_login(self):
        """Test user login functionality."""
        # Register a user first
        self.auth_service.register(
            username="charlie",
            password="TestPass789!",
            confirm_password="TestPass789!"
        )
        
        # Test successful login
        success, user, msg = self.auth_service.login(
            username="charlie",
            password="TestPass789!"
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "charlie")
        self.assertIsNotNone(user.last_login)
        
        # Test invalid password
        success, user, msg = self.auth_service.login(
            username="charlie",
            password="WrongPassword"
        )
        
        self.assertFalse(success)
        self.assertIsNone(user)
        self.assertIn("Invalid username or password", msg)
        
        # Test non-existent user
        success, user, msg = self.auth_service.login(
            username="nonexistent",
            password="anypassword"
        )
        
        self.assertFalse(success)
        self.assertIsNone(user)
    
    def test_api_key_management(self):
        """Test API key generation and authentication."""
        # Register and get user
        success, user, _ = self.auth_service.register(
            username="david",
            password="ApiKeyTest123!"
        )
        self.assertTrue(success)
        
        # Generate API key
        success, api_key, msg = self.auth_service.generate_api_key(user.id)
        
        self.assertTrue(success)
        self.assertIsNotNone(api_key)
        self.assertTrue(api_key.startswith("sk-"))
        # token_urlsafe(32) generates ~43 chars, so total is ~46 chars
        self.assertGreaterEqual(len(api_key), 40)  # Should be at least 40 chars
        self.assertLessEqual(len(api_key), 50)  # Should be at most 50 chars
        
        # Test login with API key
        success, authed_user, msg = self.auth_service.login_with_api_key(api_key)
        
        self.assertTrue(success)
        self.assertIsNotNone(authed_user)
        self.assertEqual(authed_user.username, "david")
        
        # Revoke API key
        success, msg = self.auth_service.revoke_api_key(user.id)
        self.assertTrue(success)
        
        # Test login with revoked API key
        success, authed_user, msg = self.auth_service.login_with_api_key(api_key)
        self.assertFalse(success)
        self.assertIsNone(authed_user)
    
    def test_password_change(self):
        """Test password change functionality."""
        # Register a user
        success, user, _ = self.auth_service.register(
            username="emma",
            password="OldPass123!"
        )
        self.assertTrue(success)
        
        # Change password
        success, msg = self.auth_service.change_password(
            user_id=user.id,
            current_password="OldPass123!",
            new_password="NewPass456!"
        )
        
        self.assertTrue(success)
        self.assertIn("changed successfully", msg)
        
        # Test login with new password
        success, user, _ = self.auth_service.login(
            username="emma",
            password="NewPass456!"
        )
        self.assertTrue(success)
        
        # Test login with old password should fail
        success, user, _ = self.auth_service.login(
            username="emma",
            password="OldPass123!"
        )
        self.assertFalse(success)
        
        # Test change with wrong current password
        success, msg = self.auth_service.change_password(
            user_id=user.id if user else 1,
            current_password="WrongCurrent",
            new_password="AnotherPass789!"
        )
        self.assertFalse(success)
        self.assertIn("incorrect", msg)
    
    def test_user_preferences(self):
        """Test user preferences management."""
        # Register a user
        success, user, _ = self.auth_service.register(
            username="frank",
            password="PrefTest123!"
        )
        self.assertTrue(success)
        
        # Update preferences
        preferences = {
            "temperature": 0.8,
            "max_tokens": 1500,
            "auto_checkpoint": True,
            "system_prompt": "You are a helpful assistant"
        }
        
        success, msg = self.auth_service.update_user_preferences(
            user_id=user.id,
            preferences=preferences
        )
        
        self.assertTrue(success)
        
        # Retrieve and verify preferences
        retrieved_user = self.user_repo.find_by_id(user.id)
        self.assertIsNotNone(retrieved_user)
        self.assertEqual(retrieved_user.preferences["temperature"], 0.8)
        self.assertEqual(retrieved_user.preferences["max_tokens"], 1500)
        
        # Test invalid preferences
        invalid_prefs = {
            "temperature": 2.5,  # Out of range
            "max_tokens": -100   # Negative value
        }
        
        success, msg = self.auth_service.update_user_preferences(
            user_id=user.id,
            preferences=invalid_prefs
        )
        
        self.assertFalse(success)
    
    def test_session_management(self):
        """Test user session tracking."""
        # Register a user
        success, user, _ = self.auth_service.register(
            username="grace",
            password="SessionTest123!"
        )
        self.assertTrue(success)
        
        # Add sessions
        success, msg = self.auth_service.add_user_session(user.id, 101)
        self.assertTrue(success)
        
        success, msg = self.auth_service.add_user_session(user.id, 102)
        self.assertTrue(success)
        
        # Get sessions
        sessions = self.auth_service.get_user_sessions(user.id)
        self.assertEqual(len(sessions), 2)
        self.assertIn(101, sessions)
        self.assertIn(102, sessions)
        
        # Verify ownership
        owns_101 = self.auth_service.verify_session_ownership(user.id, 101)
        self.assertTrue(owns_101)
        
        owns_999 = self.auth_service.verify_session_ownership(user.id, 999)
        self.assertFalse(owns_999)
        
        # Remove session
        success, msg = self.auth_service.remove_user_session(user.id, 101)
        self.assertTrue(success)
        
        sessions = self.auth_service.get_user_sessions(user.id)
        self.assertEqual(len(sessions), 1)
        self.assertNotIn(101, sessions)
        
        # Test session limit (default is 10)
        for i in range(9):  # Already have 1, add 9 more to reach limit
            self.auth_service.add_user_session(user.id, 200 + i)
        
        # Try to exceed limit
        success, msg = self.auth_service.add_user_session(user.id, 300)
        self.assertFalse(success)
        self.assertIn("limit", msg)
    
    def test_admin_operations(self):
        """Test admin user operations."""
        # Create admin user
        admin = User(username="admin", is_admin=True)
        admin.set_password("AdminPass123!")
        admin = self.user_repo.save(admin)
        
        # Create regular user
        success, regular_user, _ = self.auth_service.register(
            username="regularuser",
            password="RegularPass123!"
        )
        self.assertTrue(success)
        
        # Admin deletes regular user
        success, msg = self.auth_service.delete_user(
            admin_user_id=admin.id,
            target_username="regularuser"
        )
        
        self.assertTrue(success)
        self.assertIn("deleted successfully", msg)
        
        # Verify user is deleted
        deleted_user = self.user_repo.find_by_username("regularuser")
        self.assertIsNone(deleted_user)
        
        # Regular user cannot delete others
        success, another_user, _ = self.auth_service.register(
            username="another",
            password="Pass123!"
        )
        success, target, _ = self.auth_service.register(
            username="target",
            password="Pass123!"
        )
        
        success, msg = self.auth_service.delete_user(
            admin_user_id=another_user.id,
            target_username="target"
        )
        
        self.assertFalse(success)
        self.assertIn("Admin permission required", msg)


if __name__ == "__main__":
    unittest.main()