"""Tests for agentgit.database.repositories.user_repository."""

import pytest

from agentgit.auth.user import User
from agentgit.database.repositories.user_repository import UserRepository


@pytest.fixture
def user_repo(tmp_path):
    """Create a UserRepository backed by a temporary SQLite database."""
    db_file = tmp_path / "user_repository.db"
    return UserRepository(db_path=str(db_file))


def test_user_repository_full_flow(user_repo):
    # root user should be created automatically
    root = user_repo.find_by_username("rootusr")
    assert root is not None
    assert root.is_admin is True

    # create a new user and persist it
    user = User(username="alice", is_admin=False, session_limit=3)
    user.set_password("secret")
    saved_user = user_repo.save(user)

    assert saved_user.id is not None
    assert user_repo.find_by_id(saved_user.id).username == "alice"
    assert user_repo.find_by_username("alice").id == saved_user.id

    # exercise update path of save
    saved_user.username = "alice_renamed"
    saved_user.is_admin = True
    saved_user.api_key = "API123"
    user_repo.save(saved_user)
    updated = user_repo.find_by_username("alice_renamed")
    assert updated is not None and updated.is_admin is True

    # find_all should include both root and the renamed user
    usernames = {u.username for u in user_repo.find_all()}
    assert {"rootusr", "alice_renamed"}.issubset(usernames)

    # update_last_login and verify
    assert user_repo.update_last_login(saved_user.id) is True
    assert user_repo.find_by_id(saved_user.id).last_login is not None

    # update_api_key, find_by_api_key, and removal
    assert user_repo.update_api_key(saved_user.id, "API123-NEW") is True
    assert user_repo.find_by_api_key("API123-NEW").id == saved_user.id
    assert user_repo.update_api_key(saved_user.id, None) is True
    assert user_repo.find_by_api_key("API123-NEW") is None

    # user session helpers
    assert user_repo.update_user_sessions(saved_user.id, [100, 200]) is True
    assert user_repo.get_user_sessions(saved_user.id) == [100, 200]
    assert user_repo.cleanup_inactive_sessions(saved_user.id, [200]) is True
    assert user_repo.get_user_sessions(saved_user.id) == [200]

    # user preferences helpers
    assert user_repo.update_user_preferences(saved_user.id, {"theme": "dark"}) is True
    assert user_repo.find_by_id(saved_user.id).preferences["theme"] == "dark"

    # delete user and ensure cascading helpers behave as expected
    assert user_repo.delete(saved_user.id) is True
    assert user_repo.find_by_id(saved_user.id) is None
    assert user_repo.get_user_sessions(saved_user.id) == []
