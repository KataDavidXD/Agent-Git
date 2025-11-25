"""Pytest coverage for session-related repositories using a temp SQLite database."""

from __future__ import annotations

import uuid
from typing import Dict

import pytest

from agentgit.database import db_config
from agentgit.database.db_config import get_db_connection
from agentgit.database.models import User as UserModel
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.sessions.external_session import ExternalSession
from agentgit.sessions.internal_session import InternalSession
from agentgit.checkpoints.checkpoint import Checkpoint


@pytest.fixture
def sqlite_repo_env(tmp_path, monkeypatch) -> Dict[str, str]:
    """Set up a dedicated SQLite database for each repository test."""
    db_file = tmp_path / "repo.sqlite"
    db_url = f"sqlite:///{db_file.as_posix()}"
    monkeypatch.setenv("DATABASE", "sqlite")
    monkeypatch.setenv("DATABASE_URL", db_url)

    # Reset cached engine/session factory so init_db uses this database
    db_config._engine = None
    db_config._SessionLocal = None
    db_config.init_db()

    yield {"db_path": str(db_file)}

    if db_config._engine is not None:
        db_config._engine.dispose()
    db_config._engine = None
    db_config._SessionLocal = None


def _create_user(username: str) -> int:
    """Insert a bare UserModel row and return its ID."""
    with get_db_connection() as session:
        user = UserModel(
            username=username,
            password_hash="hash",
            is_admin=False,
        )
        session.add(user)
        session.flush()
        return user.id


def test_external_session_repository_flow(sqlite_repo_env):
    user_id = _create_user("owner")
    repo = ExternalSessionRepository()

    # Create session and verify retrieval APIs
    session = ExternalSession(user_id=user_id, session_name="Project Chat")
    saved = repo.create(session)
    assert saved.id is not None
    assert repo.get_by_id(saved.id).session_name == "Project Chat"

    # Listing helpers
    assert len(repo.get_user_sessions(user_id)) == 1
    assert len(repo.get_user_sessions(user_id, active_only=True)) == 1
    assert repo.count_user_sessions(user_id) == 1
    assert repo.check_ownership(saved.id, user_id) is True

    # Internal session tracking helpers
    assert repo.add_internal_session(saved.id, "lang-123") is True
    assert repo.get_by_internal_session("lang-123").id == saved.id
    assert repo.set_current_internal_session(saved.id, "lang-123") is True

    # Update metadata/branch info
    saved.branch_count = 2
    saved.total_checkpoints = 5
    saved.metadata["topic"] = "design"
    assert repo.update(saved) is True

    # Deactivate then delete
    assert repo.deactivate(saved.id) is True
    assert repo.get_user_sessions(user_id, active_only=True) == []
    assert repo.delete(saved.id) is True
    assert repo.get_by_id(saved.id) is None
    assert repo.count_user_sessions(user_id) == 0


def test_internal_session_repository_branching(sqlite_repo_env):
    user_id = _create_user("owner-int")
    ext_repo = ExternalSessionRepository()
    external = ext_repo.create(ExternalSession(user_id=user_id, session_name="Main"))

    repo = InternalSessionRepository()

    root_session = InternalSession(
        external_session_id=external.id,
        langgraph_session_id="lg-root",
        session_state={"step": 1},
        conversation_history=[{"role": "user", "content": "hi"}],
    )
    root = repo.create(root_session)
    assert repo.get_by_id(root.id).langgraph_session_id == "lg-root"
    assert repo.get_current_session(external.id).id == root.id

    # Create a checkpoint to reference in the branch
    cp_repo = CheckpointRepository()
    checkpoint = cp_repo.create(Checkpoint(
        internal_session_id=root.id,
        checkpoint_name="Branch Point",
        user_id=user_id,
    ))

    # Branch session referencing the root and a real checkpoint
    branch = InternalSession(
        external_session_id=external.id,
        langgraph_session_id="lg-branch",
        parent_session_id=root.id,
        branch_point_checkpoint_id=checkpoint.id,
        is_current=False,
    )
    child = repo.create(branch)
    assert repo.get_by_langgraph_session_id("lg-branch").id == child.id
    assert repo.get_branch_sessions(root.id)[0].id == child.id

    lineage = repo.get_session_lineage(child.id)
    assert [s.id for s in lineage] == [root.id, child.id]

    # Update branch content and tool usage
    child.session_state["step"] = 2
    child.conversation_history.append({"role": "assistant", "content": "ack"})
    repo.update(child)
    repo.update_tool_count(child.id, increment=2)
    assert repo.get_by_id(child.id).tool_invocation_count == 2

    # Current-session helpers & counts
    assert repo.set_current_session(root.id) is True
    assert repo.get_current_session(external.id).id == root.id
    assert repo.count_sessions(external.id) == 2
    assert len(repo.get_by_external_session(external.id)) == 2

    # Deletion and cleanup
    assert repo.delete(child.id) is True
    assert repo.get_branch_sessions(root.id) == []
    assert repo.delete(root.id) is True
    assert repo.get_by_id(root.id) is None
    assert repo.count_sessions(external.id) == 0


def test_checkpoint_repository_end_to_end(sqlite_repo_env):
    user_id = _create_user("owner-cp")
    ext_repo = ExternalSessionRepository()
    external = ext_repo.create(ExternalSession(user_id=user_id, session_name="Rollback"))
    int_repo = InternalSessionRepository()
    internal = int_repo.create(
        InternalSession(
            external_session_id=external.id,
            langgraph_session_id="lg-cp",
        )
    )

    repo = CheckpointRepository()

    manual_cp = Checkpoint(
        internal_session_id=internal.id,
        checkpoint_name="Manual",
        session_state={"counter": 1},
        conversation_history=[{"role": "user", "content": "start"}],
        tool_invocations=[{"tool": "search"}],
        user_id=user_id,
    )
    saved_manual = repo.create(manual_cp)

    auto_one = Checkpoint(
        internal_session_id=internal.id,
        checkpoint_name="AutoOne",
        is_auto=True,
        user_id=user_id,
    )
    saved_auto_one = repo.create(auto_one)

    auto_two = Checkpoint(
        internal_session_id=internal.id,
        checkpoint_name="AutoTwo",
        is_auto=True,
        user_id=user_id,
    )
    saved_auto_two = repo.create(auto_two)

    # Basic fetch helpers
    assert repo.get_by_id(saved_manual.id).checkpoint_name == "Manual"
    assert len(repo.get_by_internal_session(internal.id)) == 3
    assert len(repo.get_by_internal_session(internal.id, auto_only=True)) == 2
    assert repo.get_latest_checkpoint(internal.id).id == saved_auto_two.id

    counts = repo.count_checkpoints(internal.id)
    assert counts == {"total": 3, "auto": 2, "manual": 1}

    # Metadata & search helpers
    assert repo.update_checkpoint_metadata(saved_manual.id, {"tool_track_position": 5}) is True
    assert repo.get_by_id(saved_manual.id).metadata["tool_track_position"] == 5
    assert repo.search_checkpoints(internal.id, "Manual")[0].id == saved_manual.id

    # User-based queries and tool-filtered list
    assert repo.get_by_user(user_id, limit=2)
    with_tools = repo.get_checkpoints_with_tools(internal.id)
    assert len(with_tools) == 1 and with_tools[0].id == saved_manual.id

    # Auto-checkpoint cleanup
    deleted = repo.delete_auto_checkpoints(internal.id, keep_latest=1)
    assert deleted == 1
    assert len(repo.get_by_internal_session(internal.id, auto_only=True)) == 1

    # Deletion
    assert repo.delete(saved_manual.id) is True
    assert repo.get_by_id(saved_manual.id) is None
    assert repo.delete(saved_auto_two.id) is True
