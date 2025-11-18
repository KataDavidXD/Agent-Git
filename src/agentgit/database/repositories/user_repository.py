"""User repository for database operations.

Provides ORM functionality for User entities with SQLite backend.
"""

import sqlite3
import json
from typing import Optional, List
from datetime import datetime
from pathlib import Path

from agentgit.auth.user import User
from agentgit.database.db_config import (
    get_database_path,
    get_db_connection,
    is_postgres_backend,
)


class UserRepository:
    """Repository for User CRUD operations with SQLite.

    This class handles all database operations for User entities,
    including automatic initialization of the database schema and
    creation of the default admin user (rootusr).

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> repo = UserRepository()
        >>> user = User(username="alice")
        >>> user.set_password("secret")
        >>> saved_user = repo.save(user)
        >>> found_user = repo.find_by_username("alice")
        >>> found_user.verify_password("secret")
        True
    """
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the user repository.

        Args:
            db_path: Path to SQLite database file. If None, uses configured default.
        """
        self.db_path = db_path or get_database_path()
        # Cache backend and parameter style for this repository instance
        self._backend = "postgres" if is_postgres_backend() else "sqlite"
        self._param = "%s" if self._backend == "postgres" else "?"
        self._init_db()

    def _init_db(self):
        """Initialize database schema and create default admin user.

        Creates the users table if it doesn't exist and ensures
        the rootusr admin account is present with default password "1234".
        Also handles migration for new fields.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if self._backend == "postgres":
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0,
                        created_at TEXT,
                        last_login TEXT,
                        data TEXT,
                        api_key TEXT,
                        session_limit INTEGER DEFAULT 5
                    )
                    """
                )
                cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'users'
                    """
                )
                columns = [row[0] for row in cursor.fetchall()]
            else:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0,
                        created_at TEXT,
                        last_login TEXT,
                        data TEXT,
                        api_key TEXT,
                        session_limit INTEGER DEFAULT 5
                    )
                    """
                )
                cursor.execute("PRAGMA table_info(users)")
                columns = [column[1] for column in cursor.fetchall()]

            if "api_key" not in columns:
                cursor.execute("ALTER TABLE users ADD COLUMN api_key TEXT")

            if "session_limit" not in columns:
                cursor.execute(
                    "ALTER TABLE users ADD COLUMN session_limit INTEGER DEFAULT 5"
                )

            conn.commit()

            cursor.execute(
                """
                SELECT COUNT(*) FROM users WHERE username = 'rootusr'
                """
            )

            if cursor.fetchone()[0] == 0:
                root_user = User(
                    username="rootusr",
                    is_admin=True,
                    created_at=datetime.now(),
                )
                root_user.set_password("1234")
                self.save(root_user)

            conn.commit()

    def save(self, user: User) -> User:
        """Save or update a user in the database.

        Performs insert if user.id is None, otherwise updates existing record.
        Stores both structured fields and full JSON representation for flexibility.

        Args:
            user: User object to save.

        Returns:
            The saved User object with id populated if newly created.

        Note:
            Password hash is stored separately from the JSON data for security.
        """
        user_dict = user.to_dict()
        user_dict['password_hash'] = user.password_hash
        json_data = json.dumps(user_dict)

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if user.id is None:
                if self._backend == "postgres":
                    cursor.execute(
                        """
                        INSERT INTO users (username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            user.username,
                            user.password_hash,
                            1 if user.is_admin else 0,
                            user.created_at.isoformat() if user.created_at else None,
                            user.last_login.isoformat() if user.last_login else None,
                            json_data,
                            user.api_key,
                            user.session_limit,
                        ),
                    )
                    user.id = cursor.fetchone()[0]
                else:
                    cursor.execute(
                        """
                        INSERT INTO users (username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            user.username,
                            user.password_hash,
                            1 if user.is_admin else 0,
                            user.created_at.isoformat() if user.created_at else None,
                            user.last_login.isoformat() if user.last_login else None,
                            json_data,
                            user.api_key,
                            user.session_limit,
                        ),
                    )
                    user.id = cursor.lastrowid
            else:
                if self._backend == "postgres":
                    cursor.execute(
                        """
                        UPDATE users 
                        SET username = %s, password_hash = %s, is_admin = %s, 
                            created_at = %s, last_login = %s, data = %s, api_key = %s, session_limit = %s
                        WHERE id = %s
                        """,
                        (
                            user.username,
                            user.password_hash,
                            1 if user.is_admin else 0,
                            user.created_at.isoformat() if user.created_at else None,
                            user.last_login.isoformat() if user.last_login else None,
                            json_data,
                            user.api_key,
                            user.session_limit,
                            user.id,
                        ),
                    )
                else:
                    cursor.execute(
                        """
                        UPDATE users 
                        SET username = ?, password_hash = ?, is_admin = ?, 
                            created_at = ?, last_login = ?, data = ?, api_key = ?, session_limit = ?
                        WHERE id = ?
                        """,
                        (
                            user.username,
                            user.password_hash,
                            1 if user.is_admin else 0,
                            user.created_at.isoformat() if user.created_at else None,
                            user.last_login.isoformat() if user.last_login else None,
                            json_data,
                            user.api_key,
                            user.session_limit,
                            user.id,
                        ),
                    )

            conn.commit()

        return user

    def find_by_id(self, user_id: int) -> Optional[User]:
        """Find a user by their database ID.

        Args:
            user_id: The unique identifier of the user.

        Returns:
            User object if found, None otherwise.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if self._backend == "postgres":
                cursor.execute(
                    """
                    SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                    FROM users WHERE id = %s
                    """,
                    (user_id,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                    FROM users WHERE id = ?
                    """,
                    (user_id,),
                )

            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

        return None

    def find_by_username(self, username: str) -> Optional[User]:
        """Find a user by their username.

        Args:
            username: The username to search for.

        Returns:
            User object if found, None otherwise.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if self._backend == "postgres":
                cursor.execute(
                    """
                    SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                    FROM users WHERE username = %s
                    """,
                    (username,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                    FROM users WHERE username = ?
                    """,
                    (username,),
                )

            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

        return None

    def find_all(self) -> List[User]:
        """Retrieve all users from the database.

        Returns:
            List of all User objects in the database.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                FROM users
                """
            )
            rows = cursor.fetchall()
            return [self._row_to_user(row) for row in rows]

    def find_by_api_key(self, api_key: str) -> Optional[User]:
        """Find a user by their API key.

        Args:
            api_key: The API key to search for.

        Returns:
            User object if found, None otherwise.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if self._backend == "postgres":
                cursor.execute(
                    """
                    SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                    FROM users WHERE api_key = %s
                    """,
                    (api_key,),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, username, password_hash, is_admin, created_at, last_login, data, api_key, session_limit
                    FROM users WHERE api_key = ?
                    """,
                    (api_key,),
                )

            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)

        return None

    def update_last_login(self, user_id: int) -> bool:
        """Update the last login timestamp for a user.

        Args:
            user_id: The ID of the user to update.

        Returns:
            True if updated successfully, False otherwise.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            if self._backend == "postgres":
                cursor.execute(
                    """
                    UPDATE users SET last_login = %s WHERE id = %s
                    """,
                    (now, user_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE users SET last_login = ? WHERE id = ?
                    """,
                    (now, user_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def update_api_key(self, user_id: int, api_key: Optional[str]) -> bool:
        """Update or remove a user's API key.

        Args:
            user_id: The ID of the user to update.
            api_key: New API key or None to remove.

        Returns:
            True if updated successfully, False otherwise.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if self._backend == "postgres":
                cursor.execute(
                    """
                    UPDATE users SET api_key = %s WHERE id = %s
                    """,
                    (api_key, user_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE users SET api_key = ? WHERE id = ?
                    """,
                    (api_key, user_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def get_user_sessions(self, user_id: int) -> List[int]:
        """Get all active session IDs for a user.

        Args:
            user_id: The ID of the user.

        Returns:
            List of active session IDs.
        """
        user = self.find_by_id(user_id)
        if user:
            return user.active_sessions
        return []

    def update_user_sessions(self, user_id: int, session_ids: List[int]) -> bool:
        """Update the active sessions list for a user.

        Args:
            user_id: The ID of the user.
            session_ids: New list of active session IDs.

        Returns:
            True if updated successfully, False otherwise.
        """
        user = self.find_by_id(user_id)
        if user:
            user.active_sessions = session_ids
            self.save(user)
            return True
        return False

    def update_user_preferences(self, user_id: int, preferences: dict) -> bool:
        """Update user preferences.

        Args:
            user_id: The ID of the user.
            preferences: Dictionary of preferences to update.

        Returns:
            True if updated successfully, False otherwise.
        """
        user = self.find_by_id(user_id)
        if user:
            user.preferences.update(preferences)
            self.save(user)
            return True
        return False

    def cleanup_inactive_sessions(self, user_id: int, active_session_ids: List[int]) -> bool:
        """Clean up inactive sessions for a user.

        Removes session IDs that are no longer active from the user's active_sessions list.

        Args:
            user_id: The ID of the user.
            active_session_ids: List of session IDs that are still active.

        Returns:
            True if cleanup was performed, False otherwise.
        """
        user = self.find_by_id(user_id)
        if user:
            # Keep only sessions that are in the active list
            user.active_sessions = [sid for sid in user.active_sessions if sid in active_session_ids]
            self.save(user)
            return True
        return False

    def delete(self, user_id: int) -> bool:
        """Delete a user from the database.

        Args:
            user_id: The ID of the user to delete.

        Returns:
            True if a user was deleted, False if no user found.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if self._backend == "postgres":
                cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
            else:
                cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    def _row_to_user(self, row) -> User:
        """Convert a database row to a User object.

        Args:
            row: Tuple containing database fields (id, username, password_hash,
                 is_admin, created_at, last_login, json_data, api_key, session_limit).
            
        Returns:
            User object reconstructed from database data.
            
        Note:
            Prioritizes JSON data if available, falls back to individual fields.
        """
        user_id, username, password_hash, is_admin, created_at, last_login, json_data, api_key, session_limit = row
        
        if json_data:
            user_dict = json.loads(json_data)
            # Ensure api_key and session_limit from columns override JSON data
            # This handles migration cases where JSON might not have these fields
            if api_key is not None:
                user_dict["api_key"] = api_key
            if session_limit is not None:
                user_dict["session_limit"] = session_limit
        else:
            user_dict = {
                "id": user_id,
                "username": username,
                "is_admin": bool(is_admin),
                "created_at": created_at,
                "last_login": last_login,
                "api_key": api_key,
                "session_limit": session_limit or 5,
                "active_sessions": [],
                "preferences": {},
                "metadata": {}
            }
        
        user_dict["password_hash"] = password_hash
        
        user = User.from_dict(user_dict)
        user.id = user_id
        
        return user