"""Database configuration for the rollback agent system."""

import os


def get_database_path() -> str:
    """Get the path to the SQLite database.
    
    Returns:
        Path to the database file
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Return path to database file
    return os.path.join(data_dir, "rollback_agent.db")