#!/usr/bin/env python3
"""CLI Chat System with Rollback Capabilities.

A complete example of using the Full Snapshot Agent Rollback System with:
- User authentication and management
- Session management with multiple conversations
- Checkpoint creation and rollback functionality
- Rich CLI interface with colored output
"""

import os
import sys
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path
import getpass
from enum import Enum

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentgit.auth.auth_service import AuthService
from agentgit.auth.user import User
from agentgit.agents.agent_service import AgentService
from agentgit.agents.rollback_agent import RollbackAgent
from agentgit.sessions.external_session import ExternalSession
from agentgit.database.repositories.external_session_repository import ExternalSessionRepository
from agentgit.database.repositories.internal_session_repository import InternalSessionRepository
from agentgit.database.repositories.checkpoint_repository import CheckpointRepository
from agentgit.database.db_config import get_database_path


class Color:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


class MenuChoice(Enum):
    """Menu choices for the application."""
    # Auth menu
    LOGIN = "1"
    REGISTER = "2"
    EXIT = "0"
    
    # Main menu
    NEW_CHAT = "1"
    RESUME_CHAT = "2"
    LIST_SESSIONS = "3"
    USER_SETTINGS = "4"
    LOGOUT = "5"
    
    # Chat menu
    SEND_MESSAGE = "1"
    CREATE_CHECKPOINT = "2"
    LIST_CHECKPOINTS = "3"
    ROLLBACK = "4"
    VIEW_HISTORY = "5"
    BRANCH_INFO = "6"
    BACK_TO_MAIN = "7"


class CLIChatApp:
    """Main CLI application for the chat system."""

    def __init__(self):
        """Initialize the CLI chat application."""
        # The repositories handle their own table initialization
        self.auth_service = AuthService()
        self.agent_service = AgentService()
        self.external_session_repo = ExternalSessionRepository()
        self.internal_session_repo = InternalSessionRepository()
        self.checkpoint_repo = CheckpointRepository()

        self.current_user: Optional[User] = None
        self.current_external_session: Optional[ExternalSession] = None
        self.current_agent: Optional[RollbackAgent] = None

        # Ensure database path exists
        self._ensure_database_path()
    
    def _ensure_database_path(self):
        """Ensure the database directory exists."""
        try:
            db_path = get_database_path()
            db_dir = os.path.dirname(db_path)
            os.makedirs(db_dir, exist_ok=True)
            print(f"{Color.DIM}Database location: {db_path}{Color.ENDC}")
        except Exception as e:
            print(f"{Color.RED}Failed to initialize database path: {e}{Color.ENDC}")
            sys.exit(1)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, text: str):
        """Print a formatted header."""
        print(f"\n{Color.HEADER}{Color.BOLD}{'='*60}{Color.ENDC}")
        print(f"{Color.HEADER}{Color.BOLD}{text.center(60)}{Color.ENDC}")
        print(f"{Color.HEADER}{Color.BOLD}{'='*60}{Color.ENDC}\n")
    
    def print_menu(self, title: str, options: List[Tuple[str, str]]):
        """Print a formatted menu."""
        print(f"{Color.CYAN}{Color.BOLD}{title}{Color.ENDC}")
        print(f"{Color.DIM}{'-'*40}{Color.ENDC}")
        for key, description in options:
            print(f"  {Color.BOLD}[{key}]{Color.ENDC} {description}")
        print()
    
    def get_input(self, prompt: str) -> str:
        """Get user input with colored prompt."""
        return input(f"{Color.GREEN}➜ {prompt}: {Color.ENDC}")
    
    def print_success(self, message: str):
        """Print a success message."""
        print(f"{Color.GREEN}✓ {message}{Color.ENDC}")
    
    def print_error(self, message: str):
        """Print an error message."""
        print(f"{Color.RED}✗ {message}{Color.ENDC}")
    
    def print_warning(self, message: str):
        """Print a warning message."""
        print(f"{Color.WARNING}⚠ {message}{Color.ENDC}")
    
    def print_info(self, message: str):
        """Print an info message."""
        print(f"{Color.BLUE}ℹ {message}{Color.ENDC}")
    
    def run(self):
        """Run the main application loop."""
        self.clear_screen()
        self.print_header("ROLLBACK AGENT CHAT SYSTEM")
        print(f"{Color.DIM}A LangGraph-powered chat with checkpoint and rollback capabilities{Color.ENDC}\n")
        
        while True:
            if not self.current_user:
                self.show_auth_menu()
            else:
                self.show_main_menu()
    
    def show_auth_menu(self):
        """Show the authentication menu."""
        self.print_menu("Authentication", [
            (MenuChoice.LOGIN.value, "Login"),
            (MenuChoice.REGISTER.value, "Register new user"),
            (MenuChoice.EXIT.value, "Exit")
        ])
        
        choice = self.get_input("Select option")
        
        if choice == MenuChoice.LOGIN.value:
            self.handle_login()
        elif choice == MenuChoice.REGISTER.value:
            self.handle_register()
        elif choice == MenuChoice.EXIT.value:
            print(f"\n{Color.CYAN}Goodbye!{Color.ENDC}\n")
            sys.exit(0)
        else:
            self.print_error("Invalid choice. Please try again.")
    
    def handle_login(self):
        """Handle user login."""
        print(f"\n{Color.BOLD}User Login{Color.ENDC}")
        username = self.get_input("Username")
        password = getpass.getpass(f"{Color.GREEN}➜ Password: {Color.ENDC}")
        
        success, user, message = self.auth_service.login(username, password)
        
        if success:
            self.current_user = user
            self.print_success(message)
            self.print_info(f"Welcome back, {user.username}!")
        else:
            self.print_error(message)
    
    def handle_register(self):
        """Handle user registration."""
        print(f"\n{Color.BOLD}New User Registration{Color.ENDC}")
        username = self.get_input("Choose username")
        password = getpass.getpass(f"{Color.GREEN}➜ Choose password: {Color.ENDC}")
        confirm_password = getpass.getpass(f"{Color.GREEN}➜ Confirm password: {Color.ENDC}")
        
        success, user, message = self.auth_service.register(
            username, password, confirm_password
        )
        
        if success:
            self.current_user = user
            self.print_success(message)
            self.print_info(f"Welcome to the system, {user.username}!")
        else:
            self.print_error(message)
    
    def show_main_menu(self):
        """Show the main menu after login."""
        print(f"\n{Color.BOLD}Welcome, {self.current_user.username}!{Color.ENDC}")
        self.print_menu("Main Menu", [
            (MenuChoice.NEW_CHAT.value, "Start new chat session"),
            (MenuChoice.RESUME_CHAT.value, "Resume existing chat"),
            (MenuChoice.LIST_SESSIONS.value, "List all sessions"),
            (MenuChoice.USER_SETTINGS.value, "User settings"),
            (MenuChoice.LOGOUT.value, "Logout")
        ])
        
        choice = self.get_input("Select option")
        
        if choice == MenuChoice.NEW_CHAT.value:
            self.start_new_chat()
        elif choice == MenuChoice.RESUME_CHAT.value:
            self.resume_chat()
        elif choice == MenuChoice.LIST_SESSIONS.value:
            self.list_sessions()
        elif choice == MenuChoice.USER_SETTINGS.value:
            self.show_user_settings()
        elif choice == MenuChoice.LOGOUT.value:
            self.handle_logout()
        else:
            self.print_error("Invalid choice. Please try again.")
    
    def start_new_chat(self):
        """Start a new chat session."""
        session_name = self.get_input("Enter session name (or press Enter for default)")
        if not session_name:
            session_name = f"Session {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create external session
        external_session = ExternalSession(
            user_id=self.current_user.id,
            session_name=session_name,
            created_at=datetime.now()
        )
        
        saved_session = self.external_session_repo.create(external_session)
        
        if saved_session:
            self.current_external_session = saved_session
            
            # Create agent
            self.current_agent = self.agent_service.create_new_agent(
                external_session_id=saved_session.id,
                session_name=session_name
            )
            
            self.print_success(f"Created new chat session: {session_name}")
            
            # Add session to user's active sessions
            self.auth_service.add_user_session(self.current_user.id, saved_session.id)
            
            # Enter chat interface
            self.show_chat_interface()
        else:
            self.print_error("Failed to create chat session")
    
    def resume_chat(self):
        """Resume an existing chat session."""
        sessions = self.external_session_repo.get_user_sessions(self.current_user.id)
        
        if not sessions:
            self.print_warning("No existing sessions found")
            return
        
        print(f"\n{Color.BOLD}Available Sessions:{Color.ENDC}")
        for i, session in enumerate(sessions, 1):
            status = f"{Color.GREEN}[ACTIVE]{Color.ENDC}" if session.is_active else f"{Color.DIM}[INACTIVE]{Color.ENDC}"
            print(f"  {Color.BOLD}[{i}]{Color.ENDC} {session.session_name} {status}")
            print(f"      Created: {session.created_at}")
            print(f"      Branches: {session.branch_count}, Checkpoints: {session.total_checkpoints}")
        
        choice = self.get_input("Select session number (0 to cancel)")
        
        try:
            choice_idx = int(choice) - 1
            if choice_idx == -1:
                return
            
            if 0 <= choice_idx < len(sessions):
                selected_session = sessions[choice_idx]
                self.current_external_session = selected_session
                
                # Resume agent
                self.current_agent = self.agent_service.resume_agent(
                    external_session_id=selected_session.id
                )
                
                if self.current_agent:
                    self.print_success(f"Resumed session: {selected_session.session_name}")
                    self.show_chat_interface()
                else:
                    self.print_error("Failed to resume session")
            else:
                self.print_error("Invalid session number")
        except ValueError:
            self.print_error("Please enter a valid number")
    
    def show_chat_interface(self):
        """Show the chat interface for the current session."""
        if not self.current_agent or not self.current_external_session:
            self.print_error("No active session")
            return
        
        while True:
            print(f"\n{Color.CYAN}{'='*60}{Color.ENDC}")
            print(f"{Color.BOLD}Chat Session: {self.current_external_session.session_name}{Color.ENDC}")
            
            # Show current branch info
            if self.current_agent.internal_session:
                if self.current_agent.internal_session.parent_session_id:
                    checkpoint_id = self.current_agent.internal_session.branch_point_checkpoint_id
                    print(f"{Color.DIM}Branch from checkpoint {checkpoint_id}{Color.ENDC}")
            
            self.print_menu("Chat Options", [
                (MenuChoice.SEND_MESSAGE.value, "Send message"),
                (MenuChoice.CREATE_CHECKPOINT.value, "Create checkpoint"),
                (MenuChoice.LIST_CHECKPOINTS.value, "List checkpoints"),
                (MenuChoice.ROLLBACK.value, "Rollback to checkpoint"),
                (MenuChoice.VIEW_HISTORY.value, "View conversation history"),
                (MenuChoice.BRANCH_INFO.value, "View branch information"),
                (MenuChoice.BACK_TO_MAIN.value, "Back to main menu")
            ])
            
            choice = self.get_input("Select option")
            
            if choice == MenuChoice.SEND_MESSAGE.value:
                self.send_message()
            elif choice == MenuChoice.CREATE_CHECKPOINT.value:
                self.create_checkpoint()
            elif choice == MenuChoice.LIST_CHECKPOINTS.value:
                self.list_checkpoints()
            elif choice == MenuChoice.ROLLBACK.value:
                self.handle_rollback()
            elif choice == MenuChoice.VIEW_HISTORY.value:
                self.view_history()
            elif choice == MenuChoice.BRANCH_INFO.value:
                self.view_branch_info()
            elif choice == MenuChoice.BACK_TO_MAIN.value:
                break
            else:
                self.print_error("Invalid choice. Please try again.")
    
    def send_message(self):
        """Send a message to the agent."""
        message = self.get_input("Your message")
        
        if not message:
            return
        
        print(f"\n{Color.DIM}Processing...{Color.ENDC}")
        
        try:
            # Send message to agent using the run method
            response = self.current_agent.run(message)
            
            # Display response
            print(f"\n{Color.BOLD}Agent:{Color.ENDC}")
            if response and 'messages' in response and response['messages']:
                # Get the last AI message
                for msg in reversed(response['messages']):
                    if hasattr(msg, '__class__') and 'AI' in msg.__class__.__name__:
                        print(f"{msg.content}")
                        break
            elif response:
                # Fallback if response format is different
                print(f"{response}")
            
            # Check for rollback request
            if self.agent_service.handle_agent_response(self.current_agent, response):
                checkpoint_id = self.current_agent.internal_session.session_state.get('rollback_checkpoint_id') if self.current_agent.internal_session else None
                if checkpoint_id:
                    self.print_info(f"Agent requested rollback to checkpoint {checkpoint_id}")
                    self.perform_rollback(checkpoint_id)
            
        except Exception as e:
            self.print_error(f"Error processing message: {e}")
    
    def create_checkpoint(self):
        """Create a checkpoint at the current state."""
        name = self.get_input("Checkpoint name (or press Enter for auto-name)")
        description = self.get_input("Checkpoint description (optional)")
        
        if not name:
            name = f"Checkpoint {datetime.now().strftime('%H:%M:%S')}"
        
        try:
            # Use the create_checkpoint_tool method
            result = self.current_agent.create_checkpoint_tool(name=name)
            
            if "successfully" in result.lower():
                self.print_success(result)
                
                # If description provided, we can store it in checkpoint metadata
                # This would need additional implementation in the checkpoint creation
                if description:
                    self.print_info(f"Note: {description}")
            else:
                self.print_error(result)
        except Exception as e:
            self.print_error(f"Error creating checkpoint: {e}")
    
    def list_checkpoints(self):
        """List all checkpoints for the current session."""
        if not self.current_agent.internal_session:
            self.print_warning("No internal session active")
            return
        
        checkpoints = self.checkpoint_repo.get_by_internal_session(
            self.current_agent.internal_session.id
        )
        
        if not checkpoints:
            self.print_info("No checkpoints found")
            return
        
        print(f"\n{Color.BOLD}Checkpoints:{Color.ENDC}")
        for cp in checkpoints:
            name = cp.checkpoint_name or f"Checkpoint {cp.id}"
            print(f"\n  {Color.BOLD}[{cp.id}]{Color.ENDC} {name}")
            print(f"      Created: {cp.created_at}")
            if cp.metadata and cp.metadata.get('description'):
                print(f"      Description: {cp.metadata.get('description')}")
            print(f"      Turn: {cp.metadata.get('turn_number', 'N/A') if cp.metadata else 'N/A'}")
            print(f"      Messages: {cp.metadata.get('message_count', 0) if cp.metadata else 0}")
            if cp.is_auto:
                print(f"      Type: Auto-checkpoint")
    
    def handle_rollback(self):
        """Handle rollback to a checkpoint."""
        checkpoints = self.checkpoint_repo.get_by_internal_session(
            self.current_agent.internal_session.id
        )
        
        if not checkpoints:
            self.print_warning("No checkpoints available for rollback")
            return
        
        # List checkpoints
        self.list_checkpoints()
        
        checkpoint_id = self.get_input("Enter checkpoint ID to rollback to (0 to cancel)")
        
        try:
            checkpoint_id = int(checkpoint_id)
            if checkpoint_id == 0:
                return
            
            # Confirm rollback
            confirm = self.get_input(f"Rollback to checkpoint {checkpoint_id}? This will create a new branch. (y/n)")
            
            if confirm.lower() == 'y':
                self.perform_rollback(checkpoint_id)
        except ValueError:
            self.print_error("Please enter a valid checkpoint ID")
    
    def perform_rollback(self, checkpoint_id: int):
        """Perform the actual rollback operation."""
        print(f"\n{Color.DIM}Rolling back to checkpoint {checkpoint_id}...{Color.ENDC}")
        
        # Perform rollback
        new_agent = self.agent_service.rollback_to_checkpoint(
            external_session_id=self.current_external_session.id,
            checkpoint_id=checkpoint_id,
            rollback_tools=True
        )
        
        if new_agent:
            self.current_agent = new_agent
            self.print_success(f"Successfully rolled back to checkpoint {checkpoint_id}")
            self.print_info("You are now on a new branch. The original timeline is preserved.")
            
            # Clear the rollback request from state
            if self.current_agent.internal_session and 'rollback_checkpoint_id' in self.current_agent.internal_session.session_state:
                del self.current_agent.internal_session.session_state['rollback_checkpoint_id']
                self.current_agent._save_internal_session()
        else:
            self.print_error("Failed to rollback")
    
    def view_history(self):
        """View conversation history."""
        history = self.current_agent.get_conversation_history()
        
        if not history:
            self.print_info("No conversation history yet")
            return
        
        print(f"\n{Color.BOLD}Conversation History:{Color.ENDC}")
        print(f"{Color.DIM}(Showing last 20 messages){Color.ENDC}\n")
        
        for msg in history[-20:]:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            # Format based on role
            if role == 'user':
                print(f"{Color.GREEN}You:{Color.ENDC} {content}")
            elif role == 'assistant':
                print(f"{Color.BLUE}Agent:{Color.ENDC} {content}")
            else:
                print(f"{Color.DIM}{role}:{Color.ENDC} {content}")
            
            if timestamp:
                print(f"{Color.DIM}  [{timestamp}]{Color.ENDC}")
            print()
    
    def view_branch_info(self):
        """View branch/timeline information."""
        tree = self.agent_service.get_branch_tree(self.current_external_session.id)
        
        print(f"\n{Color.BOLD}Session Branch Tree:{Color.ENDC}")
        
        def print_tree(node_dict, indent=0):
            for _, info in node_dict.items():
                prefix = "  " * indent
                current_marker = " ← (current)" if info['is_current'] else ""
                branch_marker = " [BRANCH]" if info['is_branch'] else ""
                
                print(f"{prefix}• Session {info['session_id']}{branch_marker}{current_marker}")
                print(f"{prefix}  Checkpoints: {info['checkpoint_count']}, Tools: {info['tool_invocations']}")
                
                if info['children']:
                    for child in info['children']:
                        print_tree({child['id']: child}, indent + 1)
        
        print_tree(tree)
    
    def show_user_settings(self):
        """Show user settings menu."""
        print(f"\n{Color.BOLD}User Settings{Color.ENDC}")
        print(f"Username: {self.current_user.username}")
        print(f"User ID: {self.current_user.id}")
        print(f"Created: {self.current_user.created_at}")
        print(f"Last login: {self.current_user.last_login}")
        
        if self.current_user.api_key:
            print(f"API Key: {self.current_user.api_key[:8]}...")
        
        self.print_menu("Settings Options", [
            ("1", "Change password"),
            ("2", "Generate API key"),
            ("3", "Update preferences"),
            ("0", "Back")
        ])
        
        choice = self.get_input("Select option")
        
        if choice == "1":
            self.change_password()
        elif choice == "2":
            self.generate_api_key()
        elif choice == "3":
            self.update_preferences()
    
    def change_password(self):
        """Change user password."""
        current_password = getpass.getpass(f"{Color.GREEN}➜ Current password: {Color.ENDC}")
        new_password = getpass.getpass(f"{Color.GREEN}➜ New password: {Color.ENDC}")
        confirm_password = getpass.getpass(f"{Color.GREEN}➜ Confirm new password: {Color.ENDC}")
        
        if new_password != confirm_password:
            self.print_error("Passwords do not match")
            return
        
        success, message = self.auth_service.change_password(
            self.current_user.id, current_password, new_password
        )
        
        if success:
            self.print_success(message)
        else:
            self.print_error(message)
    
    def generate_api_key(self):
        """Generate a new API key."""
        confirm = self.get_input("Generate new API key? This will invalidate the old one. (y/n)")
        
        if confirm.lower() == 'y':
            success, api_key, message = self.auth_service.generate_api_key(self.current_user.id)
            
            if success:
                self.print_success(message)
                print(f"\n{Color.BOLD}Your new API key:{Color.ENDC}")
                print(f"{Color.WARNING}{api_key}{Color.ENDC}")
                print(f"{Color.DIM}Save this key securely. It won't be shown again.{Color.ENDC}")
            else:
                self.print_error(message)
    
    def update_preferences(self):
        """Update user preferences."""
        print(f"\n{Color.BOLD}Update Preferences{Color.ENDC}")
        print("Enter new values or press Enter to keep current:")
        
        preferences = {}
        
        # Model preferences
        model_id = self.get_input("Model ID (e.g., gpt-4o-mini)")
        if model_id:
            preferences['model_id'] = model_id
        
        temperature = self.get_input("Temperature (0.0-1.0)")
        if temperature:
            try:
                preferences['temperature'] = float(temperature)
            except ValueError:
                self.print_error("Invalid temperature value")
                return
        
        # Session preferences
        auto_checkpoint = self.get_input("Auto-checkpoint after tools? (y/n)")
        if auto_checkpoint:
            preferences['auto_checkpoint'] = auto_checkpoint.lower() == 'y'
        
        if preferences:
            success, message = self.auth_service.update_user_preferences(
                self.current_user.id, preferences
            )
            
            if success:
                self.print_success(message)
                # Update current user object
                self.current_user.preferences.update(preferences)
            else:
                self.print_error(message)
        else:
            self.print_info("No changes made")
    
    def list_sessions(self):
        """List all user sessions."""
        sessions = self.external_session_repo.get_user_sessions(self.current_user.id)
        
        if not sessions:
            self.print_info("No sessions found")
            return
        
        print(f"\n{Color.BOLD}Your Sessions:{Color.ENDC}")
        for session in sessions:
            status = f"{Color.GREEN}[ACTIVE]{Color.ENDC}" if session.is_active else f"{Color.DIM}[INACTIVE]{Color.ENDC}"
            print(f"\n• {Color.BOLD}{session.session_name}{Color.ENDC} {status}")
            print(f"  ID: {session.id}")
            print(f"  Created: {session.created_at}")
            print(f"  Updated: {session.updated_at}")
            print(f"  Branches: {session.branch_count}")
            print(f"  Total Checkpoints: {session.total_checkpoints}")
    
    def handle_logout(self):
        """Handle user logout."""
        self.current_user = None
        self.current_external_session = None
        self.current_agent = None
        self.print_success("Logged out successfully")
        self.clear_screen()
        self.print_header("ROLLBACK AGENT CHAT SYSTEM")


def main():
    """Main entry point for the CLI application."""
    try:
        app = CLIChatApp()
        app.run()
    except KeyboardInterrupt:
        print(f"\n\n{Color.CYAN}Goodbye!{Color.ENDC}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Color.RED}Fatal error: {e}{Color.ENDC}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()