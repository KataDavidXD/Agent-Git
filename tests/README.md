# Test Suite for LangGraph Rollback Agent System

## Overview

This test suite provides comprehensive testing for the LangGraph rollback agent system, including user management, checkpoint/rollback functionality, and session management.

## Prerequisites

Before running the tests, ensure you have:

1. **Environment Variables Set**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export BASE_URL="your-base-url"  # Optional, for custom OpenAI endpoints
   ```

2. **Dependencies Installed**:
   ```bash
   pip install pytest unittest
   ```

## Test Files

### 1. `test_user_management.py`
Tests user-related functionality:
- User registration and validation
- Authentication (username/password and API keys)
- Password management
- User preferences for agent configuration
- Session ownership tracking
- Admin operations

### 2. `test_rollback_functionality.py`
Tests the core rollback and checkpoint features using real OpenAI models:
- Manual and automatic checkpoint creation
- Memory preservation after rollback (agent forgets information after checkpoint)
- State preservation in checkpoints
- Multiple rollbacks creating separate branches
- Checkpoint deletion and cleanup
- Checkpoint search functionality

### 3. `test_session_management.py`
Tests the two-tier session architecture:
- External session creation and management
- Internal session creation and linking
- Multiple internal sessions per external session
- Branching from checkpoints
- Complete lifecycle with AgentService and OpenAI models
- Session statistics and metadata
- Cascade deletion
- Session ownership verification

## Running Tests

### Run All Tests
```bash
# Using pytest
pytest tests/

# Using unittest
python -m unittest discover tests/
```

### Run Individual Test Files
```bash
# User management tests
python -m unittest tests.test_user_management

# Rollback functionality tests
python -m unittest tests.test_rollback_functionality

# Session management tests
python -m unittest tests.test_session_management
```

### Run with Verbose Output
```bash
pytest tests/ -v

# Or with unittest
python -m unittest discover tests/ -v
```

### Run Specific Test Methods
```bash
# Run a specific test class
python -m unittest tests.test_rollback_functionality.TestRollbackFunctionality

# Run a specific test method
python -m unittest tests.test_rollback_functionality.TestRollbackFunctionality.test_rollback_memory_preservation
```

## Important Notes

1. **API Key Required**: The rollback and session tests now use real OpenAI models. Tests will skip if `OPENAI_API_KEY` is not set.

2. **Cost Considerations**: Since these tests use real OpenAI API calls, they will incur costs. The tests use `gpt-4o-mini` with low temperature (0.3) to minimize costs.

3. **Temporary Databases**: Each test creates its own temporary SQLite database that is cleaned up after the test completes.

4. **Test Isolation**: Tests are designed to be independent and can be run in any order.

5. **Network Requirements**: Tests require internet connection for OpenAI API calls.

## Troubleshooting

### Tests Skipping
If tests are being skipped, ensure:
- `OPENAI_API_KEY` environment variable is set
- API key has sufficient credits/quota

### Import Errors
Ensure you're running tests from the project root:
```bash
cd /path/to/rollback_langchain_agnorefactor
python -m pytest tests/
```

### Database Errors
If you encounter database-related errors:
- Check file permissions in the temp directory
- Ensure SQLite is properly installed
- Try clearing any leftover temp files

## Adding New Tests

When adding new tests:
1. Follow the existing pattern of using real OpenAI models
2. Use temporary databases for isolation
3. Clean up resources in `tearDown()`
4. Test both success and failure cases
5. Document what the test verifies