# Testing Framework for RAG System

This directory contains comprehensive tests for the RAG system, covering API endpoints, unit components, and integration scenarios.

## Structure

```
backend/tests/
├── conftest.py           # Shared fixtures and test configuration
├── test_api.py          # FastAPI endpoint tests
├── test_infrastructure.py # Basic testing infrastructure validation
└── README.md            # This file
```

## Running Tests

```bash
# Run all tests
uv run python -m pytest

# Run with verbose output
uv run python -m pytest -v

# Run specific test file
uv run python -m pytest tests/test_api.py

# Run with coverage (if coverage is installed)
uv run python -m pytest --cov=backend

# Run tests by markers
uv run python -m pytest -m api      # API tests only
uv run python -m pytest -m unit     # Unit tests only
```

## Test Configuration

Tests are configured in `pyproject.toml` with the following settings:

- **Test discovery**: `backend/tests` directory
- **Markers**: `unit`, `integration`, `api` for test categorization
- **Dependencies**: pytest, pytest-asyncio, httpx for FastAPI testing

## Fixtures

### Core Fixtures (conftest.py)

- **mock_config**: Mock configuration object with test settings
- **temp_directory**: Temporary directory for test files
- **sample_course**: Sample course data for testing
- **sample_course_chunks**: Sample course chunks for vector storage tests
- **mock_rag_system**: Mock RAG system with controlled responses
- **client**: TestClient for API endpoint testing

### Test Data

- Sample courses and lessons with proper model structure
- Mock vector store with configurable search responses
- Mock AI generator for testing response generation
- Mock session manager for conversation history testing

## API Tests

### Coverage

- **Query Endpoint (`/api/query`)**:
  - Success scenarios with/without session IDs
  - Error handling (validation, system errors)
  - Edge cases (empty queries, long queries)
  - Response model validation

- **Courses Endpoint (`/api/courses`)**:
  - Course analytics retrieval
  - Empty course scenarios
  - Error handling
  - Response model validation

- **CORS & Middleware**:
  - Cross-origin request handling
  - Middleware configuration testing

### Static Files Handling

The test framework creates a separate test app that doesn't mount static files, avoiding import issues that would occur when testing the main app which references non-existent frontend files during testing.

## Adding New Tests

1. **API Tests**: Add to `test_api.py` using the `@pytest.mark.api` decorator
2. **Unit Tests**: Create new files for component testing using `@pytest.mark.unit`
3. **Integration Tests**: Use `@pytest.mark.integration` for end-to-end scenarios
4. **Fixtures**: Add shared fixtures to `conftest.py` for reuse across test files

## Mock Strategy

The testing framework uses comprehensive mocking to isolate components:

- **RAG System**: Mocked to return controlled responses
- **Vector Store**: Mocked for search and analytics operations
- **AI Generator**: Mocked for consistent response generation
- **Session Manager**: Mocked for conversation history management

This approach ensures tests run quickly and reliably without external dependencies.