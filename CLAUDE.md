# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials that allows users to query educational content using semantic search and AI-powered responses. The system uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a FastAPI web interface.

## Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

- **RAGSystem** (`backend/rag_system.py`): Main orchestrator that coordinates all components
- **VectorStore** (`backend/vector_store.py`): ChromaDB wrapper for vector storage and semantic search
- **AIGenerator** (`backend/ai_generator.py`): Anthropic Claude API wrapper with tool support
- **DocumentProcessor** (`backend/document_processor.py`): Processes course documents into structured chunks
- **SessionManager** (`backend/session_manager.py`): Manages conversation history and sessions
- **ToolManager & CourseSearchTool** (`backend/search_tools.py`): Tool-based search system for AI agent

### Data Models

- **Course**: Represents a complete course with title, instructor, and lessons
- **Lesson**: Individual lesson within a course with title and optional link
- **CourseChunk**: Text chunks for vector storage with metadata (course, lesson, position)

### API Layer

FastAPI application (`backend/app.py`) with:
- `/api/query` - Process queries and return responses with sources
- `/api/courses` - Get course analytics and statistics
- Static file serving for frontend

## Development Commands

### Running the Application

```bash
# Quick start (recommended)
chmod +x run.sh
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management

```bash
# Install dependencies
uv sync

# Add new dependency
uv add package_name
```

### Environment Setup

Create `.env` file in root directory:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Key Configuration

Configuration is centralized in `backend/config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation messages to remember
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer model
- `ANTHROPIC_MODEL: "claude-sonnet-4-20250514"` - Claude model version

## Data Flow

1. **Document Processing**: Course documents in `docs/` are processed into structured chunks
2. **Vector Storage**: Content and metadata stored in ChromaDB with embeddings
3. **Query Processing**: User queries trigger semantic search via AI tools
4. **Response Generation**: Claude generates responses using retrieved context
5. **Session Management**: Conversation history maintained for context

## Important Implementation Details

- **Tool-based Search**: AI uses tools to search course content rather than direct RAG
- **Deduplication**: System avoids re-processing existing courses by checking titles
- **CORS Configuration**: Allows all origins for development (`allow_origins=["*"]`)
- **Session Persistence**: Sessions created automatically if not provided
- **Error Handling**: Comprehensive error handling with HTTP status codes

## Frontend Integration

Static frontend files in `frontend/` directory:
- `index.html` - Main web interface
- `script.js` - Frontend logic and API calls
- `style.css` - Styling

Frontend served via FastAPI's static file handler with development-friendly no-cache headers.

## Dependencies

Key Python packages:
- `fastapi==0.116.1` & `uvicorn==0.35.0` - Web framework and server
- `chromadb==1.0.15` - Vector database
- `anthropic==0.58.2` - Claude API client
- `sentence-transformers==5.0.0` - Embedding model
- `python-multipart==0.0.20` - File upload support
- `python-dotenv==1.1.1` - Environment variable loading