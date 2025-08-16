import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import json
from pathlib import Path

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    config = Mock(spec=Config)
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.DB_PATH = ":memory:"
    return config


@pytest.fixture
def temp_directory():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Test Course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Lesson 1", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Lesson 2", lesson_link="https://example.com/lesson2")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="This is the first chunk of content from lesson 1.",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This is the second chunk of content from lesson 1.",
            course_title="Test Course", 
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="This is content from lesson 2.",
            course_title="Test Course",
            lesson_number=2, 
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    vector_store = Mock()
    vector_store.add_chunks = Mock()
    vector_store.search = Mock(return_value=[])
    vector_store.get_course_analytics = Mock(return_value={
        "total_courses": 1,
        "course_titles": ["Test Course"]
    })
    return vector_store


@pytest.fixture
def mock_ai_generator():
    """Mock AI generator for testing"""
    ai_generator = Mock()
    ai_generator.generate_response = AsyncMock(return_value="Test response")
    return ai_generator


@pytest.fixture
def mock_document_processor():
    """Mock document processor for testing"""
    processor = Mock()
    processor.process_course_folder = Mock()
    return processor


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    session_manager = Mock()
    session_manager.create_session = Mock(return_value="test-session-id")
    session_manager.add_message = Mock()
    session_manager.get_recent_history = Mock(return_value=[])
    return session_manager


@pytest.fixture
def mock_rag_system(mock_config, mock_vector_store, mock_ai_generator, 
                   mock_document_processor, mock_session_manager):
    """Mock RAG system for testing"""
    rag_system = Mock()
    rag_system.query = Mock(return_value=("Test response", ["Source 1", "Source 2"]))
    rag_system.get_course_analytics = Mock(return_value={
        "total_courses": 1,
        "course_titles": ["Test Course"]
    })
    rag_system.session_manager = mock_session_manager
    return rag_system


@pytest.fixture
def test_app():
    """Test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    app = FastAPI(title="Test RAG System")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    return app, QueryRequest, QueryResponse, CourseStats


@pytest.fixture
def client(test_app, mock_rag_system):
    """Test client with mocked dependencies"""
    from fastapi import HTTPException
    
    app, QueryRequest, QueryResponse, CourseStats = test_app
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id or "test-session-id"
            # Handle both sync and async mock calls
            result = mock_rag_system.query(request.query, session_id)
            if hasattr(result, '__await__'):
                answer, sources = await result
            else:
                answer, sources = result
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return TestClient(app)


@pytest.fixture
def sample_documents():
    """Sample document content for testing"""
    return {
        "course1.txt": """Course Title: Introduction to Python
Instructor: Dr. Smith

Lesson 1: Python Basics
This lesson covers the fundamentals of Python programming including variables, data types, and basic operations.

Lesson 2: Control Structures  
This lesson explains conditional statements, loops, and how to control program flow in Python.
""",
        "course2.txt": """Course Title: Web Development
Instructor: Prof. Johnson

Lesson 1: HTML Fundamentals
Learn the basic structure of HTML documents and common HTML elements.

Lesson 2: CSS Styling
Introduction to CSS for styling web pages and creating responsive layouts.
"""
    }


@pytest.fixture
def create_test_documents(temp_directory, sample_documents):
    """Create test document files in temporary directory"""
    def _create_docs():
        docs_path = os.path.join(temp_directory, "docs")
        os.makedirs(docs_path, exist_ok=True)
        
        for filename, content in sample_documents.items():
            filepath = os.path.join(docs_path, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        return docs_path
    
    return _create_docs