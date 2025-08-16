"""
Integration tests for RAGSystem end-to-end functionality
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the backend directory to the Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class MockConfig:
    """Mock configuration for testing"""
    ANTHROPIC_API_KEY = "test-api-key"
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514" 
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    MAX_RESULTS = 5  # Fixed value, not 0
    MAX_HISTORY = 2
    CHROMA_PATH = "./test_chroma"


class TestRAGSystemIntegration:
    """Integration tests for RAGSystem"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MockConfig()
        
        # Patch all the dependencies to avoid actual initialization
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vs, \
             patch('rag_system.AIGenerator') as mock_ai, \
             patch('rag_system.SessionManager'):
            
            self.rag_system = RAGSystem(self.config)
            
            # Store references to mocked components for testing
            self.mock_vector_store = self.rag_system.vector_store
            self.mock_ai_generator = self.rag_system.ai_generator
            self.mock_session_manager = self.rag_system.session_manager
            
            # Mock the tool_manager methods properly
            self.mock_tool_manager = Mock()
            self.rag_system.tool_manager = self.mock_tool_manager
    
    def test_initialization(self):
        """Test RAGSystem initialization"""
        # Should have initialized all components
        assert self.rag_system.config == self.config
        assert self.rag_system.document_processor is not None
        assert self.rag_system.vector_store is not None
        assert self.rag_system.ai_generator is not None
        assert self.rag_system.session_manager is not None
        assert self.rag_system.tool_manager is not None
        assert self.rag_system.search_tool is not None
        assert self.rag_system.outline_tool is not None
    
    def test_query_without_session(self):
        """Test query processing without session ID"""
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Python is a programming language"
        
        # Mock tool manager sources
        self.mock_tool_manager.get_last_sources.return_value = ["Python Course - Lesson 1"]
        
        response, sources = self.rag_system.query("What is Python?")
        
        # Verify AI generator was called correctly
        self.mock_ai_generator.generate_response.assert_called_once()
        call_args = self.mock_ai_generator.generate_response.call_args
        
        assert "Answer this question about course materials: What is Python?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
        
        # Verify response and sources
        assert response == "Python is a programming language"
        assert sources == ["Python Course - Lesson 1"]
        
        # Verify sources were retrieved and reset
        self.mock_tool_manager.get_last_sources.assert_called_once()
        self.mock_tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session(self):
        """Test query processing with session ID"""
        session_id = "test-session-123"
        
        # Mock session manager
        self.mock_session_manager.get_conversation_history.return_value = "Previous conversation"
        
        # Mock AI generator response
        self.mock_ai_generator.generate_response.return_value = "Follow-up response"
        
        # Mock empty sources
        self.mock_tool_manager.get_last_sources.return_value = []
        
        response, sources = self.rag_system.query("Follow-up question", session_id)
        
        # Verify session history was retrieved
        self.mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify AI generator was called with history
        call_args = self.mock_ai_generator.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"
        
        # Verify session was updated with new exchange
        self.mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Follow-up response"
        )
        
        assert response == "Follow-up response"
        assert sources == []
    
    def test_query_with_tool_sources(self):
        """Test query that returns tool sources with links"""
        # Mock AI response
        self.mock_ai_generator.generate_response.return_value = "Course content found"
        
        # Mock tool sources with link format
        mock_sources = [
            "Python Basics|https://example.com/lesson1",
            "Advanced Python - Lesson 2|https://example.com/lesson2"
        ]
        self.mock_tool_manager.get_last_sources.return_value = mock_sources
        
        response, sources = self.rag_system.query("Search for Python content")
        
        # Verify sources are returned correctly
        assert sources == mock_sources
        assert response == "Course content found"
    
    def test_add_course_document_success(self):
        """Test successfully adding a course document"""
        file_path = "/path/to/course.pdf"
        
        # Mock document processor
        mock_course = Course(title="Test Course", instructor="Test Instructor")
        mock_chunks = [
            CourseChunk(content="Content 1", course_title="Test Course", chunk_index=0),
            CourseChunk(content="Content 2", course_title="Test Course", chunk_index=1)
        ]
        
        self.rag_system.document_processor.process_course_document.return_value = (mock_course, mock_chunks)
        
        course, chunk_count = self.rag_system.add_course_document(file_path)
        
        # Verify document was processed
        self.rag_system.document_processor.process_course_document.assert_called_once_with(file_path)
        
        # Verify data was added to vector store
        self.mock_vector_store.add_course_metadata.assert_called_once_with(mock_course)
        self.mock_vector_store.add_course_content.assert_called_once_with(mock_chunks)
        
        assert course == mock_course
        assert chunk_count == 2
    
    def test_add_course_document_error(self):
        """Test error handling when adding course document"""
        file_path = "/path/to/invalid.pdf"
        
        # Mock processing error
        self.rag_system.document_processor.process_course_document.side_effect = Exception("Processing failed")
        
        with patch('builtins.print') as mock_print:
            course, chunk_count = self.rag_system.add_course_document(file_path)
            
            assert course is None
            assert chunk_count == 0
            mock_print.assert_called_with(f"Error processing course document {file_path}: Processing failed")
    
    def test_add_course_folder_with_existing_courses(self):
        """Test adding course folder with duplicate detection"""
        folder_path = "/path/to/courses"
        
        # Mock existing course titles
        self.mock_vector_store.get_existing_course_titles.return_value = ["Existing Course"]
        
        # Mock file system
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=["course1.pdf", "course2.txt", "other.doc"]), \
             patch('os.path.isfile', return_value=True):
            
            # Mock document processing
            mock_course1 = Course(title="New Course", instructor="Instructor")
            mock_course2 = Course(title="Existing Course", instructor="Instructor") 
            
            mock_chunks1 = [CourseChunk(content="Content", course_title="New Course", chunk_index=0)]
            mock_chunks2 = [CourseChunk(content="Content", course_title="Existing Course", chunk_index=0)]
            
            self.rag_system.document_processor.process_course_document.side_effect = [
                (mock_course1, mock_chunks1),  # New course - should be added
                (mock_course2, mock_chunks2)   # Existing course - should be skipped
            ]
            
            with patch('builtins.print') as mock_print:
                total_courses, total_chunks = self.rag_system.add_course_folder(folder_path)
                
                # Should only add the new course
                assert total_courses == 1
                assert total_chunks == 1
                
                # Verify print statements
                mock_print.assert_any_call("Added new course: New Course (1 chunks)")
                mock_print.assert_any_call("Course already exists: Existing Course - skipping")
    
    def test_add_course_folder_clear_existing(self):
        """Test adding course folder with clear_existing=True"""
        folder_path = "/path/to/courses"
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=[]), \
             patch('builtins.print') as mock_print:
            
            self.rag_system.add_course_folder(folder_path, clear_existing=True)
            
            # Should clear existing data
            self.mock_vector_store.clear_all_data.assert_called_once()
            mock_print.assert_any_call("Clearing existing data for fresh rebuild...")
    
    def test_add_course_folder_nonexistent_path(self):
        """Test adding course folder with nonexistent path"""
        folder_path = "/nonexistent/path"
        
        with patch('os.path.exists', return_value=False), \
             patch('builtins.print') as mock_print:
            
            courses, chunks = self.rag_system.add_course_folder(folder_path)
            
            assert courses == 0
            assert chunks == 0
            mock_print.assert_called_with("Folder /nonexistent/path does not exist")
    
    def test_get_course_analytics(self):
        """Test getting course analytics"""
        # Mock vector store analytics
        self.mock_vector_store.get_course_count.return_value = 5
        self.mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        analytics = self.rag_system.get_course_analytics()
        
        expected = {
            "total_courses": 5,
            "course_titles": ["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        }
        
        assert analytics == expected
        
        # Verify methods were called
        self.mock_vector_store.get_course_count.assert_called_once()
        self.mock_vector_store.get_existing_course_titles.assert_called_once()
    
    def test_tool_manager_registration(self):
        """Test that tools are properly registered with tool manager"""
        # Tools should be registered during initialization
        # We can verify this by checking that the tool manager has tools
        assert self.rag_system.search_tool is not None
        assert self.rag_system.outline_tool is not None
        
        # The actual registration calls would have been made during init
        # Since we're mocking, we can't easily verify the calls, but we can
        # verify the tools exist and have the right vector store reference
        assert self.rag_system.search_tool.store == self.mock_vector_store
        assert self.rag_system.outline_tool.store == self.mock_vector_store
    
    def test_query_exception_handling(self):
        """Test error handling during query processing"""
        # Mock AI generator to raise an exception
        self.mock_ai_generator.generate_response.side_effect = Exception("AI generation failed")
        
        # The method doesn't explicitly handle exceptions, so they should propagate
        with pytest.raises(Exception, match="AI generation failed"):
            self.rag_system.query("test query")
    
    def test_prompt_construction(self):
        """Test that query prompts are constructed correctly"""
        user_query = "What is machine learning?"
        expected_prompt = "Answer this question about course materials: What is machine learning?"
        
        self.mock_ai_generator.generate_response.return_value = "ML response"
        self.mock_tool_manager.get_last_sources.return_value = []
        
        self.rag_system.query(user_query)
        
        # Verify the prompt was constructed correctly
        call_args = self.mock_ai_generator.generate_response.call_args
        actual_prompt = call_args[1]["query"]
        
        assert actual_prompt == expected_prompt


class TestRAGSystemRealScenarios:
    """Test RAGSystem with more realistic scenarios"""
    
    def setup_method(self):
        """Set up with less mocking to test real integration points"""
        self.config = MockConfig()
    
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_full_query_flow_simulation(self, mock_session, mock_ai, mock_vs, mock_doc):
        """Test a full query flow with realistic tool interactions"""
        # Initialize RAGSystem
        rag_system = RAGSystem(self.config)
        
        # Mock a successful search tool execution
        def mock_tool_execution(tool_name, **kwargs):
            if tool_name == "search_course_content":
                return "[Python Course - Lesson 1]\nPython is a high-level programming language..."
            return "Tool not found"
        
        rag_system.tool_manager.execute_tool = Mock(side_effect=mock_tool_execution)
        rag_system.tool_manager.get_last_sources = Mock(return_value=["Python Course - Lesson 1"])
        
        # Mock AI to simulate tool use
        def mock_ai_response(query, **kwargs):
            if "Python" in query:
                return "Based on the course content, Python is a programming language..."
            return "I don't have information about that."
        
        rag_system.ai_generator.generate_response = Mock(side_effect=mock_ai_response)
        
        # Execute query
        response, sources = rag_system.query("Tell me about Python")
        
        # Verify realistic behavior
        assert "Python" in response
        assert len(sources) > 0
        assert sources[0] == "Python Course - Lesson 1"


if __name__ == "__main__":
    pytest.main([__file__])