"""
Tests for VectorStore ChromaDB operations and data loading
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import json

# Add the backend directory to the Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestVectorStore:
    """Test cases for VectorStore operations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('chromadb.PersistentClient') as mock_client, \
             patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            # Mock the client and collections
            mock_collection = Mock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            self.vector_store = VectorStore("./test_chroma", "test-model", max_results=5)
            
            # Override with properly mocked collections
            self.vector_store.course_catalog = Mock()
            self.vector_store.course_content = Mock()
        
        # Sample test data
        self.sample_course = Course(
            title="Introduction to Python",
            instructor="John Doe",
            course_link="https://example.com/python",
            lessons=[
                Lesson(lesson_number=1, title="Python Basics", lesson_link="https://example.com/lesson1"),
                Lesson(lesson_number=2, title="Variables", lesson_link="https://example.com/lesson2")
            ]
        )
        
        self.sample_chunks = [
            CourseChunk(
                content="Python is a programming language",
                course_title="Introduction to Python", 
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Variables store data in Python",
                course_title="Introduction to Python",
                lesson_number=2, 
                chunk_index=1
            )
        ]
    
    def test_initialization(self):
        """Test vector store initialization"""
        with patch('chromadb.PersistentClient') as mock_client, \
             patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):
            
            mock_client.return_value.get_or_create_collection.return_value = Mock()
            
            vs = VectorStore("/test/path", "test-embedding-model", max_results=10)
            
            assert vs.max_results == 10
            mock_client.assert_called_once()
            # Should create two collections
            assert mock_client.return_value.get_or_create_collection.call_count == 2
    
    def test_add_course_metadata(self):
        """Test adding course metadata to catalog"""
        self.vector_store.add_course_metadata(self.sample_course)
        
        # Verify course catalog was called correctly
        self.vector_store.course_catalog.add.assert_called_once()
        call_args = self.vector_store.course_catalog.add.call_args
        
        # Check document content
        assert call_args[1]["documents"] == ["Introduction to Python"]
        assert call_args[1]["ids"] == ["Introduction to Python"]
        
        # Check metadata structure
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == "Introduction to Python"
        assert metadata["instructor"] == "John Doe"
        assert metadata["course_link"] == "https://example.com/python"
        assert metadata["lesson_count"] == 2
        
        # Verify lessons are serialized as JSON
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == 2
        assert lessons_data[0]["lesson_number"] == 1
        assert lessons_data[0]["lesson_title"] == "Python Basics"
        assert lessons_data[0]["lesson_link"] == "https://example.com/lesson1"
    
    def test_add_course_content(self):
        """Test adding course content chunks"""
        self.vector_store.add_course_content(self.sample_chunks)
        
        # Verify content was added correctly
        self.vector_store.course_content.add.assert_called_once()
        call_args = self.vector_store.course_content.add.call_args
        
        # Check documents
        expected_docs = [
            "Python is a programming language",
            "Variables store data in Python"
        ]
        assert call_args[1]["documents"] == expected_docs
        
        # Check metadata
        expected_metadata = [
            {"course_title": "Introduction to Python", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Introduction to Python", "lesson_number": 2, "chunk_index": 1}
        ]
        assert call_args[1]["metadatas"] == expected_metadata
        
        # Check IDs
        expected_ids = ["Introduction_to_Python_0", "Introduction_to_Python_1"]
        assert call_args[1]["ids"] == expected_ids
    
    def test_add_course_content_empty(self):
        """Test adding empty content list"""
        self.vector_store.add_course_content([])
        
        # Should not call add when chunks is empty
        self.vector_store.course_content.add.assert_not_called()
    
    def test_search_basic(self):
        """Test basic search without filters"""
        # Mock ChromaDB response
        mock_chroma_result = {
            "documents": [["Python is great", "Learn Python basics"]],
            "metadatas": [[
                {"course_title": "Python Course", "lesson_number": 1},
                {"course_title": "Python Course", "lesson_number": 2}
            ]],
            "distances": [[0.1, 0.2]]
        }
        self.vector_store.course_content.query.return_value = mock_chroma_result
        
        result = self.vector_store.search("Python programming")
        
        # Verify search was called correctly
        self.vector_store.course_content.query.assert_called_once_with(
            query_texts=["Python programming"],
            n_results=5,
            where=None
        )
        
        # Verify result structure
        assert isinstance(result, SearchResults)
        assert result.documents == ["Python is great", "Learn Python basics"]
        assert len(result.metadata) == 2
        assert result.metadata[0]["course_title"] == "Python Course"
        assert result.error is None
    
    def test_search_with_course_filter(self):
        """Test search with course name filter"""
        # Mock course name resolution
        self.vector_store._resolve_course_name = Mock(return_value="Advanced Python")
        
        mock_chroma_result = {
            "documents": [["Advanced content"]],
            "metadatas": [[{"course_title": "Advanced Python", "lesson_number": 3}]],
            "distances": [[0.1]]
        }
        self.vector_store.course_content.query.return_value = mock_chroma_result
        
        result = self.vector_store.search("functions", course_name="Advanced")
        
        # Verify course name was resolved
        self.vector_store._resolve_course_name.assert_called_once_with("Advanced")
        
        # Verify search was called with filter
        self.vector_store.course_content.query.assert_called_once_with(
            query_texts=["functions"],
            n_results=5,
            where={"course_title": "Advanced Python"}
        )
    
    def test_search_with_lesson_filter(self):
        """Test search with lesson number filter"""
        mock_chroma_result = {
            "documents": [["Lesson content"]],
            "metadatas": [[{"course_title": "Some Course", "lesson_number": 5}]],
            "distances": [[0.1]]
        }
        self.vector_store.course_content.query.return_value = mock_chroma_result
        
        result = self.vector_store.search("topic", lesson_number=5)
        
        # Verify search was called with lesson filter
        self.vector_store.course_content.query.assert_called_once_with(
            query_texts=["topic"],
            n_results=5,
            where={"lesson_number": 5}
        )
    
    def test_search_with_both_filters(self):
        """Test search with both course and lesson filters"""
        self.vector_store._resolve_course_name = Mock(return_value="Data Science")
        
        mock_chroma_result = {
            "documents": [["Specific content"]],
            "metadatas": [[{"course_title": "Data Science", "lesson_number": 3}]],
            "distances": [[0.1]]
        }
        self.vector_store.course_content.query.return_value = mock_chroma_result
        
        result = self.vector_store.search("ML algorithms", course_name="Data Science", lesson_number=3)
        
        # Verify search was called with combined filter
        expected_filter = {"$and": [
            {"course_title": "Data Science"},
            {"lesson_number": 3}
        ]}
        self.vector_store.course_content.query.assert_called_once_with(
            query_texts=["ML algorithms"],
            n_results=5,
            where=expected_filter
        )
    
    def test_search_course_not_found(self):
        """Test search when course name can't be resolved"""
        self.vector_store._resolve_course_name = Mock(return_value=None)
        
        result = self.vector_store.search("query", course_name="Nonexistent Course")
        
        # Should return error result
        assert result.error == "No course found matching 'Nonexistent Course'"
        assert result.is_empty()
        
        # Should not search content
        self.vector_store.course_content.query.assert_not_called()
    
    def test_search_exception_handling(self):
        """Test search error handling"""
        self.vector_store.course_content.query.side_effect = Exception("Database error")
        
        result = self.vector_store.search("test query")
        
        assert result.error == "Search error: Database error"
        assert result.is_empty()
    
    def test_resolve_course_name_success(self):
        """Test successful course name resolution"""
        mock_chroma_result = {
            "documents": [["Python Course"]],
            "metadatas": [[{"title": "Introduction to Python"}]]
        }
        self.vector_store.course_catalog.query.return_value = mock_chroma_result
        
        result = self.vector_store._resolve_course_name("Python")
        
        # Verify catalog was queried
        self.vector_store.course_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
        
        assert result == "Introduction to Python"
    
    def test_resolve_course_name_no_results(self):
        """Test course name resolution with no results"""
        mock_chroma_result = {
            "documents": [[]],
            "metadatas": [[]]
        }
        self.vector_store.course_catalog.query.return_value = mock_chroma_result
        
        result = self.vector_store._resolve_course_name("Nonexistent")
        
        assert result is None
    
    def test_resolve_course_name_exception(self):
        """Test course name resolution error handling"""
        self.vector_store.course_catalog.query.side_effect = Exception("Query error")
        
        with patch('builtins.print') as mock_print:
            result = self.vector_store._resolve_course_name("test")
            
            assert result is None
            mock_print.assert_called_with("Error resolving course name: Query error")
    
    def test_get_existing_course_titles(self):
        """Test getting existing course titles"""
        mock_result = {"ids": ["Course 1", "Course 2", "Course 3"]}
        self.vector_store.course_catalog.get.return_value = mock_result
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == ["Course 1", "Course 2", "Course 3"]
        self.vector_store.course_catalog.get.assert_called_once()
    
    def test_get_existing_course_titles_empty(self):
        """Test getting existing course titles when none exist"""
        mock_result = {"ids": []}
        self.vector_store.course_catalog.get.return_value = mock_result
        
        titles = self.vector_store.get_existing_course_titles()
        
        assert titles == []
    
    def test_get_existing_course_titles_exception(self):
        """Test error handling for getting course titles"""
        self.vector_store.course_catalog.get.side_effect = Exception("DB error")
        
        with patch('builtins.print') as mock_print:
            titles = self.vector_store.get_existing_course_titles()
            
            assert titles == []
            mock_print.assert_called_with("Error getting existing course titles: DB error")
    
    def test_get_course_count(self):
        """Test getting course count"""
        mock_result = {"ids": ["Course 1", "Course 2"]}
        self.vector_store.course_catalog.get.return_value = mock_result
        
        count = self.vector_store.get_course_count()
        
        assert count == 2
    
    def test_get_lesson_link(self):
        """Test getting lesson link"""
        # Mock course data with lessons
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "https://example.com/lesson2"}
        ]
        mock_result = {
            "metadatas": [{"lessons_json": json.dumps(lessons_data)}]
        }
        self.vector_store.course_catalog.get.return_value = mock_result
        
        link = self.vector_store.get_lesson_link("Test Course", 2)
        
        # Verify course was queried by ID
        self.vector_store.course_catalog.get.assert_called_once_with(ids=["Test Course"])
        
        assert link == "https://example.com/lesson2"
    
    def test_get_lesson_link_not_found(self):
        """Test getting lesson link when lesson doesn't exist"""
        lessons_data = [{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"}]
        mock_result = {"metadatas": [{"lessons_json": json.dumps(lessons_data)}]}
        self.vector_store.course_catalog.get.return_value = mock_result
        
        link = self.vector_store.get_lesson_link("Test Course", 999)
        
        assert link is None
    
    def test_build_filter_combinations(self):
        """Test various filter combinations"""
        # No filters
        filter_result = self.vector_store._build_filter(None, None)
        assert filter_result is None
        
        # Course only
        filter_result = self.vector_store._build_filter("Python Course", None)
        assert filter_result == {"course_title": "Python Course"}
        
        # Lesson only
        filter_result = self.vector_store._build_filter(None, 5)
        assert filter_result == {"lesson_number": 5}
        
        # Both filters
        filter_result = self.vector_store._build_filter("Python Course", 5)
        expected = {"$and": [
            {"course_title": "Python Course"},
            {"lesson_number": 5}
        ]}
        assert filter_result == expected
    
    def test_clear_all_data(self):
        """Test clearing all data"""
        with patch.object(self.vector_store, 'client') as mock_client:
            mock_client.delete_collection = Mock()
            mock_client.get_or_create_collection = Mock()
            
            self.vector_store.clear_all_data()
            
            # Should delete both collections
            assert mock_client.delete_collection.call_count == 2
            mock_client.delete_collection.assert_any_call("course_catalog")
            mock_client.delete_collection.assert_any_call("course_content")
            
            # Should recreate collections
            assert mock_client.get_or_create_collection.call_count == 2


class TestSearchResults:
    """Test SearchResults helper class"""
    
    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_result = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"meta1": "val1"}, {"meta2": "val2"}]],
            "distances": [[0.1, 0.2]]
        }
        
        result = SearchResults.from_chroma(chroma_result)
        
        assert result.documents == ["doc1", "doc2"]
        assert result.metadata == [{"meta1": "val1"}, {"meta2": "val2"}]
        assert result.distances == [0.1, 0.2]
        assert result.error is None
        assert not result.is_empty()
    
    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB response"""
        chroma_result = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        result = SearchResults.from_chroma(chroma_result)
        
        assert result.documents == []
        assert result.metadata == []
        assert result.distances == []
        assert result.is_empty()
    
    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        result = SearchResults.empty("Database connection failed")
        
        assert result.documents == []
        assert result.metadata == []
        assert result.distances == []
        assert result.error == "Database connection failed"
        assert result.is_empty()


if __name__ == "__main__":
    pytest.main([__file__])