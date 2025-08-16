"""
Tests for CourseSearchTool.execute() method
"""
import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the backend directory to the Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool execute method"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)
    
    def test_execute_successful_search(self):
        """Test successful search with results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Course content chunk 1", "Course content chunk 2"],
            metadata=[
                {"course_title": "Introduction to Python", "lesson_number": 1},
                {"course_title": "Introduction to Python", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        # Execute search
        result = self.search_tool.execute("python basics")
        
        # Verify search was called correctly
        self.mock_vector_store.search.assert_called_once_with(
            query="python basics",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result format
        assert "[Introduction to Python - Lesson 1]" in result
        assert "[Introduction to Python - Lesson 2]" in result
        assert "Course content chunk 1" in result
        assert "Course content chunk 2" in result
        
        # Verify sources were tracked
        assert len(self.search_tool.last_sources) == 2
        assert "Introduction to Python - Lesson 1" in self.search_tool.last_sources[0]
        assert "Introduction to Python - Lesson 2" in self.search_tool.last_sources[1]
    
    def test_execute_with_course_name_filter(self):
        """Test search with course name filtering"""
        mock_results = SearchResults(
            documents=["Filtered course content"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("functions", course_name="Advanced Python")
        
        # Verify search was called with course filter
        self.mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name="Advanced Python",
            lesson_number=None
        )
        
        assert "[Advanced Python - Lesson 3]" in result
        assert "Filtered course content" in result
    
    def test_execute_with_lesson_number_filter(self):
        """Test search with lesson number filtering"""
        mock_results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Data Science", "lesson_number": 5}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("machine learning", lesson_number=5)
        
        # Verify search was called with lesson filter
        self.mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=5
        )
        
        assert "[Data Science - Lesson 5]" in result
        assert "Lesson specific content" in result
    
    def test_execute_with_both_filters(self):
        """Test search with both course name and lesson number filters"""
        mock_results = SearchResults(
            documents=["Specific lesson content"],
            metadata=[{"course_title": "Web Development", "lesson_number": 2}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("HTML tags", course_name="Web Development", lesson_number=2)
        
        # Verify search was called with both filters
        self.mock_vector_store.search.assert_called_once_with(
            query="HTML tags",
            course_name="Web Development", 
            lesson_number=2
        )
        
        assert "[Web Development - Lesson 2]" in result
        assert "Specific lesson content" in result
    
    def test_execute_search_error(self):
        """Test handling of search errors"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        # Should return error message directly
        assert result == "Database connection failed"
        
        # Should not track sources for errors
        assert self.search_tool.last_sources == []
    
    def test_execute_empty_results(self):
        """Test handling of empty search results"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("nonexistent topic")
        
        assert result == "No relevant content found."
        assert self.search_tool.last_sources == []
    
    def test_execute_empty_results_with_course_filter(self):
        """Test empty results message includes course filter info"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test", course_name="Nonexistent Course")
        
        assert result == "No relevant content found in course 'Nonexistent Course'."
    
    def test_execute_empty_results_with_lesson_filter(self):
        """Test empty results message includes lesson filter info"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test", lesson_number=999)
        
        assert result == "No relevant content found in lesson 999."
    
    def test_execute_empty_results_with_both_filters(self):
        """Test empty results message includes both filter info"""
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test", course_name="Course", lesson_number=5)
        
        assert result == "No relevant content found in course 'Course' in lesson 5."
    
    def test_execute_with_lesson_links(self):
        """Test that lesson links are properly included in sources"""
        # Mock get_lesson_link to return a URL
        self.mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        mock_results = SearchResults(
            documents=["Course content with link"],
            metadata=[{"course_title": "Course With Links", "lesson_number": 1}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test content")
        
        # Verify lesson link was requested
        self.mock_vector_store.get_lesson_link.assert_called_once_with("Course With Links", 1)
        
        # Verify source includes link using delimiter format
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0] == "Course With Links - Lesson 1|https://example.com/lesson1"
    
    def test_execute_with_unknown_metadata(self):
        """Test handling of incomplete/unknown metadata"""
        mock_results = SearchResults(
            documents=["Content with incomplete metadata"],
            metadata=[{"course_title": "unknown", "lesson_number": None}],
            distances=[0.1],
            error=None
        )
        self.mock_vector_store.search.return_value = mock_results
        
        result = self.search_tool.execute("test query")
        
        # Should handle unknown course title gracefully
        assert "[unknown]" in result
        assert "Content with incomplete metadata" in result
        assert len(self.search_tool.last_sources) == 1
        assert self.search_tool.last_sources[0] == "unknown"
        
        # Should not try to get lesson link for unknown course
        self.mock_vector_store.get_lesson_link.assert_not_called()
    
    def test_get_tool_definition(self):
        """Test that tool definition is properly structured"""
        tool_def = self.search_tool.get_tool_definition()
        
        # Verify required structure
        assert tool_def["name"] == "search_course_content"
        assert "description" in tool_def
        assert "input_schema" in tool_def
        
        # Verify required and optional parameters
        schema = tool_def["input_schema"]
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]
        
        # Verify parameter types
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["course_name"]["type"] == "string"
        assert schema["properties"]["lesson_number"]["type"] == "integer"


if __name__ == "__main__":
    pytest.main([__file__])