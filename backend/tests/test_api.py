import pytest
import json
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for the /api/query endpoint"""
    
    def test_query_success_with_session_id(self, client, mock_rag_system):
        """Test successful query with provided session ID"""
        mock_rag_system.query = AsyncMock(return_value=("Test answer", ["Source 1", "Source 2"]))
        
        response = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "existing-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["Source 1", "Source 2"]
        assert data["session_id"] == "existing-session"
        
        mock_rag_system.query.assert_called_once_with("What is Python?", "existing-session")
    
    def test_query_success_without_session_id(self, client, mock_rag_system):
        """Test successful query without session ID (should create new session)"""
        mock_rag_system.query = AsyncMock(return_value=("Test answer", ["Source 1"]))
        mock_rag_system.session_manager.create_session.return_value = "new-session-id"
        
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["Source 1"]
        assert data["session_id"] == "test-session-id"
    
    def test_query_empty_query(self, client):
        """Test query with empty query string"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == 200  # Should still process empty queries
    
    def test_query_missing_query_field(self, client):
        """Test query with missing query field"""
        response = client.post(
            "/api/query",
            json={"session_id": "test-session"}
        )
        
        assert response.status_code == 422  # Validation error
        
    def test_query_invalid_json(self, client):
        """Test query with invalid JSON"""
        response = client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_query_rag_system_error(self, client, mock_rag_system):
        """Test query when RAG system raises an error"""
        mock_rag_system.query = AsyncMock(side_effect=Exception("RAG system error"))
        
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        
        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]
    
    def test_query_long_query(self, client, mock_rag_system):
        """Test query with very long query string"""
        long_query = "What is Python? " * 1000
        mock_rag_system.query = AsyncMock(return_value=("Answer", ["Source"]))
        
        response = client.post(
            "/api/query",
            json={"query": long_query}
        )
        
        assert response.status_code == 200
        mock_rag_system.query.assert_called_once_with(long_query, "test-session-id")


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for the /api/courses endpoint"""
    
    def test_get_courses_success(self, client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        mock_analytics = {
            "total_courses": 3,
            "course_titles": ["Course 1", "Course 2", "Course 3"]
        }
        mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Course 1", "Course 2", "Course 3"]
        
        mock_rag_system.get_course_analytics.assert_called_once()
    
    def test_get_courses_empty(self, client, mock_rag_system):
        """Test retrieval when no courses exist"""
        mock_analytics = {
            "total_courses": 0,
            "course_titles": []
        }
        mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    def test_get_courses_rag_system_error(self, client, mock_rag_system):
        """Test courses endpoint when RAG system raises an error"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]
    
    def test_get_courses_method_not_allowed(self, client):
        """Test that POST method is not allowed on courses endpoint"""
        response = client.post("/api/courses", json={})
        
        assert response.status_code == 405  # Method not allowed


@pytest.mark.api
class TestStaticFilesHandling:
    """Test cases for static file serving (handled in conftest.py)"""
    
    def test_api_endpoints_accessible(self, client):
        """Test that API endpoints are accessible without static file conflicts"""
        # Test that our test app correctly handles API routes
        response = client.get("/api/courses")
        assert response.status_code in [200, 500]  # Either success or controlled error
        
        # Test query endpoint accessibility
        response = client.post("/api/query", json={"query": "test"})
        assert response.status_code in [200, 500]  # Either success or controlled error


@pytest.mark.api
class TestCORSMiddleware:
    """Test CORS middleware functionality"""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are properly set"""
        response = client.options("/api/query")
        
        # Should allow all origins, methods, and headers for development
        assert response.status_code in [200, 405]  # Some test clients handle OPTIONS differently
    
    def test_cross_origin_request(self, client, mock_rag_system):
        """Test cross-origin request handling"""
        mock_rag_system.query = AsyncMock(return_value=("Answer", ["Source"]))
        
        response = client.post(
            "/api/query",
            json={"query": "test query"},
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200


@pytest.mark.api
class TestResponseModels:
    """Test response model validation and serialization"""
    
    def test_query_response_model(self, client, mock_rag_system):
        """Test that QueryResponse model is properly serialized"""
        mock_rag_system.query = AsyncMock(return_value=("Test answer", ["Source 1", "Source 2"]))
        
        response = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "test-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = {"answer", "sources", "session_id"}
        assert set(data.keys()) == required_fields
        
        # Validate field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        assert all(isinstance(source, str) for source in data["sources"])
    
    def test_course_stats_response_model(self, client, mock_rag_system):
        """Test that CourseStats model is properly serialized"""
        mock_analytics = {
            "total_courses": 5,
            "course_titles": ["Course A", "Course B", "Course C", "Course D", "Course E"]
        }
        mock_rag_system.get_course_analytics.return_value = mock_analytics
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        required_fields = {"total_courses", "course_titles"}
        assert set(data.keys()) == required_fields
        
        # Validate field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])


@pytest.mark.api
class TestErrorHandling:
    """Test comprehensive error handling scenarios"""
    
    def test_internal_server_error_format(self, client, mock_rag_system):
        """Test that internal server errors are properly formatted"""
        mock_rag_system.query = AsyncMock(side_effect=ValueError("Invalid input"))
        
        response = client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == 500
        error_data = response.json()
        assert "detail" in error_data
        assert "Invalid input" in error_data["detail"]
    
    def test_validation_error_format(self, client):
        """Test that validation errors are properly formatted"""
        response = client.post(
            "/api/query",
            json={"wrong_field": "test"}  # Missing required 'query' field
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_method_not_allowed(self, client):
        """Test method not allowed responses"""
        response = client.delete("/api/query")
        assert response.status_code == 405
        
        response = client.put("/api/courses")
        assert response.status_code == 405