import pytest
import sys
import os


def test_pytest_configuration():
    """Test that pytest configuration is working correctly"""
    assert pytest.__version__ >= "8.0"


def test_python_path():
    """Test that Python path includes backend directory"""
    backend_path = os.path.join(os.path.dirname(__file__), '..')
    assert backend_path in sys.path or any(backend_path in p for p in sys.path)


@pytest.mark.unit
def test_basic_functionality():
    """Basic functionality test to verify testing infrastructure"""
    assert 1 + 1 == 2
    assert "test" in "testing"


def test_fixture_availability(mock_config):
    """Test that fixtures from conftest.py are available"""
    assert mock_config is not None
    assert hasattr(mock_config, 'CHUNK_SIZE')
    assert mock_config.CHUNK_SIZE == 800


def test_sample_data_fixtures(sample_course, sample_course_chunks):
    """Test that sample data fixtures work correctly"""
    assert sample_course.title == "Test Course"
    assert sample_course.instructor == "Test Instructor"
    assert len(sample_course.lessons) == 2
    
    assert len(sample_course_chunks) == 3
    assert all(chunk.course_title == "Test Course" for chunk in sample_course_chunks)


def test_mock_fixtures(mock_vector_store, mock_ai_generator, mock_session_manager):
    """Test that mock fixtures are properly configured"""
    assert mock_vector_store is not None
    assert mock_ai_generator is not None
    assert mock_session_manager is not None
    
    # Test mock methods exist
    assert hasattr(mock_vector_store, 'search')
    assert hasattr(mock_ai_generator, 'generate_response')
    assert hasattr(mock_session_manager, 'create_session')


@pytest.mark.api
def test_client_fixture(client):
    """Test that the test client fixture works"""
    assert client is not None
    # Basic smoke test - the client should be able to make requests
    response = client.get("/api/courses")
    # Should either succeed or fail in a controlled way (not crash)
    assert response.status_code in [200, 404, 500]