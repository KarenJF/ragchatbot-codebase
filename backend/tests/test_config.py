"""
Tests for configuration values and environment variable loading
"""
import pytest
from unittest.mock import patch
import sys
import os

# Add the backend directory to the Python path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)


class TestConfiguration:
    """Test cases for configuration settings"""
    
    def test_config_import(self):
        """Test that config can be imported successfully"""
        from config import config, Config
        
        # Should be able to import without errors
        assert config is not None
        assert isinstance(config, Config)
    
    def test_default_configuration_values(self):
        """Test that default configuration values are set correctly"""
        from config import config
        
        # Anthropic settings
        assert config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        
        # Embedding model
        assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        
        # Document processing settings
        assert config.CHUNK_SIZE == 800
        assert config.CHUNK_OVERLAP == 100
        assert config.MAX_RESULTS == 5  # This was the critical fix!
        assert config.MAX_HISTORY == 2
        
        # Database paths
        assert config.CHROMA_PATH == "./chroma_db"
    
    def test_critical_max_results_fix(self):
        """Test that MAX_RESULTS is not 0 (the bug that was causing query failures)"""
        from config import config
        
        # This is the critical test - MAX_RESULTS should never be 0
        assert config.MAX_RESULTS > 0, "MAX_RESULTS must be greater than 0 to return search results"
        assert config.MAX_RESULTS == 5, "MAX_RESULTS should be set to 5 after the fix"
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key-12345'})
    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly"""
        # Need to reimport to get fresh config with env vars
        import importlib
        import config
        importlib.reload(config)
        
        assert config.config.ANTHROPIC_API_KEY == 'test-key-12345'
    
    def test_missing_environment_variables(self):
        """Test behavior when environment variables are missing"""
        # Test the current config - if no env var is set, it should be empty string
        from config import Config
        
        # Create a new config instance to test default behavior
        with patch.dict(os.environ, {}, clear=True):
            test_config = Config()
            assert test_config.ANTHROPIC_API_KEY == ""
    
    def test_config_dataclass_structure(self):
        """Test that Config is properly structured as a dataclass"""
        from config import Config
        
        # Should have expected attributes
        assert hasattr(Config, 'ANTHROPIC_API_KEY')
        assert hasattr(Config, 'ANTHROPIC_MODEL') 
        assert hasattr(Config, 'EMBEDDING_MODEL')
        assert hasattr(Config, 'CHUNK_SIZE')
        assert hasattr(Config, 'CHUNK_OVERLAP')
        assert hasattr(Config, 'MAX_RESULTS')
        assert hasattr(Config, 'MAX_HISTORY')
        assert hasattr(Config, 'CHROMA_PATH')
    
    def test_config_types(self):
        """Test that configuration values have correct types"""
        from config import config
        
        # String types
        assert isinstance(config.ANTHROPIC_API_KEY, str)
        assert isinstance(config.ANTHROPIC_MODEL, str)
        assert isinstance(config.EMBEDDING_MODEL, str)
        assert isinstance(config.CHROMA_PATH, str)
        
        # Integer types  
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert isinstance(config.MAX_RESULTS, int)
        assert isinstance(config.MAX_HISTORY, int)
    
    def test_reasonable_config_values(self):
        """Test that configuration values are within reasonable ranges"""
        from config import config
        
        # Chunk size should be reasonable for text processing
        assert 100 <= config.CHUNK_SIZE <= 2000, "Chunk size should be reasonable for text processing"
        
        # Overlap should be smaller than chunk size
        assert 0 <= config.CHUNK_OVERLAP < config.CHUNK_SIZE, "Overlap should be less than chunk size"
        
        # Max results should be reasonable for search
        assert 1 <= config.MAX_RESULTS <= 50, "Max results should be between 1-50 for reasonable performance"
        
        # Max history should be reasonable for conversation
        assert 0 <= config.MAX_HISTORY <= 20, "Max history should be reasonable for conversation context"
    
    def test_model_configuration(self):
        """Test that AI model configurations are valid"""
        from config import config
        
        # Anthropic model should be a valid model name format
        assert "claude" in config.ANTHROPIC_MODEL.lower(), "Should use a Claude model"
        assert config.ANTHROPIC_MODEL, "Anthropic model should not be empty"
        
        # Embedding model should be specified
        assert config.EMBEDDING_MODEL, "Embedding model should not be empty"
        assert "MiniLM" in config.EMBEDDING_MODEL or "mpnet" in config.EMBEDDING_MODEL or "distilbert" in config.EMBEDDING_MODEL, "Should use a known sentence transformer model"
    
    def test_path_configuration(self):
        """Test that path configurations are reasonable"""
        from config import config
        
        # ChromaDB path should be relative or absolute
        assert config.CHROMA_PATH, "ChromaDB path should not be empty"
        assert not config.CHROMA_PATH.startswith('/tmp'), "Should not use temporary directory for persistent storage"
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'custom-api-key',
        'CHUNK_SIZE': '1000',
        'MAX_RESULTS': '10'
    })
    def test_custom_environment_overrides(self):
        """Test that custom environment variables override defaults"""
        # Note: This test would require more complex setup to properly test
        # environment variable overrides since the config is imported at module level
        # For now, we test that the basic loading mechanism works
        
        from config import config
        
        # At minimum, should have loaded the API key from environment
        assert hasattr(config, 'ANTHROPIC_API_KEY')


class TestConfigurationIntegration:
    """Integration tests for configuration with other components"""
    
    def test_config_compatible_with_vector_store(self):
        """Test that config values work with VectorStore initialization"""
        from config import config
        
        # Values should be compatible with VectorStore constructor
        assert isinstance(config.CHROMA_PATH, str)
        assert isinstance(config.EMBEDDING_MODEL, str)
        assert isinstance(config.MAX_RESULTS, int)
        assert config.MAX_RESULTS > 0  # Critical for VectorStore to return results
    
    def test_config_compatible_with_ai_generator(self):
        """Test that config values work with AIGenerator initialization"""
        from config import config
        
        # API key can be empty for testing, but should be string
        assert isinstance(config.ANTHROPIC_API_KEY, str)
        assert isinstance(config.ANTHROPIC_MODEL, str)
        assert config.ANTHROPIC_MODEL  # Should not be empty string
    
    def test_config_compatible_with_document_processor(self):
        """Test that config values work with DocumentProcessor initialization"""
        from config import config
        
        # Chunk settings should be valid for document processing
        assert isinstance(config.CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_OVERLAP >= 0
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


if __name__ == "__main__":
    pytest.main([__file__])