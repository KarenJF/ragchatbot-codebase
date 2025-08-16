"""
Tests for AIGenerator tool calling functionality
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add the backend directory to the Python path  
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from ai_generator import AIGenerator


class MockAnthropicContent:
    """Mock for Anthropic response content block"""
    def __init__(self, text=None, tool_use_data=None):
        if tool_use_data:
            self.type = "tool_use"
            self.name = tool_use_data["name"]
            self.input = tool_use_data["input"]
            self.id = tool_use_data.get("id", "tool_use_123")
        else:
            self.type = "text"
            self.text = text or "Default response"


class MockAnthropicResponse:
    """Mock for Anthropic API response"""
    def __init__(self, content=None, stop_reason="end_turn"):
        if content is None:
            content = [MockAnthropicContent("Default response")]
        elif not isinstance(content, list):
            content = [content]
        self.content = content
        self.stop_reason = stop_reason


class TestAIGenerator:
    """Test cases for AIGenerator tool calling functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_tool_manager = Mock()
        
        # Mock tools definition
        self.mock_tools = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "course_name": {"type": "string"},
                        "lesson_number": {"type": "integer"}
                    },
                    "required": ["query"]
                }
            }
        ]
    
    def _create_ai_generator(self):
        """Helper to create AIGenerator instance after mock is set up"""
        return AIGenerator("test-api-key", "claude-sonnet-4-20250514")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test basic response generation without tools"""
        ai_generator = self._create_ai_generator()
        
        # Mock response without tool use
        mock_response = MockAnthropicResponse(
            MockAnthropicContent("This is a direct response")
        )
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        result = ai_generator.generate_response("What is Python?")
        
        # Verify API was called correctly
        mock_anthropic.return_value.messages.create.assert_called_once()
        call_args = mock_anthropic.return_value.messages.create.call_args
        
        # Check that tools were not included
        assert "tools" not in call_args[1]
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
        
        assert result == "This is a direct response"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response generation with tools available but not used"""
        ai_generator = self._create_ai_generator()
        
        mock_response = MockAnthropicResponse(
            MockAnthropicContent("Direct answer without using tools")
        )
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        result = ai_generator.generate_response(
            "What is 2+2?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify API was called with tools
        call_args = mock_anthropic.return_value.messages.create.call_args
        assert call_args[1]["tools"] == self.mock_tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
        
        assert result == "Direct answer without using tools"
        
        # Tool manager should not be called
        self.mock_tool_manager.execute_tool.assert_not_called()
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_single_tool_use(self, mock_anthropic):
        """Test response generation when AI uses tools (single round)"""
        ai_generator = self._create_ai_generator()
        
        # First response: AI requests tool use
        tool_use_response = MockAnthropicResponse(
            content=[MockAnthropicContent(
                tool_use_data={
                    "name": "search_course_content", 
                    "input": {"query": "Python basics"},
                    "id": "tool_123"
                }
            )],
            stop_reason="tool_use"
        )
        
        # Second response: AI processes tool results (no more tools)
        final_response = MockAnthropicResponse(
            MockAnthropicContent("Based on the search results, Python is a programming language...")
        )
        
        mock_anthropic.return_value.messages.create.side_effect = [
            tool_use_response,
            final_response
        ]
        
        # Mock tool execution result
        self.mock_tool_manager.execute_tool.return_value = "Python course content found"
        
        result = ai_generator.generate_response(
            "Tell me about Python",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify tool was executed correctly
        self.mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )
        
        # Verify second API call was made with tool results
        assert mock_anthropic.return_value.messages.create.call_count == 2
        
        # Check the second call includes tool results and tools for potential second round
        second_call_args = mock_anthropic.return_value.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        
        # Should have: original query, AI tool use, tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Tool results should be in the last message
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Python course content found"
        
        # Second call should include tools for potential second round
        assert "tools" in second_call_args[1]
        assert second_call_args[1]["tools"] == self.mock_tools
        
        assert result == "Based on the search results, Python is a programming language..."
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation includes conversation history"""
        ai_generator = self._create_ai_generator()
        
        mock_response = MockAnthropicResponse(
            MockAnthropicContent("Response with history context")
        )
        mock_anthropic.return_value.messages.create.return_value = mock_response
        
        history = "User: Previous question\nAI: Previous answer"
        
        result = ai_generator.generate_response(
            "Follow-up question",
            conversation_history=history
        )
        
        # Verify system prompt includes history
        call_args = mock_anthropic.return_value.messages.create.call_args
        system_content = call_args[1]["system"]
        
        assert "Previous conversation:" in system_content
        assert history in system_content
        assert ai_generator.SYSTEM_PROMPT in system_content
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_multiple_tool_calls(self, mock_anthropic):
        """Test handling of multiple tool calls in one response"""
        ai_generator = self._create_ai_generator()
        
        # AI requests multiple tool uses
        tool_use_response = MockAnthropicResponse(
            content=[
                MockAnthropicContent(tool_use_data={
                    "name": "search_course_content",
                    "input": {"query": "Python"},
                    "id": "tool_1"
                }),
                MockAnthropicContent(tool_use_data={
                    "name": "search_course_content", 
                    "input": {"query": "JavaScript"},
                    "id": "tool_2"
                })
            ],
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            MockAnthropicContent("Compared Python and JavaScript...")
        )
        
        mock_anthropic.return_value.messages.create.side_effect = [
            tool_use_response,
            final_response
        ]
        
        # Mock tool execution results
        self.mock_tool_manager.execute_tool.side_effect = [
            "Python content",
            "JavaScript content"
        ]
        
        result = ai_generator.generate_response(
            "Compare Python and JavaScript",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify both tools were executed
        assert self.mock_tool_manager.execute_tool.call_count == 2
        self.mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python")
        self.mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="JavaScript")
        
        # Verify tool results were included in second API call
        second_call_args = mock_anthropic.return_value.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_results = messages[2]["content"]
        
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1"
        assert tool_results[0]["content"] == "Python content"
        assert tool_results[1]["tool_use_id"] == "tool_2"
        assert tool_results[1]["content"] == "JavaScript content"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_tool_execution_error(self, mock_anthropic):
        """Test handling when tool execution fails"""
        ai_generator = self._create_ai_generator()
        
        tool_use_response = MockAnthropicResponse(
            content=[MockAnthropicContent(tool_use_data={
                "name": "search_course_content",
                "input": {"query": "test"},
                "id": "tool_123"
            })],
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            MockAnthropicContent("I encountered an error searching...")
        )
        
        mock_anthropic.return_value.messages.create.side_effect = [
            tool_use_response,
            final_response
        ]
        
        # Mock tool execution failure
        self.mock_tool_manager.execute_tool.return_value = "Tool execution failed: Database error"
        
        result = ai_generator.generate_response(
            "Search for content",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify tool was still called
        self.mock_tool_manager.execute_tool.assert_called_once()
        
        # Verify error message was passed to AI
        second_call_args = mock_anthropic.return_value.messages.create.call_args_list[1]
        tool_results = second_call_args[1]["messages"][2]["content"]
        assert tool_results[0]["content"] == "Tool execution failed: Database error"
        
        assert result == "I encountered an error searching..."
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_sequential_tool_use(self, mock_anthropic):
        """Test response generation with sequential tool calls (2 rounds)"""
        ai_generator = self._create_ai_generator()
        
        # First response: AI requests first tool use
        first_tool_response = MockAnthropicResponse(
            content=[MockAnthropicContent(
                tool_use_data={
                    "name": "get_course_outline",
                    "input": {"course_name": "Python Fundamentals"},
                    "id": "tool_1"
                }
            )],
            stop_reason="tool_use"
        )
        
        # Second response: AI requests second tool use after seeing first results
        second_tool_response = MockAnthropicResponse(
            content=[MockAnthropicContent(
                tool_use_data={
                    "name": "search_course_content",
                    "input": {"query": "variables and data types"},
                    "id": "tool_2"
                }
            )],
            stop_reason="tool_use"
        )
        
        # Third response: AI provides final answer
        final_response = MockAnthropicResponse(
            MockAnthropicContent("Based on the course outline and content search, lesson 2 covers variables...")
        )
        
        mock_anthropic.return_value.messages.create.side_effect = [
            first_tool_response,
            second_tool_response, 
            final_response
        ]
        
        # Mock tool execution results
        self.mock_tool_manager.execute_tool.side_effect = [
            "Course outline: Lesson 1: Intro, Lesson 2: Variables, Lesson 3: Functions",
            "Variables and data types content found"
        ]
        
        result = ai_generator.generate_response(
            "What does lesson 2 of Python Fundamentals cover?",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Verify both tools were executed in sequence
        assert self.mock_tool_manager.execute_tool.call_count == 2
        self.mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Python Fundamentals")
        self.mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="variables and data types")
        
        # Verify 3 API calls were made (initial, after first tool, after second tool)
        assert mock_anthropic.return_value.messages.create.call_count == 3
        
        # Check that tools were included in first two calls but not the third
        call_args_list = mock_anthropic.return_value.messages.create.call_args_list
        
        # First call (initial) should have tools
        assert "tools" in call_args_list[0][1]
        
        # Second call (after first tool) should have tools for potential second round
        assert "tools" in call_args_list[1][1]
        
        # Third call (after second tool) should NOT have tools (max rounds reached)
        assert "tools" not in call_args_list[2][1]
        
        assert result == "Based on the course outline and content search, lesson 2 covers variables..."
    
    @patch('ai_generator.anthropic.Anthropic') 
    def test_generate_response_max_rounds_termination(self, mock_anthropic):
        """Test that sequential tool calling terminates after max rounds (2)"""
        ai_generator = self._create_ai_generator()
        
        # AI tries to make tools calls in both rounds
        tool_response_1 = MockAnthropicResponse(
            content=[MockAnthropicContent(
                tool_use_data={"name": "search_course_content", "input": {"query": "test1"}, "id": "tool_1"}
            )],
            stop_reason="tool_use"
        )
        
        tool_response_2 = MockAnthropicResponse(
            content=[MockAnthropicContent(
                tool_use_data={"name": "search_course_content", "input": {"query": "test2"}, "id": "tool_2"}
            )],
            stop_reason="tool_use"
        )
        
        # This would be a third tool call attempt, but should not happen
        final_response = MockAnthropicResponse(
            MockAnthropicContent("Final answer after max rounds")
        )
        
        mock_anthropic.return_value.messages.create.side_effect = [
            tool_response_1,
            tool_response_2,
            final_response
        ]
        
        self.mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        result = ai_generator.generate_response(
            "Multi-step query", 
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Should execute exactly 2 tools (max rounds)
        assert self.mock_tool_manager.execute_tool.call_count == 2
        
        # Should make exactly 3 API calls (initial + 2 rounds)
        assert mock_anthropic.return_value.messages.create.call_count == 3
        
        # Third call should not include tools
        third_call_args = mock_anthropic.return_value.messages.create.call_args_list[2]
        assert "tools" not in third_call_args[1]
        
        assert result == "Final answer after max rounds"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_tool_failure_termination(self, mock_anthropic):
        """Test that sequential tool calling terminates on tool execution failure"""
        ai_generator = self._create_ai_generator()
        
        tool_use_response = MockAnthropicResponse(
            content=[MockAnthropicContent(
                tool_use_data={
                    "name": "search_course_content",
                    "input": {"query": "test"},
                    "id": "tool_123"
                }
            )],
            stop_reason="tool_use"
        )
        
        final_response = MockAnthropicResponse(
            MockAnthropicContent("I encountered an error with the search...")
        )
        
        mock_anthropic.return_value.messages.create.side_effect = [
            tool_use_response,
            final_response
        ]
        
        # Mock tool execution failure
        self.mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")
        
        result = ai_generator.generate_response(
            "Search for content",
            tools=self.mock_tools,
            tool_manager=self.mock_tool_manager
        )
        
        # Tool should be called once, then fail
        assert self.mock_tool_manager.execute_tool.call_count == 1
        
        # Should make exactly 2 API calls (initial + failure response)
        assert mock_anthropic.return_value.messages.create.call_count == 2
        
        # Verify error was passed to AI
        second_call_args = mock_anthropic.return_value.messages.create.call_args_list[1]
        tool_results = second_call_args[1]["messages"][2]["content"]
        assert "Tool execution failed" in tool_results[0]["content"]
        
        assert result == "I encountered an error with the search..."
    
    def test_system_prompt_structure(self):
        """Test that system prompt contains expected guidance"""
        ai_generator = self._create_ai_generator()
        system_prompt = ai_generator.SYSTEM_PROMPT
        
        # Verify key guidance is present
        assert "course materials" in system_prompt.lower()
        assert "search tools" in system_prompt.lower()
        assert "content search tool" in system_prompt.lower()
        assert "course outline tool" in system_prompt.lower()
        
        # Verify sequential tool usage guidance
        assert "maximum 2 tool calls per query" in system_prompt.lower()
        assert "sequentially" in system_prompt.lower()
        assert "additional tool calls" in system_prompt.lower()
        
        # Verify response protocols
        assert "brief" in system_prompt.lower()
        assert "concise" in system_prompt.lower()
        assert "educational" in system_prompt.lower()
    
    def test_base_params_configuration(self):
        """Test that base API parameters are configured correctly"""
        ai_generator = self._create_ai_generator()
        base_params = ai_generator.base_params
        
        assert base_params["model"] == "claude-sonnet-4-20250514"
        assert base_params["temperature"] == 0
        assert base_params["max_tokens"] == 800


if __name__ == "__main__":
    pytest.main([__file__])