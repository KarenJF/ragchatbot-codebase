from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Content Search Tool** - For finding specific information within course materials
2. **Course Outline Tool** - For getting complete course structure, lesson lists, and course details

Tool Usage Guidelines:
- **Course outline/structure questions**: Use the course outline tool to get complete course information including course title, instructor, course link, and full lesson list with numbers and titles
- **Content/detail questions**: Use the content search tool for specific educational materials and lesson content
- **Maximum 2 tool calls per query** - You may make additional tool calls after seeing results from the first call
- Use tools sequentially when initial results need refinement or additional context
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course structure questions**: Use outline tool first, then provide complete course details
- **Course content questions**: Use content search tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

For course outline queries, ensure you return:
- Course title
- Course link (if available)
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_sequential_tool_execution(
                response, api_params, tool_manager
            )

        # Return direct response
        return response.content[0].text

    def _handle_sequential_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls with support for sequential rounds (max 2).

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        current_response = initial_response
        current_round = 1
        MAX_ROUNDS = 2

        while current_round <= MAX_ROUNDS:
            # Add AI's response (with tool use) to conversation
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute tools for this round
            tool_results, has_tool_failures = self._execute_single_tool_round(
                current_response, tool_manager
            )

            # If tool execution failed, add results and make final API call
            if has_tool_failures:
                messages.append({"role": "user", "content": tool_results})
                # Make final API call without tools to get error response
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": base_params["system"],
                }
                current_response = self.client.messages.create(**final_params)
                break

            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Prepare API call for next round
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": base_params["system"],
            }

            # Only include tools if we haven't reached max rounds
            if current_round < MAX_ROUNDS:
                api_params["tools"] = base_params.get("tools", [])
                api_params["tool_choice"] = {"type": "auto"}

            # Get next response from Claude
            current_response = self.client.messages.create(**api_params)

            # Check if Claude wants to use more tools
            has_tool_use = any(
                block.type == "tool_use" for block in current_response.content
            )

            # Terminate if no tool use or max rounds reached
            if not has_tool_use or current_round >= MAX_ROUNDS:
                break

            current_round += 1

        # Return the final response text
        return current_response.content[0].text

    def _execute_single_tool_round(self, response, tool_manager):
        """
        Execute all tool calls from a single response round.

        Args:
            response: The AI response containing tool use requests
            tool_manager: Manager to execute tools

        Returns:
            Tuple of (tool_results_list, has_failures_bool)
        """
        tool_results = []
        has_failures = False

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    # Check if result indicates a failure
                    if isinstance(tool_result, str) and "failed" in tool_result.lower():
                        has_failures = True

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                except Exception as e:
                    # Handle tool execution errors gracefully
                    has_failures = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution failed: {str(e)}",
                        }
                    )

        return tool_results, has_failures
