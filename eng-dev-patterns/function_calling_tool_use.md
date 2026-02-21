# Function Calling / Tool Use Pattern

## Overview

**Function Calling / Tool Use** is an AI engineering technique that enables LLMs to call external functions, APIs, or tools through structured interfaces. This allows models to interact with the real world beyond text generation, making them capable of executing actions, querying databases, calling APIs, and performing computations.

## Description

From the AI Technologies reference guide:

> **Description**: Enabling LLMs to call external functions, APIs, or tools through structured interfaces. Allows models to interact with the real world beyond text generation.

**Key Concepts**:
- Function definitions (JSON Schema)
- Tool selection by LLM
- Parameter extraction
- Function execution
- Response integration

## How It Works

### 1. Define Tools (Functions)

Tools are defined in JSON Schema format that describes the function name, description, and parameters:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### 2. Make API Call with Tools

Include `tools` in the chat completion request using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto"  # or "none" or specific tool
)
```

### 3. Handle Tool Calls

The model may return tool calls instead of (or in addition to) text. Using the OpenAI SDK:

```python
import json
from openai import OpenAI

client = OpenAI()

message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute the function
        if function_name == "get_weather":
            result = get_weather(**arguments)
            
            # Send results back to the model
            messages.append(message)  # Add assistant's message with tool calls
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
            
            # Get final response
            second_response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
```

## Tool Use Patterns

There are three main patterns for how models can use tools: sequential, parallel, and interleaved. Understanding these patterns helps you design efficient tool workflows.

### Sequential Tool Use

**Description**: The model makes one tool call, receives the result, and then generates a final response. This is the most common pattern and works with all models that support function calling.

**Characteristics**:
- **API Calls**: 2 total (initial with tool call, final response)
- **Flow**: User request → Tool call → Tool result → Final response
- **Use Case**: Standard pattern for most applications
- **Model Support**: All models with function calling support

**Example**:

```python
# Step 1: Initial call with tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Translate 'Hello' to French"}],
    tools=[analyze_language_tool],
    tool_choice="auto"
)

message = response.choices[0].message
messages.append(message)

# Step 2: Handle tool call
if message.tool_calls:
    for tool_call in message.tool_calls:
        # Execute tool
        result = analyze_language(tool_call.function.arguments)
        
        # Add tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
        
        # Step 3: Get final response
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
```

**When to Use**:
- Simple workflows with one tool call needed
- When tool results are required before generating final output
- Most common use case for function calling

### Parallel Tool Use

**Description**: The model can call multiple tools simultaneously in a single response. All tool calls are executed, results are collected, and then sent back together for the final response.

**Characteristics**:
- **API Calls**: 2 total (initial with multiple tool calls, final response)
- **Flow**: User request → Multiple tool calls (parallel) → All tool results → Final response
- **Use Case**: Efficient for independent operations that don't depend on each other
- **Model Support**: All models with function calling support

**Example**:

```python
# Step 1: Initial call with multiple tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Translate with cultural context"}],
    tools=[analyze_language_tool, get_cultural_context_tool],
    tool_choice="auto"
)

message = response.choices[0].message
messages.append(message)

# Step 2: Handle all tool calls (model may call multiple in one response)
tool_results = {}
if message.tool_calls:
    for tool_call in message.tool_calls:
        # Execute each tool
        if tool_call.function.name == "analyze_language":
            result = analyze_language(**json.loads(tool_call.function.arguments))
        elif tool_call.function.name == "get_cultural_context":
            result = get_cultural_context(**json.loads(tool_call.function.arguments))
        
        tool_results[tool_call.id] = result
        
        # Add all tool results
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # Step 3: Get final response with all tool results
    final_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
```

**When to Use**:
- Multiple independent operations needed
- Gathering information from different sources simultaneously
- Optimizing for fewer API calls when tools don't depend on each other

**Note**: While the model can call multiple tools in one response, your code executes them sequentially. For I/O-bound tools, consider using `concurrent.futures` for true parallel execution.

### Interleaved Tool Use

**Description**: Reasoning models can make multiple tool calls in sequence, reasoning between each call. The model can iteratively refine its approach based on tool results, making additional tool calls as needed.

**Characteristics**:
- **API Calls**: N+1 total (N tool-call iterations + 1 final response, where N >= 2)
- **Flow**: User request → Tool call 1 → Result 1 → Reasoning → Tool call 2 → Result 2 → ... → Final response
- **Use Case**: Complex multi-step reasoning workflows requiring iterative refinement
- **Model Support**: Reasoning models (o1-preview, o1-mini, o3-mini, o4-mini)

**Example**:

```python
messages = [
    {"role": "user", "content": "Translate with detailed analysis"}
]

max_iterations = 5
iteration = 0

# Interleaved loop: model can make multiple tool calls in sequence
while iteration < max_iterations:
    iteration += 1
    
    # Make API call (may return tool calls or final response)
    response = client.chat.completions.create(
        model="o3-mini",
        messages=messages,
        tools=[analyze_language_tool, get_translation_suggestions_tool],
        tool_choice="auto"
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    # If no tool calls, we have the final response
    if not message.tool_calls and message.content:
        break
    
    # Handle tool calls for this iteration
    for tool_call in message.tool_calls:
        # Execute tool
        result = execute_tool(tool_call)
        
        # Add tool result for next iteration
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
    
    # Model will reason about results and potentially make another tool call
```

**When to Use**:
- Complex problem-solving requiring multiple reasoning steps
- Iterative refinement based on intermediate results
- Multi-step workflows where each step informs the next
- With reasoning models (o-series) that support extended reasoning

**Best Practices**:
- Set `max_iterations` to prevent infinite loops
- Monitor token usage (interleaved patterns use more tokens)
- Use for complex tasks that benefit from step-by-step reasoning
- Consider cost implications (more API calls = higher cost)

### Pattern Comparison

| Pattern | API Calls | Model Support | Best For | Cost Efficiency |
|---------|-----------|---------------|----------|-----------------|
| **Sequential** | 2 | All models | Simple workflows | High |
| **Parallel** | 2 | All models | Independent operations | High |
| **Interleaved** | N+1 (N≥2) | Reasoning models | Complex reasoning | Lower (more calls) |

### Choosing the Right Pattern

1. **Start with Sequential**: Most use cases only need one tool call
2. **Use Parallel**: When you need multiple independent pieces of information
3. **Use Interleaved**: Only for complex reasoning tasks with reasoning models

**Example Implementation**: See `src/scripts/examples/tool_use_patterns.py` for complete working examples of all three patterns with translation use cases.

## Tool Choice Options

The `tool_choice` parameter controls how the model uses tools:

- **`"auto"`** (default): Model decides whether to call tools based on the conversation
- **`"none"`**: Model cannot call tools, only generates text
- **`{"type": "function", "function": {"name": "get_weather"}}`**: Force the model to call a specific tool

## Response Structure

When the model calls a tool, the response includes:

```python
message.tool_calls = [
    ToolCall(
        id="call_abc123",
        type="function",
        function=Function(
            name="get_weather",
            arguments='{"location": "San Francisco, CA", "unit": "fahrenheit"}'
        )
    )
]
```

Key components:
- **`id`**: Unique identifier for the tool call
- **`type`**: Always `"function"` for function calls
- **`function.name`**: The function name to call
- **`function.arguments`**: JSON string containing the function arguments

## Relationship to Agentic Systems

When Function Calling / Tool Use is combined with planning, memory, and goal tracking, it becomes part of **Agentic Systems**:

> **Description**: Autonomous systems that can plan, execute actions, and adapt based on results. Agents use tools, memory, and reasoning to achieve goals.

**Key Components**:
- Planning capabilities
- Tool/action execution
- Memory/state management
- Goal tracking
- Reflection and adaptation

In practice, this creates a **tool-using agent** pattern where:
- The agent uses function calling to interact with external systems
- Multiple tool calls are orchestrated to achieve complex goals
- The agent can adapt its approach based on tool execution results

## Popular Solutions

- **AWS Bedrock Agents**: Managed agents with native tool/function calling support
- **AWS Lambda**: Serverless functions for tool execution
- **OpenAI Function Calling**: Native support in GPT models
- **Anthropic Tool Use**: Claude's tool calling API
- **LangChain Tools**: Tool abstraction framework
- **AutoGPT**: Agent framework with tool use
- **CrewAI**: Multi-agent framework with tools

## Best Practices

- Define clear, specific function schemas
- Validate inputs before execution
- Handle errors gracefully
- Use tool descriptions effectively
- Implement retry logic
- Monitor tool usage and execution
- Implement safety limits (max tool calls, timeouts)
- Choose the right tool use pattern:
  - Use sequential for simple workflows
  - Use parallel for independent operations
  - Use interleaved only with reasoning models for complex tasks
- Set `max_iterations` for interleaved patterns to prevent infinite loops
- Consider cost implications: interleaved patterns use more API calls

## Use Cases

- API integrations
- Database queries
- File operations
- Web scraping
- Calculator tools
- Code execution
- Document manipulation (e.g., Google Slides, PowerPoint)
- Multi-step task automation

## Related patterns

- **Schema-Driven Inference**: Tool schema descriptions act as implicit prompts; see [schema_driven_inference.md](./schema_driven_inference.md).
- **Structured Output**: Validate tool arguments or return tool results as structured data; see [structured_output.md](./structured_output.md).
- **Agentic Systems**: Function calling is the execution layer for agents; see [README.md](./README.md) and [learning_progression.md](./learning_progression.md) Pattern 14.
- **Prompt Engineering**: System and user prompts guide when the model should use tools; see [prompt_engineering.md](./prompt_engineering.md).

## Learning path

Function Calling is Pattern 4 in the [learning progression](./learning_progression.md). Learn it after **Understanding Models**, **Prompt Engineering**, and **Structured Output**; then **Schema-Driven Inference** (Pattern 5) and **Embeddings** (Pattern 6). Example: [tool_use_patterns.py](../src/scripts/examples/tool_use_patterns.py).

## Documentation Links

### Official Documentation

1. **OpenAI Function Calling Guide**
   - URL: https://platform.openai.com/docs/guides/function-calling
   - Description: Comprehensive guide covering function calling concepts, examples, and best practices

2. **OpenAI API Reference - Chat Completions**
   - URL: https://platform.openai.com/docs/api-reference/chat/create
   - Description: Complete API reference for `tools`, `tool_choice`, and response format

3. **OpenAI Cookbook - Function Calling Examples**
   - URL: https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models
   - Description: Practical code examples and patterns for implementing function calling

4. **JSON Schema Specification**
   - URL: https://json-schema.org/
   - Description: Tool parameters use JSON Schema format for validation

5. **OpenAI Python SDK Documentation**
   - URL: https://github.com/openai/openai-python
   - Description: Python SDK documentation for tool calling implementation

### Related pattern docs

- **Agentic Systems**: [README.md](./README.md); when tool use is combined with planning and memory.
- **Structured Output**: [structured_output.md](./structured_output.md); often used with function calling for validated responses.
- **Orchestration**: [README.md](./README.md); coordinating multiple tool calls in workflows.

## Key Points

1. **Tools are defined as JSON Schema objects** - Standard format for describing function interfaces
2. **The model returns tool calls, not direct execution** - Your code must execute the functions
3. **Your code executes the functions and returns results** - Bridge between LLM and external systems
4. **You can continue the conversation with tool results** - Multi-turn interactions with tool feedback
5. **Three tool use patterns available**:
   - **Sequential**: One tool call → final response (2 API calls, all models)
   - **Parallel**: Multiple tools in one response → final response (2 API calls, all models)
   - **Interleaved**: Multiple tool calls in sequence with reasoning (N+1 API calls, reasoning models)
6. **Supports multiple tool calls in one response** - Model can call multiple functions simultaneously
7. **Works with models that support function calling** - GPT-4, Claude, and other modern models
8. **Interleaved pattern requires reasoning models** - o1-preview, o1-mini, o3-mini, o4-mini for iterative tool use

## Example: Complete Workflow with OpenAI SDK

Here's a complete example using the OpenAI SDK to handle function calling:

```python
from openai import OpenAI
import json

client = OpenAI()

# Define tool for updating database records
update_records_tool = {
    "type": "function",
    "function": {
        "name": "updateRecords",
        "description": "Updates one or more database records with new values",
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "recordId": {"type": "string"},
                            "fields": {"type": "object"}
                        },
                        "required": ["table", "recordId", "fields"]
                    }
                }
            },
            "required": ["updates"]
        }
    }
}

# Initialize conversation
messages = [
    {"role": "user", "content": "Update user record with id 'user123' to set status to 'active'"}
]

# Make initial API call with tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[update_records_tool],
    tool_choice="auto"
)

# Handle tool calls
message = response.choices[0].message
messages.append(message)  # Add assistant's message with tool calls

if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute the function (in real app, this would call database API)
        if function_name == "updateRecords":
            result = {"status": "success", "updated": len(arguments["updates"])}
            
            # Send results back to the model
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
            
            # Get final response
            final_response = client.chat.completions.create(
                model="gpt-4",
                messages=messages
            )
            print(final_response.choices[0].message.content)
```

This enables the agent to:
- Generate structured commands for database operations
- Execute multiple operations in a single call
- Maintain type safety through JSON Schema validation
- Integrate seamlessly with external APIs

## Implementation Considerations

### Using OpenAI SDK

When implementing function calling with the OpenAI SDK:

1. **Initialize the client** - Create an `OpenAI()` client instance
2. **Define tools** - Use JSON Schema format for function definitions
3. **Include tools in requests** - Add `tools` parameter to `chat.completions.create()`
4. **Handle tool_choice parameter** - Support auto/none/specific tool selection
5. **Extract tool calls from response** - Parse `message.tool_calls` from response
6. **Execute functions** - Call your actual functions with extracted arguments
7. **Return results** - Add tool results to messages with `role="tool"`
8. **Continue conversation** - Make follow-up API calls with updated messages
9. **Error handling** - Handle invalid tools, parsing failures, and execution errors

This pattern enables the LLM to interact with external systems, APIs, and databases through structured function calls, making it a fundamental building block for agentic AI systems.

---

*Reference: AI Technologies Guide - Section 3 (Function Calling / Tool Use) and Section 6 (Agentic Systems)*
