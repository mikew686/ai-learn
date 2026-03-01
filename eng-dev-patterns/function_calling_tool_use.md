# Function Calling / Tool Use

## Overview

Function calling (tool use) lets LLMs invoke external functions, APIs, or tools via structured interfaces. Models can trigger actions, query data, and integrate with external systems instead of only generating text.

## Description

Tools are described with a name, description, and parameters (typically JSON Schema). The model returns a tool call (function name and arguments) when it decides to use a tool; the application executes the function and returns the result in a tool message. The model can then produce a final text response or issue further tool calls. This pattern is native in OpenAI and Anthropic chat APIs.

**Key concepts**

- Function definitions (JSON Schema for name, description, parameters)
- Tool selection by the model
- Parameter extraction from model output
- Function execution by the application
- Response integration (tool result sent back in a follow-up message)

## Translation Example

OpenAI chat with a translation-related tool. The model may call `lookup_language_code` to resolve a language name to an ISO code before or alongside producing a translation; the app runs the tool and returns the result.

```python
from openai import OpenAI
import json

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_language_code",
            "description": "Get the ISO 639-1 language code for a language name. Use when you need the code (e.g. 'fr') for a language (e.g. 'French').",
            "parameters": {
                "type": "object",
                "properties": {
                    "language_name": {"type": "string", "description": "Language name in English, e.g. French, Spanish"},
                },
                "required": ["language_name"],
            },
        },
    }
]

messages = [{"role": "user", "content": "Translate 'Good morning' into French. Use the tool to get the language code if needed."}]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools, tool_choice="auto")
msg = response.choices[0].message

if msg.tool_calls:
    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    # Stub: real implementation would map language_name -> ISO code (e.g. French -> fr)
    result = {"language_name": args["language_name"], "code": "fr"}
    messages.append(msg)
    messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    print(response.choices[0].message.content)
else:
    print(msg.content)
```

The model chooses when to call the tool; the application executes it and appends the tool result so the model can complete the translation or follow-up.
