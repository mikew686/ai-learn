# Schema-Driven Inference Pattern

## Overview

**Schema-Driven Inference** is an AI engineering technique that uses structured definitions (tool schemas, Pydantic field descriptions, JSON Schema) as implicit prompts, allowing the model to infer behavior and requirements without verbose explicit instructions. This pattern reduces prompt verbosity while maintaining high-quality, validated outputs.

## Description

Schema-Driven Inference leverages the fact that modern LLMs can extract meaning and requirements from structured metadata, not just explicit text instructions. By carefully crafting tool descriptions, Pydantic field descriptions, and schema definitions, you can communicate complex requirements to the model implicitly, reducing token usage and prompt complexity.

**Key Concepts**:
- Tool schema descriptions guide model behavior
- Pydantic field descriptions specify data requirements
- Structured definitions serve as implicit prompts
- Model infers details from schema metadata
- Reduced prompt complexity with maintained quality

## How It Works

### 1. Tool Schema Descriptions

Tool function descriptions in JSON Schema format provide implicit instructions about what tools do and when to use them:

```python
from openai import OpenAI
import json

client = OpenAI()

lookup_language_code_tool = {
    "type": "function",
    "function": {
        "name": "lookup_language_code",
        "description": "Look up ISO 639-1 language code for a language name. Use this when you need to convert a language name (e.g., 'French', 'Spanish') to its standard code (e.g., 'fr', 'es').",
        "parameters": {
            "type": "object",
            "properties": {
                "language_name": {
                    "type": "string",
                    "description": "Name of the language (e.g., 'Spanish', 'French')"
                }
            },
            "required": ["language_name"],
        },
    },
}

# The model infers from the description that it should use this tool
# to convert language names to codes, without explicit instructions
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Translate 'Hello' to French"}
    ],
    tools=[lookup_language_code_tool],
    tool_choice="auto"
)
```

### 2. Pydantic Field Descriptions

Pydantic model field descriptions guide the model on what data to extract and how to format it:

```python
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

class TranslationResult(BaseModel):
    """Structured translation result."""
    
    source_language: str = Field(
        description="Source language name in plain English (e.g., 'English', 'French', 'Spanish')"
    )
    source_language_code: str = Field(
        description="Source language code (ISO 639-1, e.g., 'en', 'fr', 'es')"
    )
    target_language: str = Field(
        description="Target language name in plain English (e.g., 'English', 'French', 'Spanish')"
    )
    target_language_code: str = Field(
        description="Target language code (ISO 639-1, e.g., 'en', 'fr', 'es')"
    )
    translated_text: str = Field(description="The translated text")
    confidence: str = Field(description="Translation confidence: low, medium, high")
    cultural_notes: str = Field(
        description="Cultural or contextual notes about the translation (always in English)"
    )

# The model infers from field descriptions:
# - What data to extract (source/target languages, translation, confidence)
# - Format requirements (plain English names, ISO codes)
# - Constraints (cultural notes must be in English)
# - All without verbose prompt instructions

response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Translate 'Hello' to French"}
    ],
    response_format=TranslationResult
)
```

### 3. Combined Pattern: Tools + Structured Output

When combining tool calling with structured output, the model infers behavior from both:

```python
from pydantic import BaseModel, Field
from openai import OpenAI
import json

client = OpenAI()

# Tool schema with descriptive function description
get_cultural_context_tool = {
    "type": "function",
    "function": {
        "name": "get_cultural_context",
        "description": "Get cultural context information for a phrase in a language. Use this to understand cultural nuances, formality levels, and regional variations that may affect translation accuracy.",
        "parameters": {
            "type": "object",
            "properties": {
                "language_code": {
                    "type": "string",
                    "description": "ISO 639-1 language code"
                },
                "phrase": {
                    "type": "string",
                    "description": "The phrase to get context for"
                },
            },
            "required": ["language_code", "phrase"],
        },
    },
}

# Pydantic model with detailed field descriptions
class TranslationResult(BaseModel):
    source_language: str = Field(
        description="Source language name in plain English"
    )
    translated_text: str = Field(description="The translated text")
    cultural_notes: str = Field(
        description="Cultural or contextual notes about the translation (always in English)"
    )

# Minimal prompt - model infers:
# 1. From tool description: Should use get_cultural_context to understand nuances
# 2. From field descriptions: Should extract source language, translation, and cultural notes
# 3. From cultural_notes field: Notes must be in English

messages = [
    {
        "role": "user",
        "content": (
            "Translate the following text to French and provide a structured translation result. "
            "Text to translate: Let's go hang out, what do you say to that?"
        ),
    }
]

# Initial call with tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[get_cultural_context_tool],
    tool_choice="auto",
)

message = response.choices[0].message
messages.append(message)

# Handle tool calls
if message.tool_calls:
    for tool_call in message.tool_calls:
        # Execute tool and add result
        # ... tool execution code ...
        pass

# Final call with structured output
final_response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=messages,
    response_format=TranslationResult,
)
```

## Key Techniques

### 1. Descriptive Tool Function Descriptions

Write tool descriptions that explain not just what the tool does, but when and why to use it:

```python
# Good: Descriptive and contextual
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a location. "
            "Use this when the user asks about weather, temperature, or conditions. "
            "Returns temperature in the requested unit (Celsius or Fahrenheit), "
            "humidity, wind speed, and conditions (sunny, rainy, etc.)."
        ),
        # ... parameters ...
    }
}

# Less effective: Too brief
weather_tool_bad = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Gets weather",  # Too vague
        # ... parameters ...
    }
}
```

### 2. Detailed Pydantic Field Descriptions

Use Field descriptions to specify format, constraints, and examples:

```python
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: str = Field(
        description="Full name of the user, including first and last name"
    )
    email: str = Field(
        description="Email address in standard format (e.g., user@example.com)"
    )
    age: int = Field(
        description="Age in years, must be between 0 and 150",
        ge=0,
        le=150
    )
    role: str = Field(
        description="User role: 'admin', 'user', or 'guest'"
    )
    preferences: dict = Field(
        description="User preferences as a dictionary with string keys and values"
    )
```

### 3. Minimal Prompts with Schema Context

Let schemas do the heavy lifting:

```python
# Verbose approach (not using schema-driven inference)
prompt = """Extract user information from the text. You must:
1. Extract the full name (first and last)
2. Extract the email address in standard format
3. Extract the age as a number between 0 and 150
4. Determine the role: 'admin', 'user', or 'guest'
5. Extract preferences as a dictionary

Text: {text}"""

# Schema-driven approach (minimal prompt)
prompt = "Extract user information from: {text}"

# The Pydantic model with Field descriptions provides all the detail
response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(text=text)}],
    response_format=UserProfile
)
```

### 4. Schema Composition and Structured Types

Schemas can compose other schemas and use structured types (enums, arrays) to provide even more implicit guidance:

#### Using Enums for Constrained Values

Instead of free-form strings, use enums to constrain values and communicate valid options:

```python
from pydantic import BaseModel, Field
from enum import Enum

# Define language codes as enum instead of free-form string
class LanguageCode(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class TranslationResult(BaseModel):
    source_language: str = Field(description="Source language name in plain English")
    source_language_code: LanguageCode = Field(
        description="Source language code (ISO 639-1)"
    )
    target_language: str = Field(description="Target language name in plain English")
    target_language_code: LanguageCode = Field(
        description="Target language code (ISO 639-1)"
    )
    translated_text: str = Field(description="The translated text")
    confidence: ConfidenceLevel = Field(description="Translation confidence level")
    cultural_notes: str = Field(
        description="Cultural or contextual notes about the translation (always in English)"
    )

# The model infers:
# - Language codes must be one of the enum values (en, es, fr, etc.)
# - Confidence must be one of: low, medium, high
# - No need to specify valid values in the prompt
```

#### Using Arrays for Lists

Use typed arrays to specify list structures:

```python
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    tags: List[str] = Field(
        description="List of product tags or categories"
    )

class ShoppingCart(BaseModel):
    items: List[Product] = Field(
        description="List of products in the shopping cart"
    )
    total: float = Field(description="Total price of all items")

# The model infers:
# - items is an array of Product objects
# - Each Product has name, price, and tags
# - tags is an array of strings
```

#### Schema Composition with Nested Models

Compose complex schemas from simpler ones:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Address(BaseModel):
    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str = Field(description="State or province code")
    zip_code: str = Field(description="ZIP or postal code")
    country: str = Field(description="Country name")

class Contact(BaseModel):
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(
        description="Phone number in international format", 
        default=None
    )

class Company(BaseModel):
    name: str = Field(description="Company name")
    address: Address = Field(description="Company address")
    contacts: List[Contact] = Field(
        description="List of contact methods"
    )
    employee_count: int = Field(
        description="Number of employees",
        ge=0
    )

# The model infers the entire structure:
# - Company has an Address (with street, city, state, zip, country)
# - Company has a list of Contact objects (each with email and optional phone)
# - No need to explain the nested structure in the prompt
```

#### Combining Enums, Arrays, and Composition

Combine all techniques for complex schemas:

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class Status(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"

class Task(BaseModel):
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    priority: Priority = Field(description="Task priority level")
    status: Status = Field(description="Current task status")
    tags: List[str] = Field(description="List of tags for categorization")
    assignee: Optional[str] = Field(
        description="Person assigned to the task, or None if unassigned",
        default=None
    )

class Project(BaseModel):
    name: str = Field(description="Project name")
    tasks: List[Task] = Field(description="List of tasks in the project")
    priority: Priority = Field(description="Overall project priority")

# The model infers:
# - priority and status must be enum values
# - tasks is an array of Task objects
# - Each Task has its own priority, status, tags (array), and optional assignee
# - Complex nested structure without verbose prompt instructions
```

#### Tool Parameters with Enums and Arrays

Apply the same principles to tool schemas:

```python
from enum import Enum

# Define enum for tool parameters
class TemperatureUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather conditions for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit", "kelvin"],
                    "description": "Temperature unit"
                },
                "forecast_days": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of days to include in forecast (0=today, 1=tomorrow, etc.)"
                }
            },
            "required": ["location", "unit"]
        }
    }
}

# The model infers:
# - unit must be one of the enum values
# - forecast_days is an array of integers
# - No need to specify valid values in the prompt
```

**Benefits of Schema Composition:**
- **Type Safety**: Enums and typed arrays provide clear constraints
- **Implicit Validation**: Model knows valid values without explicit instructions
- **Reduced Prompt Complexity**: Structure communicates requirements
- **Reusability**: Compose schemas from smaller, reusable components
- **Better Inference**: Model can infer relationships and constraints from schema structure

## Best Practices

### 1. Write Clear, Descriptive Tool Descriptions

Tool descriptions should explain:
- What the tool does
- When to use it
- What it returns
- Any important constraints

```python
analyze_sentiment_tool = {
    "type": "function",
    "function": {
        "name": "analyze_sentiment",
        "description": (
            "Analyze the emotional sentiment of text. "
            "Use this when the user asks about feelings, emotions, or sentiment. "
            "Returns sentiment score (-1.0 to 1.0, where negative is < 0, neutral is 0, positive is > 0) "
            "and a label ('positive', 'negative', 'neutral'). "
            "Works best with complete sentences or paragraphs, not single words."
        ),
        # ... parameters ...
    }
}
```

### 2. Use Detailed Pydantic Field Descriptions

Field descriptions should specify:
- What the field represents
- Format requirements
- Constraints or validation rules
- Examples when helpful

```python
class ProductReview(BaseModel):
    rating: int = Field(
        description="Product rating from 1 to 5 stars",
        ge=1,
        le=5
    )
    summary: str = Field(
        description="Brief summary of the review (1-2 sentences, max 200 characters)",
        max_length=200
    )
    pros: list[str] = Field(
        description="List of positive aspects mentioned in the review"
    )
    cons: list[str] = Field(
        description="List of negative aspects or concerns mentioned"
    )
    verified_purchase: bool = Field(
        description="Whether the reviewer made a verified purchase (true) or not (false)"
    )
```

### 2a. Use Enums and Structured Types

Prefer enums over free-form strings for constrained values, and use typed arrays for lists:

```python
from enum import Enum
from typing import List

class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"

class User(BaseModel):
    status: Status = Field(description="User account status")
    roles: List[str] = Field(description="List of user roles")
    
# Model infers valid status values from enum
# No need to specify "must be one of: active, inactive, pending" in prompt
```

### 3. Let Schemas Communicate Requirements

Trust that the model can infer requirements from well-structured schemas:

```python
# Instead of this verbose prompt:
prompt = """Translate the text and provide:
- Source language name in English
- Source language ISO code
- Target language name in English  
- Target language ISO code
- The translated text
- Confidence level (low, medium, high)
- Cultural notes in English

Text: {text}"""

# Use this minimal prompt with schema:
prompt = "Translate to {target_language}: {text}"

# The TranslationResult Pydantic model with Field descriptions
# provides all the structure and requirements
```

### 4. Combine Explicit and Implicit Instructions

Use explicit prompts for high-level guidance, schemas for detailed requirements:

```python
# High-level instruction in prompt
prompt = """Translate the following text to {target_language} and provide a structured translation result. 
Important: Cultural notes must always be written in English. 
Text to translate: {source_text}"""

# Detailed requirements in schema
class TranslationResult(BaseModel):
    source_language: str = Field(description="Source language name in plain English")
    # ... other fields with detailed descriptions ...
```

### 5. Test Schema Descriptions

Verify that your schema descriptions are clear enough for the model to infer correctly:

```python
# Test with minimal prompt
test_prompt = "Extract user information from: John Doe, john@example.com, age 30"

response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=[{"role": "user", "content": test_prompt}],
    response_format=UserProfile
)

# If the model doesn't extract correctly, improve Field descriptions
```

## Use Cases

### Translation with Language Detection

Use tool schemas to guide language detection and Pydantic models with enums to structure translation results:

```python
from enum import Enum

# Define language codes as enum for type safety and implicit constraints
class LanguageCode(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Tool for language detection
detect_language_tool = {
    "type": "function",
    "function": {
        "name": "detect_language",
        "description": "Detect the language of text. Returns ISO 639-1 language code and confidence score.",
        # ... parameters ...
    }
}

# Structured output for translation - using enums instead of free-form strings
class TranslationResult(BaseModel):
    source_language: str = Field(description="Detected source language name")
    source_language_code: LanguageCode = Field(
        description="ISO 639-1 source language code"
    )
    target_language: str = Field(description="Target language name")
    target_language_code: LanguageCode = Field(
        description="ISO 639-1 target language code"
    )
    translated_text: str = Field(description="The translated text")
    confidence: ConfidenceLevel = Field(description="Translation confidence level")

# Model infers valid language codes and confidence levels from enums
# No need to specify valid values in the prompt
```

### Data Extraction with Validation

Extract structured data from unstructured text using schema-driven inference:

```python
class InvoiceData(BaseModel):
    invoice_number: str = Field(description="Invoice number or ID")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    total_amount: float = Field(description="Total invoice amount in USD")
    line_items: list[dict] = Field(description="List of line items with 'description' and 'amount'")
    vendor: str = Field(description="Vendor or company name")
    due_date: str = Field(description="Payment due date in YYYY-MM-DD format")

# Minimal prompt - schema provides all structure
response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=[{"role": "user", "content": f"Extract invoice data from: {invoice_text}"}],
    response_format=InvoiceData
)
```

### API Integration with Tool Calling

Use tool descriptions to guide API interactions:

```python
search_products_tool = {
    "type": "function",
    "function": {
        "name": "search_products",
        "description": (
            "Search for products in the catalog. "
            "Use this when the user wants to find products by name, category, or keywords. "
            "Returns list of matching products with name, price, and availability."
        ),
        # ... parameters ...
    }
}

# Model infers when to use this tool from the description
```

## Real-World Examples

### Example 1: Translation with Cultural Context

This example demonstrates combining tool calling with structured output, where schemas provide most of the guidance:

```python
from pydantic import BaseModel, Field
from openai import OpenAI
import json

client = OpenAI()

# Tool for cultural context
get_cultural_context_tool = {
    "type": "function",
    "function": {
        "name": "get_cultural_context",
        "description": (
            "Get cultural context information for a phrase in a language. "
            "Use this to understand cultural nuances, formality levels, and regional variations "
            "that may affect translation accuracy. Returns context about communication styles, "
            "formality expectations, and regional variations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "language_code": {"type": "string", "description": "ISO 639-1 language code"},
                "phrase": {"type": "string", "description": "The phrase to get context for"},
            },
            "required": ["language_code", "phrase"],
        },
    },
}

# Structured output model
class TranslationResult(BaseModel):
    source_language: str = Field(
        description="Source language name in plain English (e.g., 'English', 'French')"
    )
    source_language_code: str = Field(
        description="Source language code (ISO 639-1, e.g., 'en', 'fr')"
    )
    target_language: str = Field(
        description="Target language name in plain English"
    )
    target_language_code: str = Field(
        description="Target language code (ISO 639-1)"
    )
    translated_text: str = Field(description="The translated text")
    confidence: str = Field(description="Translation confidence: low, medium, high")
    cultural_notes: str = Field(
        description="Cultural or contextual notes about the translation (always in English)"
    )

# Minimal prompt - schemas provide guidance
messages = [
    {
        "role": "user",
        "content": (
            "Translate the following text to French and provide a structured translation result. "
            "Text to translate: Let's go hang out, what do you say to that?"
        ),
    }
]

# Initial call with tools
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=[get_cultural_context_tool],
    tool_choice="auto",
)

# Handle tool calls and get final structured response
# ... (see full example in scripts/examples/schema_driven_translation.py)
```

### Example 2: Document Analysis

Extract structured information from documents using schema-driven inference:

```python
class DocumentAnalysis(BaseModel):
    title: str = Field(description="Document title or heading")
    author: str = Field(description="Author name if mentioned, otherwise 'Unknown'")
    date: str = Field(description="Publication or creation date in YYYY-MM-DD format, or 'Unknown'")
    key_topics: list[str] = Field(
        description="List of 3-5 main topics or themes discussed in the document"
    )
    summary: str = Field(
        description="Brief summary of the document content (2-3 sentences)"
    )
    entities: list[dict] = Field(
        description="List of named entities (people, organizations, locations) with 'type' and 'name'"
    )

# Minimal prompt
response = client.beta.chat.completions.parse(
    model="gpt-4",
    messages=[{"role": "user", "content": f"Analyze this document: {document_text}"}],
    response_format=DocumentAnalysis
)
```

## Common Pitfalls and Solutions

### Pitfall 1: Vague Schema Descriptions

**Problem:**
```python
class User(BaseModel):
    name: str = Field(description="Name")  # Too vague
    email: str = Field(description="Email")  # No format guidance
```

**Solution:**
```python
class User(BaseModel):
    name: str = Field(
        description="Full name of the user, including first and last name"
    )
    email: str = Field(
        description="Email address in standard format (e.g., user@example.com)"
    )
```

### Pitfall 2: Overly Verbose Prompts Despite Good Schemas

**Problem:**
```python
prompt = """Extract user information. You must extract:
- Full name (first and last)
- Email in standard format
- Age as a number

Text: {text}"""

# But the Pydantic model already has detailed Field descriptions
```

**Solution:**
```python
prompt = "Extract user information from: {text}"
# Let the Pydantic model's Field descriptions provide the detail
```

### Pitfall 3: Missing Context in Tool Descriptions

**Problem:**
```python
tool = {
    "type": "function",
    "function": {
        "name": "get_data",
        "description": "Gets data",  # Too brief, no context
        # ...
    }
}
```

**Solution:**
```python
tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a location. "
            "Use this when the user asks about weather, temperature, or conditions. "
            "Returns temperature, humidity, wind speed, and conditions."
        ),
        # ...
    }
}
```

## Documentation Links

### Provider-Specific Inference Guides

1. **OpenAI Production Best Practices**
   - URL: https://platform.openai.com/docs/guides/production-best-practices
   - Description: Comprehensive guide to production inference patterns, model selection, error handling, and optimization

2. **OpenAI Reasoning Best Practices**
   - URL: https://platform.openai.com/docs/guides/reasoning-best-practices
   - Description: Guide to using reasoning models (o-series) vs. GPT models, hybrid orchestration patterns, and model selection strategies

3. **Anthropic Claude API Development Guide**
   - URL: https://www.anthropic.com/learn/build-with-claude
   - Description: Comprehensive guide to building with Claude API, including model selection, prompt caching, and structured outputs

4. **Anthropic Prompting Best Practices**
   - URL: https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-4-best-practices
   - Description: Best practices for prompt engineering with Claude, including XML tags, context provision, and chain prompts

5. **AWS Bedrock Advanced Operations Playbook**
   - URL: https://repost.aws/articles/ARD6jc9NNrQQ-FAvEpBOWiwA/amazon-bedrock-advanced-operations-playbook-optimizing-performance-cost-and-availability
   - Description: Guide to optimizing Bedrock inference for performance, cost, and availability, including prompt caching and intelligent routing

6. **AWS Bedrock Latency-Optimized Inference**
   - URL: https://aws.amazon.com/blogs/machine-learning/optimizing-ai-responsiveness-a-practical-guide-to-amazon-bedrock-latency-optimized-inference/
   - Description: Practical guide to optimizing inference latency for time-sensitive workloads

### API-Specific Documentation

7. **OpenAI Function Calling Guide**
   - URL: https://platform.openai.com/docs/guides/function-calling
   - Description: Guide to defining and using function calling with descriptive tool schemas

8. **OpenAI Structured Outputs**
   - URL: https://platform.openai.com/docs/guides/structured-outputs
   - Description: Guide to using structured outputs with Pydantic models and field descriptions

9. **OpenAI API Reference - Chat Completions**
   - URL: https://platform.openai.com/docs/api-reference/chat/create
   - Description: Complete API reference for `tools`, `response_format`, and related parameters

10. **Pydantic Field Documentation**
    - URL: https://docs.pydantic.dev/latest/concepts/fields/
    - Description: Complete guide to defining Pydantic fields with descriptions and constraints

11. **JSON Schema Specification**
    - URL: https://json-schema.org/
    - Description: Standard format for tool parameter definitions

### Related patterns

- **Function Calling / Tool Use**: Tool schemas are a key component of schema-driven inference; see [function_calling_tool_use.md](./function_calling_tool_use.md).
- **Structured Output**: Pydantic field descriptions provide implicit instructions; see [structured_output.md](./structured_output.md).
- **Prompt Engineering**: Schema-driven inference complements explicit prompting; see [prompt_engineering.md](./prompt_engineering.md).

### Learning path

Schema-Driven Inference is Pattern 5 in the [learning progression](./learning_progression.md). Learn it after **Understanding Models**, **Prompt Engineering**, **Structured Output**, and **Function Calling**; then combine them (e.g. in [schema_driven_translation.py](../scripts/examples/schema_driven_translation.py)) for minimal prompts and validated, tool-augmented outputs. Embeddings (Pattern 6) follows.

## Key Points

1. **Schemas are implicit prompts** - Tool descriptions and Field descriptions guide model behavior
2. **Reduce prompt verbosity** - Let structured definitions communicate requirements
3. **Write descriptive schemas** - Clear descriptions enable better inference
4. **Use composition and structured types** - Enums, arrays, and nested models provide implicit constraints
5. **Combine explicit and implicit** - Use prompts for high-level guidance, schemas for details
6. **Test your schemas** - Verify that descriptions are clear enough for correct inference
7. **Works with Function Calling** - Tool descriptions guide when and how to use tools
8. **Works with Structured Output** - Field descriptions specify output requirements
9. **Token efficient** - Reduces prompt tokens while maintaining quality
10. **Type safety through schemas** - Enums and typed arrays communicate valid values without explicit instructions

## Implementation Considerations

### For Production Systems

1. **Schema Versioning**: Version your tool schemas and Pydantic models for compatibility
2. **Description Quality**: Invest time in writing clear, descriptive tool and field descriptions
3. **Testing**: Test with minimal prompts to ensure schemas provide sufficient guidance
4. **Monitoring**: Track how well models infer from schemas vs. explicit prompts
5. **Iteration**: Refine schema descriptions based on model behavior and outputs
6. **Documentation**: Document your schema-driven inference patterns for team understanding

### Integration with Other Patterns

- **Schema-Driven Inference + Function Calling**: Tool descriptions guide tool selection and usage
- **Schema-Driven Inference + Structured Output**: Field descriptions specify output format and constraints
- **Schema-Driven Inference + Prompt Engineering**: Combine minimal prompts with detailed schemas
- **Schema-Driven Inference + RAG**: Use schemas to structure retrieved context

---

*Reference: AI Technology Engineering Patterns - Schema-Driven Inference*
