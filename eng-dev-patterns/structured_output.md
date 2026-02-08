# Structured Output Pattern

## Overview

**Structured Output** is an AI engineering technique that enforces structured, validated responses from LLMs using schemas. This ensures outputs conform to expected formats and can be reliably parsed, making LLM responses directly usable in applications without manual parsing or validation.

## Description

From the AI Technologies reference guide:

> **Description**: Enforcing structured, validated responses from LLMs using schemas. Ensures outputs conform to expected formats and can be reliably parsed.

**Key Concepts**:
- Schema definition (JSON Schema, Pydantic models)
- Response validation
- Type safety
- Automatic parsing
- Error handling

## How It Works

### 1. Define Schema with Pydantic

Create a Pydantic model that defines the expected output structure:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class User(BaseModel):
    name: str = Field(description="Full name of the user")
    email: str = Field(description="Email address")
    age: int = Field(description="Age in years", ge=0, le=150)
    roles: List[str] = Field(description="List of user roles")
    active: bool = Field(description="Whether the user account is active")
    metadata: Optional[dict] = Field(default=None, description="Additional metadata")
```

### 2. Use OpenAI SDK with Structured Output

Use the OpenAI SDK's `.parse()` method to automatically parse responses into your Pydantic model. **Note: The `.parse()` method is currently in the `beta` namespace and requires `client.beta.chat.completions.parse()`.**

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List

client = OpenAI()

# Define your Pydantic model
class APIEndpoint(BaseModel):
    path: str
    method: str
    description: str
    parameters: List[dict]

# Make API call with response_format
# Note: .beta is required for the .parse() method
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Describe the /users endpoint for a REST API"}
    ],
    response_format=APIEndpoint
)

# Response is automatically parsed into your Pydantic model
endpoint = response.choices[0].message.parsed
print(endpoint.path)  # Type-safe access
print(endpoint.method)
```

### 3. Alternative: Using JSON Schema

You can also use JSON Schema directly with the OpenAI SDK:

```python
from openai import OpenAI
import json

client = OpenAI()

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    },
    "required": ["name", "email", "age"]
}

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Extract user information from: John Doe, john@example.com, 30 years old"}
    ],
    response_format={"type": "json_schema", "json_schema": {"schema": json_schema}}
)

# Parse the JSON response
user_data = json.loads(response.choices[0].message.content)
print(user_data)
```

## Key Approaches

### Pydantic Models (Recommended)

Pydantic provides type safety, validation, and automatic parsing. **The `.parse()` method requires the `beta` namespace:**

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD", gt=0)
    category: str = Field(description="Product category")
    tags: List[str] = Field(description="Product tags", min_items=1)
    in_stock: bool = Field(description="Availability status")
    created_at: Optional[datetime] = None
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

# Use with OpenAI - .beta is required for .parse()
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Create a product: Laptop, $999, Electronics"}],
    response_format=Product
)

product = response.choices[0].message.parsed
print(product.name)  # Type-safe
print(product.price)  # Validated
```

### JSON Schema

For more control or when not using Python:

```python
from openai import OpenAI

client = OpenAI()

schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high"]
        }
    },
    "required": ["title", "summary", "priority"]
}

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Create a task: Fix login bug, high priority"}],
    response_format={"type": "json_schema", "json_schema": {"schema": schema}}
)
```

### Response Format Options

OpenAI supports different response formats:

```python
# JSON object mode (flexible) - Stable API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract data"}],
    response_format={"type": "json_object"}
)

# JSON schema mode (strict validation) - Stable API
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract data"}],
    response_format={"type": "json_schema", "json_schema": {"schema": schema}}
)

# Pydantic parsing (type-safe, recommended) - Beta API
# Note: .beta is required for the .parse() method
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract data"}],
    response_format=YourPydanticModel
)
```

## Advanced Patterns

### Nested Models

Define complex nested structures:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Contact(BaseModel):
    phone: Optional[str] = None
    email: str

class Company(BaseModel):
    name: str
    address: Address
    contacts: List[Contact]
    employee_count: int = Field(ge=0)

# Use nested model
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Extract company info from: Acme Corp, 123 Main St, NYC"}],
    response_format=Company
)

company = response.choices[0].message.parsed
print(company.address.city)  # Type-safe nested access
```

### Union Types

Handle multiple possible response formats:

```python
from pydantic import BaseModel
from typing import Union

class SuccessResponse(BaseModel):
    status: str = "success"
    data: dict

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    code: int

ResponseType = Union[SuccessResponse, ErrorResponse]

# Note: OpenAI's parse() works with the first matching type
# For true union handling, you may need to parse manually
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Process this request"}],
    response_format=SuccessResponse  # Use most common case
)
```

### Custom Validators

Add business logic validation:

```python
from pydantic import BaseModel, Field, validator
from typing import List

class APIRequest(BaseModel):
    endpoint: str
    method: str
    headers: dict
    body: Optional[dict] = None
    
    @validator('method')
    def validate_method(cls, v):
        allowed = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        if v.upper() not in allowed:
            raise ValueError(f'Method must be one of {allowed}')
        return v.upper()
    
    @validator('endpoint')
    def validate_endpoint(cls, v):
        if not v.startswith('/'):
            raise ValueError('Endpoint must start with /')
        return v

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Create API request: GET /users"}],
    response_format=APIRequest
)

request = response.choices[0].message.parsed
# Method is automatically validated and normalized
```

### Lists and Arrays

Handle multiple items:

```python
from pydantic import BaseModel
from typing import List

class User(BaseModel):
    id: str
    name: str
    email: str

class UserList(BaseModel):
    users: List[User]
    total: int

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 sample users"}],
    response_format=UserList
)

user_list = response.choices[0].message.parsed
for user in user_list.users:
    print(f"{user.name} ({user.email})")
```

## Error Handling

### Handling Parsing Errors

```python
from openai import OpenAI
from pydantic import ValidationError

client = OpenAI()

try:
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Extract user data"}],
        response_format=User
    )
    user = response.choices[0].message.parsed
except ValidationError as e:
    print(f"Validation error: {e}")
    # Handle invalid response
except Exception as e:
    print(f"API error: {e}")
    # Handle API errors
```

### Fallback Mechanisms

```python
from openai import OpenAI
import json

client = OpenAI()

def extract_with_fallback(prompt: str, model_class):
    try:
        # Try structured parsing first
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format=model_class
        )
        return response.choices[0].message.parsed
    except Exception:
        # Fallback to JSON mode
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        # Manual validation
        return model_class(**data)
```

## Best Practices

### 1. Use Descriptive Field Descriptions

Field descriptions help the model understand what to generate:

```python
class Task(BaseModel):
    title: str = Field(description="Brief, actionable task title (max 50 chars)")
    description: str = Field(description="Detailed task description")
    priority: str = Field(description="Priority level: low, medium, high, urgent")
    due_date: Optional[str] = Field(description="Due date in ISO 8601 format (YYYY-MM-DD)")
```

### 2. Provide Examples in Prompts

Help the model understand the expected format:

```python
prompt = """Extract task information. Use this format:

Example:
{
  "title": "Fix authentication bug",
  "description": "Users cannot log in with OAuth",
  "priority": "high",
  "due_date": "2024-12-31"
}

Now extract from: {user_input}"""

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt.format(user_input=input_text)}],
    response_format=Task
)
```

### 3. Use Appropriate Types

Choose the right Pydantic types for validation:

```python
from pydantic import BaseModel, EmailStr, HttpUrl
from typing import List, Optional
from datetime import datetime

class Contact(BaseModel):
    name: str
    email: EmailStr  # Validates email format
    website: Optional[HttpUrl] = None  # Validates URL format
    tags: List[str] = []
    created_at: datetime  # Validates datetime format
```

### 4. Set Reasonable Constraints

Use Field constraints to guide the model:

```python
from pydantic import BaseModel, Field

class Article(BaseModel):
    title: str = Field(min_length=10, max_length=100)
    content: str = Field(min_length=100)
    word_count: int = Field(ge=100, le=5000)
    published: bool = False
```

### 5. Handle Optional Fields

Make fields optional when appropriate:

```python
from pydantic import BaseModel
from typing import Optional

class Product(BaseModel):
    name: str
    price: float
    description: Optional[str] = None  # May not always be provided
    category: Optional[str] = None
    tags: Optional[List[str]] = None
```

## Use Cases

### Data Extraction

Extract structured data from unstructured text:

```python
from pydantic import BaseModel
from typing import List

class ExtractedData(BaseModel):
    entities: List[str]
    dates: List[str]
    numbers: List[float]
    summary: str

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": f"Extract entities, dates, numbers, and summary from: {text}"
    }],
    response_format=ExtractedData
)

data = response.choices[0].message.parsed
```

### API Response Generation

Generate structured API responses:

```python
from pydantic import BaseModel
from typing import List, Optional

class APIResponse(BaseModel):
    status: str
    data: Optional[dict] = None
    errors: Optional[List[str]] = None
    metadata: Optional[dict] = None

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Generate API response for successful user creation"}],
    response_format=APIResponse
)
```

### Configuration Generation

Generate configuration files:

```python
from pydantic import BaseModel
from typing import List, Dict

class Config(BaseModel):
    environment: str
    database_url: str
    api_keys: Dict[str, str]
    features: List[str]
    timeout: int

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Generate production config"}],
    response_format=Config
)
```

### Form Filling

Extract form data:

```python
from pydantic import BaseModel, EmailStr
from typing import Optional

class FormData(BaseModel):
    first_name: str
    last_name: str
    email: EmailStr
    phone: Optional[str] = None
    address: Optional[str] = None

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"Extract form data from: {user_input}"}],
    response_format=FormData
)
```

## Real-World Examples

### Example 1: API Documentation Generator

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Parameter(BaseModel):
    name: str
    type: str
    required: bool
    description: str
    default: Optional[str] = None

class Endpoint(BaseModel):
    path: str
    method: str
    description: str
    parameters: List[Parameter]
    response_schema: dict
    status_codes: List[int]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Generate API documentation for a user management endpoint"
    }],
    response_format=Endpoint
)

endpoint = response.choices[0].message.parsed
print(f"{endpoint.method} {endpoint.path}")
```

### Example 2: Code Analysis

```python
from pydantic import BaseModel
from typing import List, Optional

class Function(BaseModel):
    name: str
    parameters: List[str]
    return_type: Optional[str] = None
    complexity: str
    issues: List[str]

class CodeAnalysis(BaseModel):
    functions: List[Function]
    overall_complexity: str
    recommendations: List[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": f"Analyze this code: {code_snippet}"
    }],
    response_format=CodeAnalysis
)

analysis = response.choices[0].message.parsed
for func in analysis.functions:
    print(f"{func.name}: {func.complexity}")
```

### Example 3: Database Schema Generation

```python
from pydantic import BaseModel
from typing import List

class Column(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = None

class Table(BaseModel):
    name: str
    columns: List[Column]
    indexes: List[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Generate database schema for a blog system with users and posts"
    }],
    response_format=List[Table]
)

tables = response.choices[0].message.parsed
for table in tables:
    print(f"Table: {table.name}")
    for col in table.columns:
        print(f"  {col.name}: {col.type}")
```

## Common Pitfalls and Solutions

### Pitfall 1: Overly Complex Schemas

**Problem:**
```python
# Too many nested levels and optional fields
class OverlyComplex(BaseModel):
    data: dict
    nested: dict
    more_nested: Optional[dict] = None
    # ... many more fields
```

**Solution:**
```python
# Break into smaller, focused models
class SimpleData(BaseModel):
    id: str
    name: str

class Response(BaseModel):
    data: SimpleData
    metadata: Optional[dict] = None
```

### Pitfall 2: Missing Field Descriptions

**Problem:**
```python
class Task(BaseModel):
    title: str  # No description
    priority: str  # Unclear what values are expected
```

**Solution:**
```python
class Task(BaseModel):
    title: str = Field(description="Task title, max 100 characters")
    priority: str = Field(description="Priority: low, medium, high, or urgent")
```

### Pitfall 3: Not Handling Errors

**Problem:**
```python
response = client.beta.chat.completions.parse(...)
data = response.choices[0].message.parsed  # May fail
```

**Solution:**
```python
try:
    response = client.beta.chat.completions.parse(...)
    data = response.choices[0].message.parsed
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Invalid response: {e}")
    # Fallback or retry
```

## Related patterns

- **Schema-Driven Inference**: Use Pydantic field descriptions as implicit prompts to reduce verbosity; see [schema_driven_inference.md](./schema_driven_inference.md).
- **Function Calling / Tool Use**: Tool responses are often validated with structured output; see [function_calling_tool_use.md](./function_calling_tool_use.md).
- **Prompt Engineering**: Clear prompts improve structured output quality; see [prompt_engineering.md](./prompt_engineering.md).
- **RAG**: Structure retrieved context or model answers using schemas; see [learning_progression.md](./learning_progression.md) Pattern 8.

## Practical technologies

- **Python**: [Pydantic](https://docs.pydantic.dev/) for models and validation; OpenAI `client.beta.chat.completions.parse()` for native parsing.
- **Alternatives**: [JSON Schema](https://json-schema.org/) with `response_format={"type": "json_schema", "json_schema": {...}}` when not using Pydantic or when avoiding beta API; [Outlines](https://github.com/outlines-dev/outlines) for constrained generation; [Zod](https://zod.dev/) (TypeScript) for schema validation.
- **Learning path**: Structured output is Pattern 3 in the [learning progression](./learning_progression.md); use it after Understanding Models and Prompt Engineering and before Function Calling and Embeddings.

## Documentation Links

### Official Documentation

1. **OpenAI Structured Outputs**
   - URL: https://platform.openai.com/docs/guides/structured-outputs
   - Description: Comprehensive guide to OpenAI's structured output features

2. **OpenAI API Reference - Chat Completions**
   - URL: https://platform.openai.com/docs/api-reference/chat/create
   - Description: Complete API reference for `response_format` parameter

3. **Pydantic Documentation**
   - URL: https://docs.pydantic.dev/
   - Description: Complete Pydantic documentation for model definition and validation

4. **OpenAI Python SDK - Parse Method**
   - URL: https://github.com/openai/openai-python
   - Description: Python SDK documentation for `.parse()` method

5. **JSON Schema Specification**
   - URL: https://json-schema.org/
   - Description: Standard schema format for validation

### Related Patterns

- **Function Calling**: Often used together with structured outputs for validated tool responses
- **Prompt Engineering**: Effective prompts improve structured output quality
- **RAG**: Structured outputs help format retrieved information

## Key Points

1. **Pydantic provides type safety** - Automatic validation and type checking
2. **OpenAI's `.parse()` method** - Simplifies structured output handling (requires `.beta` namespace)
3. **Field descriptions matter** - Help the model understand expected format
4. **Handle errors gracefully** - Always validate and provide fallbacks
5. **Use appropriate constraints** - Guide the model with Field validators
6. **Nested models work well** - Complex structures are supported
7. **JSON Schema is an alternative** - Useful when not using Python/Pydantic or when avoiding beta APIs
8. **Beta API note** - The `.parse()` method is in `client.beta.chat.completions.parse()` - this is the correct API to use

## Implementation Considerations

### For Production Systems

1. **Error Handling**: Always wrap parsing in try-except blocks
2. **Validation**: Use Pydantic validators for business logic
3. **Fallbacks**: Implement fallback mechanisms for reliability
4. **Monitoring**: Track parsing success rates and errors
5. **Versioning**: Version your schemas for compatibility
6. **Testing**: Test with various inputs to ensure robustness
7. **Beta API**: The `.parse()` method uses `client.beta.chat.completions.parse()` - this is the correct and recommended API for Pydantic models, despite being in the beta namespace

### Integration with Other Patterns

- **Structured Output + Function Calling**: Return validated tool results
- **Structured Output + Prompt Engineering**: Use prompts to guide structured generation
- **Structured Output + RAG**: Format retrieved context into structured data

---

*Reference: AI Technology Engineering Patterns - Structured Output*
