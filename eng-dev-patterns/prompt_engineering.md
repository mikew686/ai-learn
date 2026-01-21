# Prompt Engineering Pattern

## Overview

**Prompt Engineering** is the art and science of designing effective prompts to guide LLM behavior and achieve desired outputs. It involves structuring instructions, providing context, and using various techniques to improve response quality, reliability, and consistency.

## Description

From the AI Technologies reference guide:

> **Description**: The art and science of designing effective prompts to guide LLM behavior and achieve desired outputs. Involves structuring instructions, providing context, and using various techniques to improve response quality.

**Key Concepts**:
- Template-based prompts with variable substitution
- Few-shot learning (providing examples in prompts)
- Role-setting ("You are an expert at...")
- Structured instructions with numbered steps
- Constraint injection (business rules in prompts)
- Output format specification

## How It Works

### 1. Basic Prompt Structure

A well-structured prompt typically includes system and user messages. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

system_prompt = """You are an expert data analyst specializing in technical documentation.
Your task is to analyze API documentation and extract key information.
Always format your responses as JSON."""

user_prompt = """Analyze the following API documentation and extract:
1. Endpoints and methods
2. Required parameters
3. Response formats

API Documentation:
{doc_content}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(doc_content="...")}
    ]
)

print(response.choices[0].message.content)
```

### 2. Role-Setting

Establish the AI's role and expertise. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """You are a senior software engineer with 10 years of experience in Python.
You specialize in writing clean, maintainable code following PEP 8 standards.
Your code should include type hints and comprehensive docstrings."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 3. Few-Shot Learning

Provide examples to guide the model's behavior. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Convert the following dates to ISO 8601 format.

Example 1:
Input: "March 15, 2024"
Output: "2024-03-15"

Example 2:
Input: "12/25/2023"
Output: "2023-12-25"

Now convert:
Input: "January 1, 2025"
Output:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 4. Structured Instructions

Use numbered steps for complex tasks. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Generate a technical specification following these steps:

Step 1: Identify the core requirements
Step 2: List 3-5 key technical terms
Step 3: Outline the system architecture (Components, Integration, Testing)
Step 4: Specify required technologies
Step 5: Define success criteria

Domain: {domain}
Technology Stack: {tech_stack}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(domain="Web Application", tech_stack="Python, React, PostgreSQL")}]
)

print(response.choices[0].message.content)
```

### 5. Constraint Injection

Include business rules and constraints. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Generate API endpoint documentation.

Constraints:
- Must be appropriate for developers with intermediate experience
- Must align with REST API best practices
- Must not exceed 10 endpoints
- Must include request/response examples
- Must use clear technical language

Generate the documentation:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 6. Output Format Specification

Specify the exact format you want. Using the OpenAI SDK:

```python
from openai import OpenAI
import json

client = OpenAI()

prompt = """Extract user information from the following text.

Format your response as JSON with this exact structure:
{
  "name": "string",
  "role": "string",
  "technologies": ["string"],
  "experience_years": "number"
}

Text: {user_text}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(user_text=text)}],
    response_format={"type": "json_object"}
)

user_info = json.loads(response.choices[0].message.content)
print(user_info)
```

## Key Techniques

### Template-Based Prompts

Use templates with variable substitution for reusability. Using the OpenAI SDK:

```python
from openai import OpenAI
from string import Template

client = OpenAI()

template = Template("""You are a ${role} expert.
Analyze the following ${content_type} and provide ${output_type}.

Content:
${content}""")

prompt = template.substitute(
    role="software architect",
    content_type="technical specification",
    output_type="system requirements",
    content=spec_text
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
```

### Chain-of-Thought Prompting

Encourage step-by-step reasoning. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Solve this math problem step by step.

Problem: If a train travels 120 miles in 2 hours, how fast is it going?

Let's think through this:
1. First, identify what we're looking for: speed
2. Recall the formula: speed = distance / time
3. Plug in the values: speed = 120 miles / 2 hours
4. Calculate: speed = 60 miles per hour

Now solve: If a car travels 180 miles in 3 hours, how fast is it going?"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### Zero-Shot vs Few-Shot

**Zero-Shot** (no examples). Using the OpenAI SDK:
```python
from openai import OpenAI

client = OpenAI()

prompt = "Translate the following English text to Spanish: Hello, how are you?"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

**Few-Shot** (with examples). Using the OpenAI SDK:
```python
from openai import OpenAI

client = OpenAI()

prompt = """Translate English to Spanish:

English: "Good morning"
Spanish: "Buenos días"

English: "Thank you"
Spanish: "Gracias"

English: "Hello, how are you?"
Spanish:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### Prompt Chaining

Break complex tasks into multiple prompts. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

# Step 1: Extract information
extraction_prompt = """Extract key facts from this article: {article}"""
extraction_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": extraction_prompt.format(article=article_text)}]
)
facts = extraction_response.choices[0].message.content

# Step 2: Summarize
summary_prompt = """Summarize these facts into 3 bullet points: {facts}"""
summary_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": summary_prompt.format(facts=facts)}]
)
summary = summary_response.choices[0].message.content

# Step 3: Generate questions
question_prompt = """Generate 5 quiz questions based on this summary: {summary}"""
question_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": question_prompt.format(summary=summary)}]
)
questions = question_response.choices[0].message.content
```

### Negative Prompting

Specify what NOT to do. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Write a product description for a software library.

DO:
- Use clear technical language
- Highlight key features and benefits
- Keep it under 100 words

DON'T:
- Use marketing jargon
- Include pricing information
- Make performance claims without benchmarks

Product: {product_name}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(product_name="Data Processing Library")}]
)

print(response.choices[0].message.content)
```

## Best Practices

### 1. Be Specific and Clear

**Bad:**
```python
from openai import OpenAI

client = OpenAI()
prompt = "Write about math"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

**Good:**
```python
from openai import OpenAI

client = OpenAI()

prompt = """Write a 200-word explanation of API authentication for developers.
Use clear technical language and include one example with code snippet."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 2. Provide Context

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Context: You are helping a developer create API documentation for a REST API.
The API has been in development for 2 weeks.

Task: Create documentation for the authentication endpoint.
Include: endpoint description, request format, response format, and error handling."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### 3. Use Examples for Complex Tasks

Using the OpenAI SDK:

```python
from openai import OpenAI
import json

client = OpenAI()

prompt = """Classify these software libraries by category and use case.

Example:
Resource: "Introduction to React Hooks"
Classification: {"category": "Frontend", "use_case": "State Management", "topic": "React"}

Now classify:
Resource: "Database Migration Tools"
Classification:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)

classification = json.loads(response.choices[0].message.content)
print(classification)
```

### 4. Separate System and User Prompts

Using the OpenAI SDK to separate system and user prompts:

```python
from openai import OpenAI

client = OpenAI()

system_message = {
    "role": "system",
    "content": "You are a helpful technical documentation assistant. Always provide accurate, clear technical information."
}

user_message = {
    "role": "user",
    "content": "Explain API authentication to a developer."
}

response = client.chat.completions.create(
    model="gpt-4",
    messages=[system_message, user_message]
)

print(response.choices[0].message.content)
```

### 5. Version Control Your Prompts

```python
# prompts/v1_api_doc_generator.py
API_DOC_GENERATOR_V1 = """Generate API documentation..."""

# prompts/v2_api_doc_generator.py
API_DOC_GENERATOR_V2 = """Generate API documentation with improved structure..."""
```

### 6. Test Systematically

```python
test_cases = [
    {"input": "REST API, Python", "expected_format": "JSON"},
    {"input": "GraphQL, TypeScript", "expected_format": "JSON"},
    {"input": "WebSocket, Node.js", "expected_format": "JSON"},
]

for test in test_cases:
    result = generate_api_spec(test["input"])
    assert validate_format(result, test["expected_format"])
```

## Advanced Techniques

### Temperature Control

Adjust randomness in prompts using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

# Creative tasks - higher temperature
creative_prompt = "Write a creative story about a robot learning to paint."
creative_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": creative_prompt}],
    temperature=0.9
)

# Factual tasks - lower temperature
factual_prompt = "List the state capitals of the United States."
factual_response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": factual_prompt}],
    temperature=0.1
)
```

### Prompt Injection Prevention

Protect against prompt injection:

```python
def sanitize_user_input(user_input: str) -> str:
    """Remove potential prompt injection attempts."""
    # Remove common injection patterns
    dangerous_patterns = [
        "ignore previous instructions",
        "forget everything",
        "you are now",
    ]
    
    for pattern in dangerous_patterns:
        if pattern.lower() in user_input.lower():
            raise ValueError(f"Potential prompt injection detected: {pattern}")
    
    return user_input
```

### Dynamic Prompt Construction

Build prompts based on context. Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

def build_prompt(task_type: str, context: dict) -> str:
    base_prompt = "You are an expert {role}."
    
    if task_type == "analysis":
        prompt = base_prompt.format(role="data analyst")
        prompt += f"\nAnalyze: {context['data']}"
    elif task_type == "generation":
        prompt = base_prompt.format(role="content creator")
        prompt += f"\nGenerate: {context['topic']}"
    
    return prompt

# Use the dynamic prompt
prompt = build_prompt("analysis", {"data": "sales data for Q1"})
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

## Use Cases

### Content Generation

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Generate a blog post about {topic}.

Requirements:
- Title: Catchy and SEO-friendly
- Introduction: Hook the reader in 2-3 sentences
- Body: 3-5 paragraphs with clear points
- Conclusion: Summarize key takeaways
- Word count: 800-1000 words
- Tone: Professional but approachable"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(topic="AI in Education")}]
)

blog_post = response.choices[0].message.content
print(blog_post)
```

### Data Extraction

Using the OpenAI SDK:

```python
from openai import OpenAI
import json

client = OpenAI()

prompt = """Extract structured data from this text.

Text: {unstructured_text}

Extract:
- Names of people mentioned
- Dates mentioned
- Key statistics or numbers
- Main topics discussed

Format as JSON."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(unstructured_text=text)}],
    response_format={"type": "json_object"}
)

extracted = json.loads(response.choices[0].message.content)
print(extracted)
```

### Code Generation

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Write a Python function that {function_description}.

Requirements:
- Include type hints
- Add comprehensive docstring
- Handle edge cases
- Include error handling
- Follow PEP 8 style guide
- Include example usage

Function description: {function_description}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(function_description="calculates fibonacci numbers")}]
)

code = response.choices[0].message.content
print(code)
```

### Question Answering

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Answer the following question based on the provided context.

Context: {context}

Question: {question}

Instructions:
- Base your answer only on the provided context
- If the answer is not in the context, say "I don't have enough information"
- Be concise but complete
- Cite specific parts of the context if relevant"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(context=context_text, question=user_question)}]
)

answer = response.choices[0].message.content
print(answer)
```

### Translation

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

prompt = """Translate the following text from {source_language} to {target_language}.

Text: {text}

Guidelines:
- Maintain the original tone and style
- Preserve technical terms when appropriate
- Keep formatting (line breaks, lists, etc.)
- Ensure cultural appropriateness"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(
        source_language="English",
        target_language="Spanish",
        text="Hello, how are you?"
    )}]
)

translation = response.choices[0].message.content
print(translation)
```

## Real-World Examples

### Example 1: API Specification Generator

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

API_SPEC_PROMPT = """You are an expert software architect creating API specifications.

Generate an API specification with the following structure:

1. **Objective**: One clear, measurable goal
2. **Technology Stack**: {tech_stack}
3. **Domain**: {domain}
4. **Timeline**: {timeline} weeks
5. **Required Technologies**: Bulleted list
6. **API Structure**:
   - Authentication (setup)
   - Core endpoints (implementation)
   - Data models (definition)
   - Error handling (validation)
   - Testing (verification)
7. **Scalability**: How to adapt for different load levels
8. **Monitoring**: How to measure API performance

Feature: {feature}
Standards Alignment: {standards}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": API_SPEC_PROMPT.format(
            tech_stack="Python, FastAPI, PostgreSQL",
            domain="Data Processing",
            timeline="4",
            feature="User Authentication",
            standards="REST API Best Practices"
        )}
    ]
)

api_spec = response.choices[0].message.content
print(api_spec)
```

### Example 2: Data Extraction from PDFs

Using the OpenAI SDK:

```python
from openai import OpenAI
import json

client = OpenAI()

EXTRACTION_PROMPT = """Extract structured information from this technical document.

Document Text:
{document_text}

Extract the following information as JSON:
{{
  "title": "string",
  "author": "string",
  "technology": "string",
  "category": "string",
  "key_concepts": ["string"],
  "use_cases": ["string"],
  "implementation_methods": ["string"]
}}

Ensure all extracted data is accurate and complete."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": EXTRACTION_PROMPT.format(document_text=document_text)}
    ],
    response_format={"type": "json_object"}  # Request JSON output
)

extracted_data = json.loads(response.choices[0].message.content)
print(extracted_data)
```

### Example 3: Code Review Assistant

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI()

CODE_REVIEW_PROMPT = """Review this Python code and provide feedback.

Code:
```python
{code}
```

Review criteria:
1. **Correctness**: Does it work as intended?
2. **Style**: Does it follow PEP 8?
3. **Performance**: Are there optimization opportunities?
4. **Security**: Any security concerns?
5. **Documentation**: Is it well-documented?

Provide feedback in this format:
- **Issues**: List of problems found
- **Suggestions**: Improvement recommendations
- **Strengths**: What's done well"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": CODE_REVIEW_PROMPT.format(code=python_code)}
    ]
)

review = response.choices[0].message.content
print(review)
```

## Common Pitfalls and Solutions

### Pitfall 1: Vague Instructions

**Problem:**
```python
from openai import OpenAI

client = OpenAI()
prompt = "Write something about education"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

**Solution:**
```python
from openai import OpenAI

client = OpenAI()

prompt = """Write a 300-word article about the benefits of microservices architecture in modern applications.
Target audience: Software engineers and architects
Tone: Informative and technical
Include: 3 specific benefits with examples"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### Pitfall 2: Too Many Constraints

**Problem:**
```python
from openai import OpenAI

client = OpenAI()

prompt = """Write a story that is exactly 500 words, uses only 5-letter words, 
has 3 characters, takes place in 1920s Paris, includes a mystery, 
has a happy ending, and teaches about math."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

**Solution:**
```python
from openai import OpenAI

client = OpenAI()

prompt = """Write a short story (approximately 500 words) about a developer 
discovering the power of a new programming framework. Set in a tech startup environment. 
Include a moment of realization or breakthrough."""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

### Pitfall 3: Ignoring Context

**Problem:**
```python
from openai import OpenAI

client = OpenAI()

prompt = "Summarize this: {text}"

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(text=text)}]
)
```

**Solution:**
```python
from openai import OpenAI

client = OpenAI()

prompt = """You are summarizing technical documentation for developers.

Context: This text is from an API reference guide for a REST service.

Task: Create a concise summary (3-4 sentences) highlighting:
- Main topic
- Key technical points
- Practical applications

Text: {text}"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt.format(text=text)}]
)

print(response.choices[0].message.content)
```

## Documentation Links

### Official Documentation

1. **OpenAI Prompt Engineering Guide**
   - URL: https://platform.openai.com/docs/guides/prompt-engineering
   - Description: Comprehensive guide to prompt engineering best practices

2. **Anthropic Prompt Engineering**
   - URL: https://docs.anthropic.com/claude/docs/prompt-engineering
   - Description: Claude-specific prompt engineering techniques

3. **LangChain Prompt Templates**
   - URL: https://python.langchain.com/docs/modules/model_io/prompts/
   - Description: Template system for building reusable prompts

4. **Prompt Engineering Guide (Lil'Log)**
   - URL: https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/
   - Description: Academic perspective on prompt engineering techniques

5. **OpenAI Cookbook - Prompt Engineering**
   - URL: https://cookbook.openai.com/
   - Description: Practical examples and patterns

### Research & Papers

- **Chain-of-Thought Prompting**: https://arxiv.org/abs/2201.11903
- **ReAct: Synergizing Reasoning and Acting**: https://arxiv.org/abs/2210.03629
- **Prompt Engineering for Large Language Models**: https://arxiv.org/abs/2312.16171

### Tools & Frameworks

- **PromptLayer**: https://promptlayer.com/ - Prompt versioning and management
- **LangChain Prompt Hub**: https://smith.langchain.com/hub - Community prompt library
- **Weights & Biases Prompts**: https://wandb.ai/ - Prompt tracking and optimization

## Key Points

1. **Clarity is essential** - Be specific about what you want
2. **Context matters** - Provide relevant background information
3. **Examples help** - Few-shot learning improves consistency
4. **Structure your prompts** - Use clear sections and formatting
5. **Test and iterate** - Prompt engineering is an iterative process
6. **Version control** - Track prompt changes and their impact
7. **Separate concerns** - Use system and user messages appropriately
8. **Specify output format** - Make parsing easier with structured outputs

## Implementation Considerations

### For Production Systems

1. **Prompt Templates**: Use template systems (Jinja2, LangChain) for maintainability
2. **A/B Testing**: Test different prompt variations to find optimal performance
3. **Monitoring**: Track prompt performance and response quality
4. **Security**: Implement prompt injection protection
5. **Caching**: Cache prompts and responses when appropriate
6. **Versioning**: Maintain prompt versions for rollback capability

### Integration with Other Patterns

- **Prompt Engineering + Structured Output**: Use prompts to guide structured responses
- **Prompt Engineering + RAG**: Combine prompts with retrieved context
- **Prompt Engineering + Function Calling**: Use prompts to guide tool selection
- **Prompt Engineering + Few-Shot Learning**: Provide examples in prompts

---

*Reference: AI Technology Engineering Patterns - Prompt Engineering*
