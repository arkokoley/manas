---
title: QANode
description: Question-answering node with optional RAG support
parent: Nodes
grand_parent: API Reference
---

# QANode

`QANode` is a specialized node for question answering with optional RAG (Retrieval-Augmented Generation) support. It can process questions with or without additional context and maintain conversation history.

For more information on RAG integration, see the [RAG documentation]({{ site.baseurl }}/api/rag/).

For examples of using QANode in flows, check out:
- [Research Flow Example]({{ site.baseurl }}/examples/research-flow/)
- [Knowledge Base QA]({{ site.baseurl }}/examples/knowledge-base-qa/)

## Import

```python
from manas_ai.nodes import QANode, QAConfig
```

## Configuration

### QAConfig

Configuration class for QA nodes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | str | None | Model name to use |
| temperature | float | 0.7 | Generation temperature |
| use_rag | bool | False | Whether to use RAG |
| rag_config | Dict[str, Any] | None | RAG configuration |
| prompt_template | str | See below | Template for generating prompts |
| follow_up_template | str | See below | Template for follow-up questions |
| options | Dict[str, Any] | {} | Additional options |

Default prompt template:
```python
"Question: {question}\nContext: {context}\nAnswer:"
```

Default follow-up template:
```python
"Previous Answer: {previous_answer}\nFollow-up Question: {question}\nAnswer:"
```

## Constructor

```python
def __init__(
    name: str, 
    config: QAConfig,
    llm_node: Optional[LLMNode] = None,
    rag_node: Optional[RAGNode] = None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Name of the node |
| config | QAConfig | Node configuration |
| llm_node | Optional[LLMNode] | LLM node to use (created if not provided) |
| rag_node | Optional[RAGNode] | RAG node to use (created if enabled and not provided) |

## Methods

### answer

```python
async def answer(
    question: str,
    context: Optional[str] = None,
    session_id: Optional[str] = None,
    include_history: bool = True
) -> Dict[str, Any]
```

Process a question and generate an answer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| question | str | Required | The question to answer |
| context | Optional[str] | None | Additional context |
| session_id | Optional[str] | None | Session ID for history |
| include_history | bool | True | Whether to use history |

Returns:
```python
{
    "question": str,      # Original question
    "answer": str,       # Generated answer
    "confidence": float, # Confidence score
    "sources": List[Dict[str, Any]] # If RAG is enabled
}
```

### process

```python
async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]
```

Process node inputs. Accepts:
- `question`: Question string
- `context`: Optional context
- `session_id`: Optional session ID
- `include_history`: Whether to use history

Returns the result from `answer()`.

## Examples

### Basic Usage

```python 
# Create a simple QA node
qa_node = QANode(
    name="qa",
    config=QAConfig(model="gpt-4", temperature=0.7)
)

# Get an answer
result = await qa_node.answer("What is quantum computing?")
print(result["answer"])
```

### With RAG Support

```python
# Create a QA node with RAG
qa_node = QANode(
    name="qa",
    config=QAConfig(
        model="gpt-4",
        use_rag=True,
        rag_config={
            "chunk_size": 500,
            "chunk_overlap": 50
        }
    )
)

# Add documents to RAG
await qa_node.rag_node.add_documents([
    "path/to/documents/"
])

# Get answer with automatic context retrieval
result = await qa_node.answer(
    "Explain the theory of relativity",
    session_id="physics_session"
)
```

### In a Flow

```python
from core import Flow

# Create flow
flow = Flow()
flow.add_node(qa_node)

# Process through flow
result = await flow.process({
    "prompt": "What causes gravity?"
})
```

## Notes

- Initialize the node with `await node.initialize()` before use
- Clean up resources with `await node.cleanup()` when done
- Use session IDs to maintain conversation history
- Configure RAG parameters based on your document types