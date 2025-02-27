---
title: Building a Research Flow
description: Learn how to create a multi-node research flow that can research, analyze, and summarize topics
nav_order: 1
parent: Examples
difficulty: Intermediate
time: 30 minutes
---

# Building a Research Flow

This tutorial demonstrates how to build a multi-node flow for researching topics, analyzing findings, and generating reports.

## Overview

We'll create a flow with three specialized nodes:
- A researcher node that gathers information
- An analyst node that identifies patterns and insights
- A writer node that produces a final report

## Prerequisites

```bash
pip install "manas-ai[openai]"  # Or your preferred provider
```

## Implementation

```python
import os
from core import Flow, LLM
from core.nodes import QANode, RAGConfig

# Initialize LLM (using OpenAI as example)
model = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create nodes with specific roles
researcher = QANode(
    name="researcher",
    llm=model,
    config=RAGConfig(
        system_prompt=(
            "You are an expert researcher. Investigate the given topic thoroughly "
            "and provide comprehensive, factual information. Include relevant "
            "details, statistics, and context."
        ),
        use_rag=True  # Enable RAG for knowledge enhancement
    )
)

analyst = QANode(
    name="analyst",
    llm=model,
    config=RAGConfig(
        system_prompt=(
            "You are a data analyst specializing in finding patterns and insights. "
            "Analyze the research provided and identify key trends, implications, "
            "and notable findings. Be objective and thorough in your analysis."
        )
    )
)

writer = QANode(
    name="writer",
    llm=model,
    config=RAGConfig(
        system_prompt=(
            "You are a technical writer. Create a well-structured report based on "
            "the analysis provided. Use clear headings, concise language, and "
            "highlight key points. Format your response using Markdown."
        )
    )
)

# Create and configure the flow
flow = Flow()

# Add nodes
flow.add_node(researcher)
flow.add_node(analyst)
flow.add_node(writer)

# Connect nodes in sequence
flow.add_edge(researcher, analyst)
flow.add_edge(analyst, writer)

# Optional: Add knowledge base for RAG
if researcher.rag_node:
    researcher.rag_node.add_documents([
        "path/to/research_papers/",
        "path/to/reference_docs/"
    ])

# Process a research query
topic = "Recent advances in quantum computing and their implications"
result = await flow.process({
    "prompt": topic,
    "max_sources": 5  # Limit number of sources to include
})

print("\nResearch Report:")
print(result["writer"]["response"])
```

## Understanding the Flow

### Node Communication

1. The researcher node receives the initial query and:
   - Uses RAG to enhance its knowledge
   - Generates comprehensive research findings
   - Includes relevant sources and citations

2. The analyst node:
   - Receives the research output
   - Identifies key patterns and insights
   - Structures the information logically

3. The writer node:
   - Takes the analyzed information
   - Creates a well-formatted report
   - Uses Markdown for clear presentation

### Customization Options

You can customize the flow by:

- Adjusting system prompts for different writing styles
- Configuring RAG parameters for knowledge retrieval
- Adding more nodes for specific tasks (e.g., fact-checking)
- Modifying the flow structure for different workflows

## Error Handling

```python
try:
    result = await flow.process({"prompt": topic})
except Exception as e:
    print(f"Error in flow execution: {e}")
finally:
    # Always clean up resources
    await flow.cleanup()
```

## Complete Example

The complete example with error handling and advanced features is available in the [examples directory](https://github.com/arkokoley/manas/blob/main/examples/research_flow.py).