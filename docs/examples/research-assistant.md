---
layout: tutorial
title: Creating a Research Flow with Specialized Nodes
nav_order: 1
parent: Examples
permalink: /examples/research-assistant/
difficulty: Intermediate
time: 20 minutes
---

# Building a Research Assistant with Specialized Nodes

This example demonstrates how to create a multi-node flow that acts as a research assistant, using specialized nodes for different tasks in the research process.

## Objective

Build a research assistant flow that:
1. Researches a given topic (Researcher Node)
2. Analyzes the findings (Analyst Node) 
3. Generates a summary report (Writer Node)

## Prerequisites

```bash
pip install "manas[all-cpu]"
```

## Implementation

```python
import os
from core import Flow, LLM
from manas_ai.nodes import QANode

# Initialize LLM (use your preferred provider)
llm = LLM.from_provider(
    "openai",
    model_name="gpt-4",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Create specialized nodes with different roles
researcher_node = QANode(
    name="researcher",
    llm=llm,
    system_prompt=(
        "You are an expert researcher. Your job is to thoroughly investigate "
        "the given topic and provide comprehensive, factual information. "
        "Include relevant details, statistics, and context."
    )
)

analyst_node = QANode(
    name="analyst",
    llm=llm,
    system_prompt=(
        "You are a data analyst specializing in finding patterns and insights. "
        "Analyze the research provided to you and identify key trends, implications, "
        "and notable findings. Be objective and thorough."
    )
)

writer_node = QANode(
    name="writer",
    llm=llm,
    system_prompt=(
        "You are a skilled technical writer. Create a well-structured summary report "
        "based on the analysis provided. Use clear headings, concise language, and "
        "highlight the most important points. Format your response using Markdown."
    )
)

# Create a flow and connect the nodes
research_flow = Flow()
research_flow.add_node(researcher_node)
research_flow.add_node(analyst_node)
research_flow.add_node(writer_node)

# Connect nodes in sequence
research_flow.add_edge(researcher_node, analyst_node)
research_flow.add_edge(analyst_node, writer_node)

# Process a query through the flow
research_topic = "The impact of artificial intelligence on healthcare diagnostics"
result = research_flow.process(research_topic)

print(f"Research Report on: {research_topic}\n")
print(result)
```

## Explanation

This example demonstrates how to create a multi-node flow with specialized nodes:

1. **Researcher Node**: This node is responsible for gathering information on the topic. It uses a system prompt that focuses on thorough research and comprehensive information gathering.

2. **Analyst Node**: This node receives the research from the first node and analyzes it to identify key insights, patterns, and implications.

3. **Writer Node**: This node takes the analysis and formats it into a well-structured final report with proper formatting and organization.

The flow connects these nodes in sequence, creating a pipeline where the output of each node becomes the input for the next.

## Visualization of the Flow

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Researcher  │──────► Analyst     │──────► Writer      │
│ Node        │      │ Node        │      │ Node        │
└─────────────┘      └─────────────┘      └─────────────┘
```

## Variations

### Adding a Fact-Checker Node

For added reliability, you could add a fact-checking node between the researcher and analyst:

```python
fact_checker_node = QANode(
    name="fact_checker",
    llm=llm,
    system_prompt=(
        "You are a meticulous fact-checker. Review the research provided "
        "and verify key claims. Flag any potential inaccuracies or "
        "unsubstantiated claims. Provide corrected information where possible."
    )
)

# Add to flow
research_flow.add_node(fact_checker_node)
research_flow.add_edge(researcher_node, fact_checker_node)
research_flow.add_edge(fact_checker_node, analyst_node)
```

### Using a Document Node for External Sources

To incorporate external data sources, you could use a DocumentNode:

```python
from manas_ai.nodes import DocumentNode

document_node = DocumentNode(
    name="document_processor",
    llm=llm,
    system_prompt=(
        "Extract and summarize the key information from the provided documents "
        "that relates to the research topic."
    )
)

# Add relevant documents
document_node.add_document("healthcare_ai_research_paper.pdf")

# Add to flow (parallel to researcher)
research_flow.add_node(document_node)
research_flow.add_edge(document_node, analyst_node)
```

## Complete Example

You can find the complete example in the [examples directory](https://github.com/arkokoley/manas/blob/main/examples/research_assistant.py) of the Manas repository.