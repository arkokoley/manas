---
title: Flow
description: Flow orchestration and execution engine
parent: API Reference
---

# Flow

The `Flow` class manages the orchestration and execution of nodes in a directed graph structure. It handles node dependencies, parallel execution, and data flow between nodes.

## Import

```python
from core import Flow
from core.base import Edge
```

## Constructor

```python
def __init__(
    self,
    name: Optional[str] = None,
    parallel_execution: bool = False,
    max_concurrency: Optional[int] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | Optional[str] | None | Flow name for identification |
| parallel_execution | bool | False | Enable parallel execution of independent nodes |
| max_concurrency | Optional[int] | None | Maximum number of concurrent node executions |

## Core Methods

### add_node

```python
def add_node(self, node: Node) -> str
```

Add a node to the flow. Returns node ID.

### add_edge

```python
def add_edge(
    self,
    source_node: Union[str, Node],
    target_node: Union[str, Node],
    name: Optional[str] = None
) -> Edge
```

Connect two nodes with a directed edge.

### process

```python
async def process(
    self,
    inputs: Dict[str, Any],
    node_inputs: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]
```

Process inputs through the flow graph.

| Parameter | Type | Description |
|-----------|------|-------------|
| inputs | Dict[str, Any] | Global inputs for the flow |
| node_inputs | Optional[Dict[str, Dict[str, Any]]] | Node-specific inputs |

Returns results from all nodes.

### get_node

```python
def get_node(self, node_id: str) -> Node
```

Get a node by its ID.

### validate

```python
def validate(self) -> bool
```

Validate flow graph structure.

## Properties

### nodes

```python
@property
def nodes(self) -> Dict[str, Node]
```

Get all nodes in the flow.

### edges

```python
@property
def edges(self) -> List[Edge]
```

Get all edges in the flow.

## Example Usage

### Simple Linear Flow

```python
# Create nodes
qa_node = QANode(name="qa", config=qa_config)
summary_node = QANode(name="summary", config=summary_config)

# Create flow
flow = Flow(name="qa_flow")

# Add nodes
qa_id = flow.add_node(qa_node)
summary_id = flow.add_node(summary_node)

# Connect nodes
flow.add_edge(qa_id, summary_id)

# Process
result = await flow.process({
    "question": "Explain quantum computing",
    "summary_prompt": lambda qa_result: f"Summarize: {qa_result['answer']}"
})
```

### Parallel Flow

```python
# Create parallel flow
flow = Flow(
    name="parallel_research",
    parallel_execution=True,
    max_concurrency=3
)

# Add independent research nodes
physics = QANode(name="physics", config=physics_config)
chemistry = QANode(name="chemistry", config=chemistry_config)
biology = QANode(name="biology", config=biology_config)

# Add analyzer node
analyzer = QANode(name="analyzer", config=analyzer_config)

# Add nodes
p_id = flow.add_node(physics)
c_id = flow.add_node(chemistry)
b_id = flow.add_node(biology)
a_id = flow.add_node(analyzer)

# Connect nodes
for node_id in [p_id, c_id, b_id]:
    flow.add_edge(node_id, a_id)

# Process - physics, chemistry, and biology will run in parallel
result = await flow.process({
    "topic": "energy transformation in nature"
})
```

### Dynamic Flow Building

```python
from typing import List

def create_research_flow(topics: List[str]) -> Flow:
    """Create a dynamic research flow based on topics."""
    flow = Flow(parallel_execution=True)
    
    # Create researcher nodes for each topic
    researcher_ids = []
    for topic in topics:
        researcher = QANode(
            name=f"researcher_{topic}",
            config=QAConfig(
                system_prompt=f"Research {topic} thoroughly"
            )
        )
        researcher_ids.append(flow.add_node(researcher))
    
    # Create analyzer node
    analyzer = QANode(
        name="analyzer",
        config=QAConfig(
            system_prompt="Synthesize research findings"
        )
    )
    analyzer_id = flow.add_node(analyzer)
    
    # Connect researchers to analyzer
    for r_id in researcher_ids:
        flow.add_edge(r_id, analyzer_id)
    
    return flow

# Usage
topics = ["quantum_computing", "machine_learning", "robotics"]
flow = create_research_flow(topics)
result = await flow.process({"depth": "comprehensive"})
```

## Best Practices

1. **Graph Structure**
   - Keep flows as shallow as possible
   - Use parallel execution for independent nodes
   - Validate flows before execution

2. **Resource Management**
   - Initialize all nodes before flow execution
   - Clean up nodes after flow completion
   - Handle errors at flow level

3. **Input Management**
   - Use node_inputs for node-specific parameters
   - Pass global context through main inputs
   - Use lambda functions for dynamic inputs

4. **Optimization**
   - Enable parallel_execution when appropriate
   - Set reasonable max_concurrency limits
   - Monitor flow performance

## Notes

- Flow validates graph structure automatically
- Cycles are not allowed in the graph
- Node execution order is determined by dependencies
- Results include outputs from all nodes
- Errors in any node will stop flow execution
- Clean up nodes after flow completion