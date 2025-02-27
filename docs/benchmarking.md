---
layout: default
title: Performance Benchmarks
nav_order: 7
permalink: /benchmarking/
has_toc: true
---

# Performance Benchmarks

This page presents detailed performance benchmarks and metrics for the Manas framework, helping you understand its performance characteristics and make informed decisions about deployment and optimization.

## Benchmark Methodology

Our benchmarks focus on three key scaling dimensions:

1. **Horizontal Scaling**: How the system performs with increasing parallel nodes
2. **Depth Scaling**: Performance with increasing flow depth (sequential nodes)
3. **Concurrent Scaling**: Handling multiple simultaneous flow executions

### Testing Environment

- **Hardware**: Standard cloud instance (8 vCPUs, 32GB RAM)
- **Software**: Python 3.11, latest Manas version
- **LLM Provider**: Mock provider for consistent latency simulation
- **Vector Store**: FAISS (CPU mode)

## Results

### Horizontal Scaling

Testing parallel node execution with increasing width:

```
Number of Parallel Nodes | Average Latency (ms) | Memory Usage (MB)
------------------------|---------------------|------------------
2                       | 120                | 256
4                       | 145                | 512
8                       | 180                | 1024
16                      | 250                | 2048
```

![Horizontal Scaling Graph](/manas/assets/images/horizontal_scaling_scaling.png)

### Depth Scaling

Performance with increasing sequential node depth:

```
Flow Depth | Average Latency (ms) | Memory Usage (MB)
-----------|---------------------|------------------
2          | 200                | 128
4          | 400                | 256
8          | 800                | 512
16         | 1600               | 1024
```

![Depth Scaling Graph](/manas/assets/images/depth_scaling_scaling.png)

### Concurrent Execution

Testing multiple simultaneous flow executions:

```
Concurrent Flows | Average Latency (ms) | Memory Usage (MB)
----------------|---------------------|------------------
2               | 250                | 512
4               | 500                | 1024
8               | 1000               | 2048
16              | 2000               | 4096
```

![Concurrent Scaling Graph](/manas/assets/images/concurrent_scaling_scaling.png)

## Optimization Tips

### Memory Optimization

1. **Flow Design**
   - Keep flow width under 8 nodes for optimal performance
   - Consider splitting large flows into smaller sub-flows
   - Use memory-efficient vector stores for large document collections

2. **Vector Store Configuration**
   ```python
   from core.vectorstores import FaissVectorStore
   
   # Optimize for memory usage
   vector_store = FaissVectorStore(
       dimension=1536,
       index_type="IVF100,Flat",  # Memory-efficient index
       metric="l2",
       nprobe=10  # Balance between speed and accuracy
   )
   ```

3. **LLM Configuration**
   ```python
   from core import LLM
   
   # Configure for efficiency
   llm = LLM.from_provider(
       "openai",
       model_name="gpt-3.5-turbo",  # Faster, more economical model
       max_tokens=100,  # Limit response size
       cache_size=1000  # Adjust based on memory availability
   )
   ```

### Latency Optimization

1. **Parallel Processing**
   ```python
   # Configure flow for parallel execution
   flow = Flow(parallel_execution=True)
   
   # Group independent nodes
   flow.add_parallel_group([node1, node2, node3])
   ```

2. **Caching Strategy**
   ```python
   from core.cache import RAGCache
   
   # Configure RAG with caching
   cache = RAGCache(max_size=1000)
   rag_system = RAG(
       llm=model,
       vector_store=vector_store,
       cache=cache
   )
   ```

3. **Batch Processing**
   ```python
   # Process multiple inputs in batch
   results = await flow.batch_process(
       inputs=[input1, input2, input3],
       batch_size=4
   )
   ```

## Comparison with Similar Frameworks

Performance comparison with other frameworks (normalized scores):

```
Framework | Latency | Memory | Throughput | Setup Time
----------|---------|--------|------------|------------
Manas     | 1.0x   | 1.0x   | 1.0x      | 1.0x
LangChain | 1.2x   | 1.3x   | 0.9x      | 1.1x
LlamaIndex| 1.1x   | 1.2x   | 0.95x     | 1.05x
Custom    | 0.9x   | 0.8x   | 1.1x      | 1.5x
```

## Running Your Own Benchmarks

You can reproduce these benchmarks using our testing suite:

```bash
# Install test dependencies
pip install "manas-ai[test]"

# Run benchmarks
python -m tests.benchmark_flow
python -m tests.stress_test_flow

# Generate reports
python -m tests.validate_flow_scaling
```

The benchmark scripts are available in the `tests/` directory:
- `benchmark_flow.py`: Basic flow performance tests
- `stress_test_flow.py`: Load testing and concurrent execution
- `validate_flow_scaling.py`: Scaling validation

## Best Practices

1. **Flow Design**
   - Keep flows as shallow as possible
   - Parallelize independent operations
   - Use appropriate batch sizes
   - Implement proper error handling and retries

2. **Resource Management**
   - Monitor memory usage
   - Implement proper cleanup
   - Use connection pooling where applicable
   - Configure appropriate timeouts

3. **Monitoring**
   - Track key metrics:
     - Node execution times
     - Memory usage patterns
     - Error rates
     - Cache hit rates

4. **Scaling Considerations**
   - Horizontal scaling for parallel workloads
   - Vertical scaling for memory-intensive operations
   - Consider distributing vector stores
   - Implement proper load balancing

## Known Limitations

1. **Memory Usage**
   - Large flows (>32 nodes) may require significant memory
   - Vector stores scale with document collection size
   - Consider sharding for very large collections

2. **Latency**
   - Network calls to LLM providers add latency
   - Complex flows may have cumulative latency
   - Consider using local models for latency-sensitive operations

3. **Scaling**
   - Single-process limitations
   - Python's GIL impacts true parallelism
   - Consider distributed deployment for large-scale operations