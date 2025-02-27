---
layout: default
title: Contributing to Manas
nav_order: 9
permalink: /contributing/
has_toc: true
---

# Contributing to Manas

Thank you for your interest in contributing to Manas! This guide will help you get started with contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11 or newer
- Poetry (for dependency management)
- Git
- A compatible IDE (we recommend VS Code with Python extensions)

### Setting Up Your Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/manas.git
   cd manas
   ```

2. **Install Dependencies**
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install project dependencies
   poetry install

   # Install development extras
   poetry install --extras "test docs"
   ```

3. **Set Up Pre-commit Hooks**
   ```bash
   # Install pre-commit
   poetry run pre-commit install
   ```

## Code Style Guidelines

We follow standard Python coding conventions with some specific requirements:

### Python Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Maximum line length is 88 characters (Black formatter default)
- Use docstrings for all public modules, functions, classes, and methods

Example:
```python
from typing import List, Optional

def process_items(items: List[str], max_items: Optional[int] = None) -> List[str]:
    """Process a list of items with optional limit.
    
    Args:
        items: List of strings to process
        max_items: Maximum number of items to process
        
    Returns:
        List of processed items
        
    Raises:
        ValueError: If max_items is negative
    """
    if max_items is not None and max_items < 0:
        raise ValueError("max_items cannot be negative")
    
    return items[:max_items]
```

### Documentation

- All new features must include documentation
- Update relevant examples when adding new functionality
- Include docstrings with type hints, parameters, returns, and raises sections
- Add examples for complex features

### Testing

- Write unit tests for all new features
- Include integration tests for complex functionality
- Ensure all tests pass before submitting a PR
- Maintain or improve code coverage

Example test:
```python
import pytest
from core import LLM, Agent

async def test_agent_creation():
    """Test basic agent creation and functionality."""
    model = LLM.from_provider("mock")  # Use mock provider for testing
    agent = Agent(llm=model)
    
    response = await agent.generate("Test query")
    assert isinstance(response, str)
    assert len(response) > 0
```

## Making Contributions

### Types of Contributions

1. **Bug Fixes**
   - Identify bugs through issues
   - Write failing test case
   - Fix the bug
   - Ensure all tests pass

2. **New Features**
   - Discuss new features in issues first
   - Create detailed proposal if needed
   - Implement feature with tests
   - Update documentation

3. **Documentation**
   - Fix typos and unclear sections
   - Add examples and tutorials
   - Improve API documentation
   - Update guides and FAQs

4. **Performance Improvements**
   - Profile the code
   - Propose optimizations
   - Include benchmarks
   - Document improvements

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add/update tests
   - Update documentation
   - Run linters and formatters

3. **Commit Changes**
   ```bash
   # Run pre-commit hooks
   poetry run pre-commit run --all-files
   
   # Commit with descriptive message
   git commit -m "feat: add new feature x
   
   Detailed description of changes"
   ```

4. **Open Pull Request**
   - Use PR template
   - Link related issues
   - Add description of changes
   - Request review

### CI/CD Pipeline

Our CI pipeline checks:

1. **Code Quality**
   - Black formatting
   - isort import sorting
   - Pylint linting
   - Type checking with mypy

2. **Tests**
   - Unit tests
   - Integration tests
   - Coverage reports

3. **Documentation**
   - Build docs
   - Link checking
   - Example validation

## Project Structure

```
manas/
├── core/              # Core framework code
│   ├── agent.py      # Agent implementation
│   ├── flow.py       # Flow orchestration
│   ├── llm.py        # LLM interface
│   ├── rag.py        # RAG implementation
│   └── ...
├── docs/             # Documentation
├── examples/         # Example code
├── tests/            # Test suite
└── tools/            # Development tools
```

## Release Process

1. **Version Bump**
   - Update version in pyproject.toml
   - Update CHANGELOG.md
   - Create release notes

2. **Testing**
   - Run full test suite
   - Perform integration testing
   - Check documentation

3. **Release**
   - Create GitHub release
   - Publish to PyPI
   - Update documentation

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality
- **PATCH** version for bug fixes

## Getting Help

- **Questions**: Use GitHub Discussions
- **Bugs**: Open GitHub Issues
- **Feature Requests**: Open GitHub Issues
- **Security Issues**: See SECURITY.md

## Code of Conduct

We follow a standard code of conduct to ensure a welcoming community. Key points:

1. **Be Respectful**
   - Use inclusive language
   - Accept constructive criticism
   - Focus on what's best for the community

2. **Be Professional**
   - Keep discussions focused
   - Avoid personal attacks
   - Respect differing viewpoints

3. **Be Collaborative**
   - Help others learn
   - Share knowledge
   - Work together

## Recognition

Contributors are recognized in several ways:

1. **Contributors List**
   - Added to AUTHORS.md
   - Mentioned in release notes
   - GitHub contributors page

2. **Special Recognition**
   - Significant contributions
   - Long-term maintenance
   - Documentation improvements

## License

By contributing to Manas, you agree that your contributions will be licensed under the project's MIT License.