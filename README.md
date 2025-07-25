# RAG Playground: Modular Testing Framework for Retrieval-Augmented Generation

## Overview

RAG Playground is a modular framework for testing and comparing different Retrieval-Augmented Generation (RAG) strategies on your personal knowledge base. The project provides a unified interface to experiment with various RAG techniques, making it easy to swap between strategies and compare their effectiveness on your data.

Currently implemented strategies:

- **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval): Builds hierarchical tree structures for multi-level retrieval
- **HyDE** (Hypothetical Document Embeddings): Generates hypothetical answers to improve retrieval accuracy

## Key Features

- **Modular Architecture**: Easily swap between different RAG strategies
- **Interactive Chat Interface**: Test strategies in real-time conversation
- **Configurable**: JSON-based configuration for each strategy
- **Extensible**: Simple interface to add new RAG strategies
- **Markdown Support**: Process Obsidian/Zettelkasten markdown files

## Project Structure

```
RAG_playground/
├── main.py                    # Entry point with CLI interface
├── strategies/
│   ├── __init__.py
│   ├── base.py               # Abstract base class for strategies
│   ├── raptor.py             # RAPTOR implementation
│   └── hyde.py               # HyDE implementation
├── utils/
│   ├── __init__.py
│   └── utils.py              # Shared utility functions
├── config/
│   ├── raptor_config.json    # RAPTOR configuration
│   └── hyde_config.json      # HyDE configuration
├── knowledge-vault/          # Your markdown knowledge base
│   └── nodes/               # Markdown files go here
└── text-files/              # Processed text files (auto-generated)
```

## Setup Instructions

### Prerequisites

- Python 3.8+ (Python 3.11 recommended)
- OpenAI API key

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd RAG_playground
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export LANGCHAIN_TRACING_V2=false  # Disable tracing by default
   ```

   Or create a `.env` file:

   ```
   OPENAI_API_KEY=your-api-key-here
   LANGCHAIN_TRACING_V2=false
   ```

5. **Prepare your knowledge base**
   - Place your markdown files in `knowledge-vault/nodes/`
   - Files should contain `#sprout` tag to be processed (configurable)

## Usage

### Interactive Chat Mode

Start an interactive chat session with your chosen strategy:

```bash
# Using RAPTOR strategy
python main.py --strategy raptor --source knowledge-vault/nodes

# Using HyDE strategy
python main.py --strategy hyde --source knowledge-vault/nodes
```

In interactive mode:

- Type your questions naturally
- View retrieved sources for each answer
- Type 'quit' or 'exit' to end the session

### Single Query Mode

Run a single query without entering interactive mode:

```bash
python main.py --strategy raptor --query "What is gradient descent?"
```

### Custom Configuration

Use custom configuration files:

```bash
python main.py --strategy raptor --config my_custom_config.json
```

## Configuration

Each strategy has its own configuration file in the `config/` directory.

### RAPTOR Configuration (`config/raptor_config.json`)

```json
{
  "chunk_size": 500, // Size of text chunks
  "chunk_overlap": 75, // Overlap between chunks
  "model_name": "gpt-3.5-turbo", // OpenAI model
  "model_temperature": 0, // Temperature for responses
  "max_clusters": 50, // Maximum clusters per level
  "recursive_depth": 3, // Tree depth
  "cluster_threshold": 0.5, // Similarity threshold
  "random_state": 1234 // For reproducibility
}
```

### HyDE Configuration (`config/hyde_config.json`)

```json
{
  "chunk_size": 500,
  "chunk_overlap": 75,
  "model_name": "gpt-3.5-turbo",
  "model_temperature": 0.7, // Higher for creative hypothetical docs
  "num_hypothetical_docs": 3, // Number of hypothetical documents
  "similarity_threshold": 0.5 // Minimum similarity for retrieval
}
```

## Strategies Explained

### RAPTOR (Recursive Abstractive Processing)

RAPTOR builds a hierarchical tree structure of your documents:

1. Splits documents into chunks
2. Clusters similar chunks
3. Summarizes each cluster
4. Recursively clusters summaries
5. Retrieves from multiple abstraction levels

**Best for**: Questions requiring hierarchical understanding or broader context

### HyDE (Hypothetical Document Embeddings)

HyDE improves retrieval through hypothetical document generation:

1. Generates hypothetical answers to your query
2. Finds documents similar to these hypothetical answers
3. Uses retrieved real documents to generate final answer

**Best for**: Technical queries where terminology might not match documents directly

## Adding New Strategies

To add a new RAG strategy:

1. Create a new file in `strategies/` (e.g., `strategies/my_strategy.py`)
2. Implement the `BaseRAGStrategy` interface:

   ```python
   from strategies.base import BaseRAGStrategy, RAGResponse

   class MyStrategy(BaseRAGStrategy):
       def _setup(self):
           # Initialize strategy-specific components
           pass

       def ingest_documents(self, source_path: str) -> None:
           # Process and index documents
           pass

       def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
           # Retrieve relevant documents/chunks
           pass

       def generate_response(self, query: str, context: List[Any]) -> RAGResponse:
           # Generate final response
           pass
   ```

3. Add to the strategy registry in `main.py`:

   ```python
   STRATEGIES = {
       'raptor': RAPTORStrategy,
       'hyde': HyDEStrategy,
       'my_strategy': MyStrategy,  # Add your strategy
   }
   ```

4. Create a configuration file in `config/my_strategy_config.json`

## Example Queries

Try these example queries to test different strategies:

```bash
# Technical concepts
"What is gradient descent optimization?"
"Explain backpropagation in neural networks"

# Hierarchical questions (good for RAPTOR)
"Give me an overview of machine learning techniques"
"What are the main principles of deep learning?"

# Specific terminology (good for HyDE)
"How do you minimize the loss function?"
"What are the applications of transformers?"
```

## Troubleshooting

### Common Issues

1. **Import errors with LangChain/LangSmith**

   ```bash
   export LANGCHAIN_TRACING_V2=false
   ```

2. **Missing libmagic warning**

   ```bash
   # macOS
   brew install libmagic

   # Or with pip
   pip install python-magic-bin
   ```

3. **Pydantic compatibility issues**
   ```bash
   pip install "pydantic>=1.10,<2.0"
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- RAPTOR paper: [arXiv:2401.18059v1](https://arxiv.org/html/2401.18059v1)
- HyDE paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- Built with [LangChain](https://github.com/langchain-ai/langchain) and [OpenAI](https://openai.com)
