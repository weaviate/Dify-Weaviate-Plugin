## Weaviate Plugin for Dify

A comprehensive Dify plugin that provides seamless integration with Weaviate vector database, enabling powerful vector search, data management, and text embedding capabilities.

## Features

### 🔍 Search & Query Tools
- **Vector Search**: Perform similarity search using vector embeddings
- **Hybrid Search**: Combine vector similarity and keyword search for comprehensive results
- **Keyword Search**: BM25-based keyword search for exact text matching
- **Generative Search**: RAG-powered search with LLM-generated responses
- **Query Agent**: Natural language query interface for intelligent operations
- **Data Management**: Full CRUD operations for Weaviate objects
- **Schema Management**: Create, delete, and manage collection schemas

### 🤖 Text Embedding Model
- **Multiple Vectorizers**: Support for OpenAI, Cohere, Hugging Face, and more
- **Configurable Dimensions**: Customize embedding vector dimensions
- **Flexible Models**: Use different embedding models based on your needs

## Installation

1. Configure your Weaviate instance credentials in Dify

## Configuration

### Provider Credentials
- **Weaviate URL**: Your Weaviate instance URL (required)
- **API Key**: Authentication key (optional for open instances)

### Text Embedding Model Parameters
- **Dimensions**: Number of vector dimensions (default: 1536)
- **Vectorizer**: Choose from text2vec-openai, text2vec-cohere, text2vec-huggingface, etc.
- **Model Name**: Specific model name for the selected vectorizer

## Usage

### Vector Search
Search for similar vectors in your collections:
```json
{
  "collection_name": "MyCollection",
  "query_vector": "0.1,0.2,0.3,...",
  "limit": 10,
  "where_filter": "{\"path\": [\"category\"], \"operator\": \"Equal\", \"valueText\": \"AI\"}"
}
```

### Hybrid Search
Combine vector and keyword search:
```json
{
  "collection_name": "MyCollection",
  "query": "artificial intelligence",
  "query_vector": "0.1,0.2,0.3,...",
  "alpha": 0.7,
  "limit": 10
}
```

### Keyword Search
Perform BM25 keyword search:
```json
{
  "collection_name": "MyCollection",
  "query": "machine learning algorithms",
  "limit": 10,
  "search_properties": "title,content"
}
```

### Generative Search
RAG-powered search with LLM responses:
```json
{
  "collection_name": "MyCollection",
  "query": "What are the benefits of AI?",
  "query_vector": "0.1,0.2,0.3,...",
  "limit": 5,
  "llm_provider": "openai",
  "llm_model": "gpt-3.5-turbo"
}
```

### Query Agent
Natural language query interface:
```json
{
  "query": "Show me all documents about machine learning",
  "collection_name": "MyCollection",
  "max_results": 10
}
```

### Data Management
Insert, update, delete, or retrieve objects:
```json
{
  "operation": "insert",
  "collection_name": "MyCollection",
  "object_data": "{\"text\": \"Hello World\", \"category\": \"greeting\"}"
}
```

### Schema Management
Create and manage collection schemas:
```json
{
  "operation": "create_collection",
  "collection_name": "MyCollection",
  "properties": "[{\"name\": \"text\", \"data_type\": \"TEXT\"}, {\"name\": \"category\", \"data_type\": \"TEXT\"}]"
}
```

## File Structure

```
weaviate_plugin/
├── _assets/                    # Plugin icons
├── models/
│   └── text_embedding/         # Text embedding model
├── provider/                   # Provider configuration
├── tools/                      # Tool implementations
│   ├── vector_search.py        # Vector similarity search
│   ├── hybrid_search.py        # Hybrid search
│   ├── keyword_search.py       # BM25 keyword search
│   ├── generative_search.py    # RAG-powered search
│   ├── query_agent.py          # Natural language query agent
│   ├── data_management.py      # CRUD operations
│   └── schema_management.py    # Schema operations
├── utils/                      # Utility functions
│   ├── client.py              # Weaviate client
│   ├── validators.py          # Input validation
│   └── helpers.py             # Helper functions
├── main.py                    # Plugin entry point
├── manifest.yaml              # Plugin manifest
└── requirements.txt           # Dependencies
```

## Supported Operations

### Vector Search
- Similarity search using vector embeddings
- Configurable result limits and filters
- Metadata and property selection

### Hybrid Search
- Combines vector similarity and keyword search
- Adjustable alpha parameter for weighting
- Advanced filtering capabilities

### Keyword Search
- BM25-based keyword matching
- Configurable search properties
- Exact text matching capabilities

### Generative Search
- RAG-powered search with LLM integration
- Context-aware response generation
- Support for multiple LLM providers

### Query Agent
- Natural language query interpretation
- Intelligent operation selection
- Conversational response generation

### Data Management
- Insert single or multiple objects
- Update existing objects by UUID
- Delete objects by UUID
- Retrieve objects with property selection
- List all collections

### Schema Management
- Create collections with custom properties
- Delete collections
- Retrieve collection schemas
- Get collection statistics
- List all collections

## Error Handling

The plugin includes comprehensive error handling for:
- Invalid credentials
- Network connectivity issues
- Malformed input data
- Weaviate API errors
- Validation failures

## Development

To extend the plugin:

1. Add new tools in the `tools/` directory
2. Create corresponding YAML configurations
3. Register tools in `main.py`
4. Update the provider YAML to include new tools

## License

This plugin is provided by Weaviate for integration with Dify.

**Author:** weaviate
**Version:** 0.0.1
**Type:** tool

## Repository

Source code: https://github.com/weaviate/Dify-Weaviate-Plugin