"""
Query Agent Tool for Weaviate Plugin

This module provides an intelligent query agent that interprets natural language
queries and translates them into appropriate Weaviate operations. It uses Large
Language Models (LLMs) to understand user intent and automatically execute the
corresponding database operations.

The query agent acts as a natural language interface to Weaviate, allowing users
to interact with their vector database using conversational queries instead of
technical API calls. It supports various operations including search, collection
management, and data retrieval.

Classes:
    QueryAgentTool: Main tool class for natural language query processing
"""

from collections.abc import Generator
from typing import Any, Dict, List
import json
import logging
import openai
import anthropic
import re

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_limit
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class QueryAgentTool(Tool):
    """
    An intelligent query agent that interprets natural language queries and executes
    corresponding Weaviate operations using Large Language Models.
    
    This tool provides a conversational interface to Weaviate by using LLMs to
    understand user intent and automatically translate natural language queries
    into appropriate database operations. It supports various operations including
    search, collection management, schema inspection, and data retrieval.
    
    The agent uses a three-step process:
    1. Query Interpretation: Uses LLM to analyze the natural language query
    2. Operation Execution: Performs the identified Weaviate operation
    3. Response Generation: Creates a natural language response with results
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Process natural language queries and execute corresponding Weaviate operations.
        
        This method serves as the main entry point for the query agent, handling
        the complete pipeline from natural language input to structured response.
        It interprets user queries, executes the appropriate Weaviate operations,
        and generates human-readable responses.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing query parameters
                - query (str): Natural language query to process (required)
                - collection_name (str): Target collection name (optional, can be inferred)
                - max_results (int): Maximum number of results to return (1-100, default: 10)
                - llm_provider (str): LLM provider to use - "openai" or "anthropic" (default: "openai")
                - llm_model (str): Specific LLM model to use (default: "gpt-3.5-turbo")
                - llm_api_key (str): API key for the LLM provider (required)
        
        Yields:
            ToolInvokeMessage: JSON messages containing processed results or error information
            
        Supported Operations:
            search: Search for documents using vector, keyword, or hybrid search
                - Automatically determines search type based on query context
                - Supports filtering and property selection
                
            list_collections: List all available collections
                - Returns collection names and metadata
                
            get_schema: Retrieve schema information for a collection
                - Shows property definitions and types
                
            get_stats: Get statistics for a collection
                - Returns document counts and collection metrics
                
            insert: Insert new documents (planned)
            update: Update existing documents (planned)
            delete: Delete documents (planned)
            get: Get specific document by ID (planned)
            
        Query Interpretation:
            The agent uses LLM-based interpretation to understand user intent:
            - Analyzes natural language for operation type
            - Extracts relevant parameters and filters
            - Determines appropriate search strategies
            - Handles ambiguous queries with fallback logic
            
        Response Generation:
            Creates natural language responses that:
            - Summarize search results in readable format
            - Provide relevant document excerpts
            - Include operation metadata and statistics
            - Handle empty results gracefully
            
        Error Handling:
            Comprehensive error handling for:
            - Invalid or missing parameters
            - LLM API failures
            - Weaviate connection issues
            - Operation execution errors
            
        Returns:
            Generator[ToolInvokeMessage]: Stream of JSON messages containing:
                - Natural language responses
                - Operation details and metadata
                - Search results and statistics
                - Error messages for failed operations
        """
        try:
            query = tool_parameters.get('query', '').strip()
            collection_name = tool_parameters.get('collection_name', '').strip()
            max_results = tool_parameters.get('max_results', 10)
            llm_provider = tool_parameters.get('llm_provider', 'openai').strip().lower()
            llm_model = tool_parameters.get('llm_model', 'gpt-3.5-turbo').strip()
            llm_api_key = tool_parameters.get('llm_api_key', '').strip()
            
            if not query:
                yield self.create_text_message("Error: Query is required")
                return
            
            if not validate_limit(max_results) or max_results > 100:
                yield self.create_text_message("Error: Max results must be between 1 and 100")
                return
            
            credentials = self.runtime.credentials
            client = WeaviateClient(
                url=credentials['url'],
                api_key=credentials.get('api_key'),
                timeout=60
            )
            
            try:
                # Interpret the query using LLM
                interpretation = self._interpret_query(
                    query=query,
                    collection_name=collection_name,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key
                )
                
                if not interpretation:
                    yield self.create_text_message("Error: Could not interpret the query")
                    return
                
                # Execute the interpreted operation
                result = self._execute_operation(
                    client=client,
                    interpretation=interpretation,
                    max_results=max_results
                )
                
                if result is None:
                    yield self.create_text_message("Error: Could not execute the operation")
                    return
                
                # Generate a natural language response
                response_text = self._generate_response(
                    query=query,
                    result=result,
                    interpretation=interpretation,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    llm_api_key=llm_api_key
                )
                
                response = create_success_response(
                    data={
                        'response': response_text,
                        'operation': interpretation.get('operation'),
                        'collection': interpretation.get('collection_name'),
                        'result_data': result
                    },
                    message="Query processed successfully"
                )
                
                yield self.create_json_message(response)
                
            except Exception as e:
                logger.error(f"Query agent error: {str(e)}")
                yield self.create_text_message(f"Query processing failed: {str(e)}")
            finally:
                client.disconnect()
                
        except Exception as e:
            logger.error(f"Tool execution error: {str(e)}")
            yield self.create_text_message(f"Tool execution failed: {str(e)}")
    
    def _interpret_query(self, query: str, collection_name: str, llm_provider: str, 
                        llm_model: str, llm_api_key: str) -> Dict[str, Any]:
        """
        Interpret natural language query using LLM to determine appropriate operation.
        
        This method uses a Large Language Model to analyze the user's natural language
        query and determine what Weaviate operation should be performed. It extracts
        relevant parameters and constructs a structured interpretation that can be
        executed by the operation execution engine.
        
        Parameters:
            query (str): The natural language query to interpret.
            collection_name (str): Target collection name (may be empty for inference).
            llm_provider (str): LLM provider to use ("openai" or "anthropic").
            llm_model (str): Specific LLM model to use.
            llm_api_key (str): API key for the LLM provider.
            
        Returns:
            Dict[str, Any]: Structured interpretation containing:
                - operation: Type of operation to perform
                - collection_name: Target collection name
                - search_type: Search method for search operations
                - search_query: Search terms for search operations
                - filters: Additional filters to apply
                - properties: Specific properties to return
                - document_data: Data for insert/update operations
                - document_id: ID for get/update/delete operations
                
        Supported Operations:
            search: Document search operations
                - search_type: "vector", "keyword", or "hybrid"
                - search_query: Extracted search terms
                - filters: Additional filtering criteria
                - properties: Properties to return
                
            list_collections: Collection listing
                - No additional parameters required
                
            get_schema: Schema retrieval
                - collection_name: Target collection
                
            get_stats: Statistics retrieval
                - collection_name: Target collection
                
        Fallback Logic:
            If LLM interpretation fails, falls back to pattern-based interpretation:
            - Searches for keywords like "search", "find", "look for"
            - Defaults to hybrid search for ambiguous queries
            - Uses collection name inference when possible
            
        Error Handling:
            - Handles LLM API failures gracefully
            - Falls back to pattern-based interpretation
            - Returns None for complete interpretation failures
        """
        try:
            prompt = f"""You are a Weaviate query agent. Analyze the following natural language query and determine what operation to perform.

Available operations:
1. search - Search for documents (vector, keyword, or hybrid)
2. list_collections - List all collections
3. get_schema - Get collection schema
4. get_stats - Get collection statistics
5. insert - Insert new documents
6. update - Update existing documents
7. delete - Delete documents
8. get - Get specific document by ID

Query: "{query}"
Collection: "{collection_name if collection_name else 'not specified'}"

Return a JSON object with:
- operation: the operation type
- collection_name: the collection to work with (infer if not specified)
- search_type: "vector", "keyword", or "hybrid" (for search operations)
- search_query: the search terms (for search operations)
- filters: any filters to apply
- properties: specific properties to return
- document_data: data for insert/update operations
- document_id: ID for get/update/delete operations

Example response:
{{"operation": "search", "collection_name": "Documents", "search_type": "hybrid", "search_query": "artificial intelligence", "filters": null, "properties": ["title", "content"]}}"""
            
            if llm_provider == "openai":
                client = openai.OpenAI(api_key=llm_api_key)
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                interpretation_text = response.choices[0].message.content.strip()
            elif llm_provider == "anthropic":
                client = anthropic.Anthropic(api_key=llm_api_key)
                response = client.messages.create(
                    model=llm_model,
                    max_tokens=500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                interpretation_text = response.content[0].text.strip()
            else:
                # Fallback interpretation
                return self._fallback_interpretation(query, collection_name)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', interpretation_text, re.DOTALL)
            if json_match:
                return safe_json_parse(json_match.group(), {})
            else:
                return self._fallback_interpretation(query, collection_name)
                
        except Exception as e:
            logger.error(f"Query interpretation error: {str(e)}")
            return self._fallback_interpretation(query, collection_name)
    
    def _fallback_interpretation(self, query: str, collection_name: str) -> Dict[str, Any]:
        """
        Provide fallback query interpretation using pattern matching.
        
        This method provides a backup interpretation mechanism when LLM-based
        interpretation fails. It uses simple keyword matching to determine
        the most likely operation type and parameters.
        
        Parameters:
            query (str): The natural language query to interpret.
            collection_name (str): Target collection name (may be empty).
            
        Returns:
            Dict[str, Any]: Basic interpretation structure with inferred operation.
            
        Pattern Matching:
            - "search", "find", "look for", "query" -> search operation
            - "list", "show", "get all" -> list_collections operation
            - Default -> search operation with hybrid search type
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['search', 'find', 'look for', 'query']):
            return {
                'operation': 'search',
                'collection_name': collection_name or 'Documents',
                'search_type': 'hybrid',
                'search_query': query,
                'filters': None,
                'properties': None
            }
        elif any(word in query_lower for word in ['list', 'show', 'get all']):
            return {
                'operation': 'list_collections',
                'collection_name': None,
                'search_type': None,
                'search_query': None,
                'filters': None,
                'properties': None
            }
        else:
            return {
                'operation': 'search',
                'collection_name': collection_name or 'Documents',
                'search_type': 'hybrid',
                'search_query': query,
                'filters': None,
                'properties': None
            }
    
    def _execute_operation(self, client: WeaviateClient, interpretation: Dict[str, Any], 
                          max_results: int) -> Any:
        """
        Execute the interpreted Weaviate operation.
        
        This method takes the structured interpretation from query analysis and
        executes the corresponding Weaviate operation using the provided client.
        It handles different operation types and maps them to appropriate
        Weaviate API calls.
        
        Parameters:
            client (WeaviateClient): Connected Weaviate client instance.
            interpretation (Dict[str, Any]): Structured operation interpretation.
            max_results (int): Maximum number of results to return.
            
        Returns:
            Any: Operation results or None if execution fails.
            
        Supported Operations:
            search: Performs document search using various methods
                - vector: Vector similarity search (with dummy vector fallback)
                - keyword: BM25 text search
                - hybrid: Combined vector and text search
                
            list_collections: Returns list of available collections
            get_schema: Returns collection schema information
            get_stats: Returns collection statistics
            
        Search Implementation:
            - Vector search uses dummy vector when no vector provided
            - Keyword search uses BM25 algorithm
            - Hybrid search combines both methods with configurable alpha
            - All search types support filtering and property selection
            
        Error Handling:
            - Catches and logs operation execution errors
            - Returns None for failed operations
            - Provides graceful degradation for unsupported operations
        """
        try:
            operation = interpretation.get('operation')
            collection_name = interpretation.get('collection_name')
            
            if operation == 'search':
                search_type = interpretation.get('search_type', 'hybrid')
                search_query = interpretation.get('search_query', '')
                filters = interpretation.get('filters')
                properties = interpretation.get('properties')
                
                if search_type == 'vector':
                    # This would need a vector, so fallback to hybrid
                    return client.hybrid_search(
                        class_name=collection_name,
                        query=search_query,
                        query_vector=[0.0] * 1536,  # Dummy vector
                        alpha=0.5,
                        limit=max_results,
                        where_filter=filters,
                        return_properties=properties
                    )
                elif search_type == 'keyword':
                    return client.text_search(
                        class_name=collection_name,
                        query=search_query,
                        limit=max_results,
                        where_filter=filters,
                        return_properties=properties
                    )
                else:  # hybrid
                    return client.hybrid_search(
                        class_name=collection_name,
                        query=search_query,
                        query_vector=[0.0] * 1536,  # Dummy vector
                        alpha=0.7,
                        limit=max_results,
                        where_filter=filters,
                        return_properties=properties
                    )
            
            elif operation == 'list_collections':
                return {'collections': client.list_collections()}
            
            elif operation == 'get_schema':
                schema = client.get_collection_schema(collection_name)
                return {'schema': schema} if schema else None
            
            elif operation == 'get_stats':
                stats = client.get_collection_stats(collection_name)
                return {'stats': stats} if stats else None
            
            else:
                return {'message': f'Operation {operation} not yet implemented in query agent'}
                
        except Exception as e:
            logger.error(f"Operation execution error: {str(e)}")
            return None
    
    def _generate_response(self, query: str, result: Any, interpretation: Dict[str, Any],
                          llm_provider: str, llm_model: str, llm_api_key: str) -> str:
        """
        Generate natural language response from operation results.
        
        This method creates human-readable responses based on the executed
        operation and its results. It formats different types of results
        appropriately and provides context about what was found or performed.
        
        Parameters:
            query (str): Original user query for context.
            result (Any): Results from the executed operation.
            interpretation (Dict[str, Any]): Original operation interpretation.
            llm_provider (str): LLM provider name (currently unused).
            llm_model (str): LLM model name (currently unused).
            llm_api_key (str): LLM API key (currently unused).
            
        Returns:
            str: Natural language response describing the results.
            
        Response Types:
            search: Formats search results with document summaries
                - Shows document count and titles
                - Includes content excerpts for relevant documents
                - Handles empty results gracefully
                
            list_collections: Lists available collections
                - Shows collection names and counts
                - Handles empty collection lists
                
            get_schema: Confirms schema retrieval
                - Provides collection name context
                - Handles schema retrieval failures
                
            get_stats: Shows collection statistics
                - Displays document counts
                - Provides collection metrics
                
        Content Formatting:
            - Limits document excerpts to 200 characters
            - Shows first 5 results with overflow indication
            - Uses consistent formatting across response types
            - Provides meaningful context for each operation type
            
        Error Handling:
            - Handles missing or malformed results
            - Provides fallback responses for errors
            - Logs response generation errors
        """
        try:
            operation = interpretation.get('operation')
            
            if operation == 'search' and isinstance(result, list):
                if not result:
                    return f"I searched for '{query}' but didn't find any matching documents."
                
                response = f"I found {len(result)} documents matching '{query}':\n\n"
                for i, doc in enumerate(result[:5], 1):  # Show first 5
                    properties = doc.get('properties', {})
                    title = properties.get('title', properties.get('name', f'Document {i}'))
                    response += f"{i}. {title}\n"
                    if 'content' in properties:
                        content = properties['content'][:200] + "..." if len(properties['content']) > 200 else properties['content']
                        response += f"   {content}\n\n"
                
                if len(result) > 5:
                    response += f"... and {len(result) - 5} more documents."
                
                return response
            
            elif operation == 'list_collections':
                collections = result.get('collections', [])
                if not collections:
                    return "No collections found in your Weaviate instance."
                return f"I found {len(collections)} collections: {', '.join(collections)}"
            
            elif operation == 'get_schema':
                schema = result.get('schema')
                if not schema:
                    return f"Could not retrieve schema for collection '{interpretation.get('collection_name')}'."
                return f"Retrieved schema for collection '{interpretation.get('collection_name')}'."
            
            elif operation == 'get_stats':
                stats = result.get('stats')
                if not stats:
                    return f"Could not retrieve statistics for collection '{interpretation.get('collection_name')}'."
                count = stats.get('total_count', 0)
                return f"Collection '{interpretation.get('collection_name')}' contains {count} documents."
            
            else:
                return f"Processed your query: '{query}' using operation '{operation}'."
                
        except Exception as e:
            logger.error(f"Response generation error: {str(e)}")
            return f"Processed your query: '{query}' but encountered an error generating the response."