"""
Vector Search Tool for Weaviate Plugin

This module provides a vector similarity search tool that enables users to find
documents based on semantic similarity using vector embeddings. It uses cosine
similarity to rank documents by their vector distance from the query vector.

The vector search tool is ideal for finding semantically similar documents,
implementing recommendation systems, and performing similarity-based retrieval
in vector databases. It supports filtering and property selection for refined
search results.

Classes:
    VectorSearchTool: Main tool class for vector similarity search operations
"""

from collections.abc import Generator
from typing import Any
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class VectorSearchTool(Tool):
    """
    A vector similarity search tool that finds documents based on semantic similarity.
    
    This tool provides a straightforward interface for performing vector similarity
    searches in Weaviate collections. It uses cosine similarity to rank documents
    by their vector distance from the provided query vector, enabling semantic
    search capabilities.
    
    The tool is particularly useful for finding documents that are semantically
    similar to a given query vector, implementing recommendation systems, and
    performing similarity-based retrieval in vector databases.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute vector similarity search using provided query vector.
        
        This method performs a vector similarity search on the specified Weaviate
        collection using the provided query vector. It finds documents that are
        semantically similar to the query vector and returns them ranked by
        similarity score.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing search parameters
                - collection_name (str): Name of the Weaviate collection to search (required)
                - query_vector (str): Query vector for similarity search (required)
                    - String: JSON array string or comma-separated values
                    - Must be a non-empty list of numbers
                - limit (int): Maximum number of results to return (1-1000, default: 10)
                - where_filter (str): JSON string containing filter criteria for document filtering (optional)
                - return_properties (str): Comma-separated list of properties to return from results (optional)
        
        Yields:
            ToolInvokeMessage: JSON messages containing search results or error information
            
        Search Algorithm:
            The tool uses cosine similarity to rank documents:
            - Calculates cosine similarity between query vector and document vectors
            - Returns documents ranked by similarity score (highest first)
            - Supports filtering to narrow down search scope
            - Allows property selection to control returned data
            
        Vector Format Support:
            The tool accepts query vectors in multiple formats:
            - JSON array string: "[0.1, 0.2, 0.3, ...]"
            - Comma-separated values: "0.1, 0.2, 0.3, ..."
            - Both formats are automatically parsed and validated
            
        Property Filtering:
            - return_properties: Controls which document properties are included in results
            - where_filter: Applies additional filtering criteria using Weaviate's query language
            - Both parameters are optional and can be used together
            
        Input Validation:
            - Collection name is required
            - Query vector is required and must be valid numeric array
            - Limit must be between 1 and 1000
            - Where filter must be valid JSON if provided
            - Property lists are parsed and validated
            
        Error Handling:
            Comprehensive validation for all input parameters with descriptive error messages.
            Handles connection errors and search failures gracefully.
            
        Returns:
            Generator[ToolInvokeMessage]: Stream of JSON messages containing:
                - Search results with documents and similarity scores
                - Result count and collection information
                - Error messages for validation failures
        """
        try:
            collection_name = (tool_parameters.get('collection_name') or '').strip()
            qv_raw = (tool_parameters.get('query_vector') or '').strip()
            limit_raw = tool_parameters.get('limit', 10)
            where_filter_str = (tool_parameters.get('where_filter') or '').strip()
            return_properties_str = (tool_parameters.get('return_properties') or '').strip()

            if not collection_name:
                yield self.create_json_message(create_error_response("Collection name is required"))
                return
            if not qv_raw:
                yield self.create_json_message(create_error_response("Query vector is required"))
                return

            # Parse limit safely
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_json_message(create_error_response("Limit must be an integer between 1 and 1000"))
                return
            if not validate_limit(limit):
                yield self.create_json_message(create_error_response("Limit must be between 1 and 1000"))
                return

            # Accept JSON array OR CSV string for vectors
            if qv_raw.startswith('['):
                query_vector = safe_json_parse(qv_raw)
            else:
                try:
                    query_vector = [float(x.strip()) for x in qv_raw.split(',') if x.strip()]
                except ValueError:
                    yield self.create_json_message(create_error_response(
                        "Invalid query vector. Use a JSON array or comma-separated numbers"
                    ))
                    return

            if not validate_vector(query_vector):
                yield self.create_json_message(create_error_response(
                    "Query vector must be a non-empty list of numbers"
                ))
                return

            where_filter = None
            if where_filter_str:
                where_filter = safe_json_parse(where_filter_str)
                if where_filter is None or not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response(
                        "Invalid where filter format. Provide valid JSON"
                    ))
                    return

            return_properties = None
            if return_properties_str:
                return_properties = [p.strip() for p in return_properties_str.split(',') if p.strip()]

            # Connect
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds['url'],
                api_key=creds.get('api_key'),
                timeout=60
            )

            try:
                results = client.vector_search(
                    class_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    where_filter=where_filter,
                    return_properties=return_properties
                )

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={'results': [], 'count': 0, 'collection': collection_name},
                        message="No results found"
                    ))
                    return

                yield self.create_json_message(create_success_response(
                    data={'results': results, 'count': len(results), 'collection': collection_name},
                    message=f"Found {len(results)} similar vectors"
                ))

            except Exception as e:
                logger.exception("Vector search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))