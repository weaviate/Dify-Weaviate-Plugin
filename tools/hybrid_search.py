"""
Hybrid Search Tool for Weaviate Plugin

This module provides a hybrid search tool that combines keyword-based BM25 search
with vector similarity search in Weaviate. It allows users to leverage both
textual relevance and semantic similarity for more comprehensive search results.

The hybrid search combines the strengths of both search methods:
- BM25: Excellent for exact keyword matches and term frequency
- Vector Search: Excellent for semantic similarity and conceptual understanding

Classes:
    HybridSearchTool: Main tool class for hybrid search operations
"""

from collections.abc import Generator
from typing import Any
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_limit, validate_where_filter, validate_alpha
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

class HybridSearchTool(Tool):
    """
    A hybrid search tool that combines BM25 keyword search with vector similarity search.
    
    This tool provides a unified interface for performing hybrid searches in Weaviate,
    which combines the precision of keyword-based BM25 search with the semantic
    understanding of vector similarity search. The alpha parameter controls the
    weighting between the two search methods.
    
    The hybrid search is particularly useful when you want to find documents that are
    both semantically similar to your query and contain relevant keywords. This
    approach often provides more comprehensive and relevant results than either
    search method alone.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute hybrid search combining BM25 and vector search methods.
        
        This method performs a hybrid search that combines keyword-based BM25 search
        with vector similarity search. The alpha parameter controls the relative
        weighting between the two search methods, allowing fine-tuning of the
        search behavior based on the use case.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing search parameters
                - collection_name (str): Name of the Weaviate collection to search (required)
                - query (str): Text query for BM25 search component (optional if query_vector provided)
                - query_vector (str/list): Vector for similarity search component (optional if query provided)
                    - String: JSON array string or comma-separated values
                    - List: Direct list of numbers
                - alpha (float): Weighting factor between BM25 and vector search (0.0-1.0, default: 0.7)
                    - 0.0: Pure vector search
                    - 1.0: Pure BM25 search
                    - 0.7: Balanced hybrid (default)
                - limit (int): Maximum number of results to return (1-1000, default: 10)
                - where_filter (str): JSON string containing filter criteria for document filtering
                - return_properties (str): Comma-separated list of properties to return from results
        
        Yields:
            ToolInvokeMessage: JSON messages containing search results or error information
            
        Search Behavior:
            The hybrid search combines two scoring methods:
            1. BM25 Score: Based on term frequency and document frequency
            2. Vector Similarity Score: Based on cosine similarity between query and document vectors
            
            Final Score = alpha * BM25_Score + (1 - alpha) * Vector_Score
            
            This allows fine-tuning the balance between keyword matching and semantic similarity.
            
        Input Validation:
            - Collection name is required
            - At least one of query or query_vector must be provided
            - Alpha must be between 0.0 and 1.0
            - Limit must be between 1 and 1000
            - Query vector must be valid numeric array
            - Where filter must be valid JSON if provided
            
        Error Handling:
            Comprehensive validation for all input parameters with descriptive error messages.
            Handles connection errors and search failures gracefully.
            
        Returns:
            Generator[ToolInvokeMessage]: Stream of JSON messages containing:
                - Search results with documents and metadata
                - Result count and search parameters
                - Error messages for validation failures
        """
        try:
            collection_name = (tool_parameters.get('collection_name') or '').strip()
            query = (tool_parameters.get('query') or '').strip()
            where_filter_str = (tool_parameters.get('where_filter') or '').strip()
            return_properties_str = (tool_parameters.get('return_properties') or '').strip()

            if not collection_name:
                yield self.create_text_message("Error: Collection name is required")
                return

            # Check that at least one of query or query_vector is provided
            qv_raw = tool_parameters.get('query_vector', '')
            if isinstance(qv_raw, str):
                qv_raw = qv_raw.strip()
            
            if not query and not qv_raw:
                yield self.create_text_message("Error: Either query text or query vector is required")
                return

            # alpha - safe cast from string
            alpha_raw = tool_parameters.get('alpha', 0.7)
            try:
                alpha = float(alpha_raw)
            except (TypeError, ValueError):
                yield self.create_text_message("Error: Alpha must be a number between 0.0 and 1.0")
                return
            if not validate_alpha(alpha):
                yield self.create_text_message("Error: Alpha must be between 0.0 and 1.0")
                return

            # limit - safe cast from string
            limit_raw = tool_parameters.get('limit', 10)
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_text_message("Error: Limit must be an integer between 1 and 1000")
                return
            if not validate_limit(limit):
                yield self.create_text_message("Error: Limit must be between 1 and 1000")
                return

            # query_vector - accept JSON array OR CSV string
            query_vector = None
            if qv_raw:
                if qv_raw.startswith('['):
                    query_vector = safe_json_parse(qv_raw)
                else:
                    try:
                        query_vector = [float(x.strip()) for x in qv_raw.split(',') if x.strip()]
                    except ValueError:
                        yield self.create_text_message("Error: Invalid query vector. Use JSON array or comma-separated numbers")
                        return
                if not validate_vector(query_vector):
                    yield self.create_text_message("Error: Query vector must be a non-empty list of numbers")
                    return

            where_filter = None
            if where_filter_str:
                where_filter = safe_json_parse(where_filter_str)
                if where_filter is None or not validate_where_filter(where_filter):
                    yield self.create_text_message("Error: Invalid where filter format. Use valid JSON")
                    return

            return_properties = None
            if return_properties_str:
                return_properties = [p.strip() for p in return_properties_str.split(',') if p.strip()]

            # credentials
            creds = self.runtime.credentials
            client = WeaviateClient(url=creds['url'], api_key=creds.get('api_key'), timeout=60)

            try:
                results = client.hybrid_search(
                    class_name=collection_name,
                    query=query,
                    query_vector=query_vector,
                    alpha=alpha,
                    limit=limit,
                    where_filter=where_filter,
                    return_properties=return_properties
                )

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={'results': [], 'count': 0, 'collection': collection_name, 'alpha': alpha, 'query': query},
                        message="No results found"
                    ))
                    return

                yield self.create_json_message(create_success_response(
                    data={'results': results, 'count': len(results), 'collection': collection_name, 'alpha': alpha, 'query': query},
                    message=f"Found {len(results)} results using hybrid search"
                ))

            except Exception as e:
                logger.exception("Hybrid search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))

            finally:
                client.disconnect()

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))