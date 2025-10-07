"""
Keyword Search Tool for Weaviate Plugin

This module provides a keyword-based search tool that uses BM25 algorithm for
text search in Weaviate collections. It enables users to search for documents
based on keyword matching with support for property filtering and result customization.

The keyword search tool is ideal for finding documents that contain specific
terms or phrases, leveraging the BM25 ranking algorithm which considers both
term frequency and document frequency to provide relevant results.

Classes:
    KeywordSearchTool: Main tool class for keyword search operations

Functions:
    _to_list: Convert various input types to a list of strings
"""

from collections.abc import Generator
from typing import Any, List, Optional
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_limit, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

def _to_list(value) -> Optional[List[str]]:
    """
    Convert various input types to a list of strings.
    
    This utility function normalizes different input formats (None, list, string)
    into a consistent list of strings format. It handles comma-separated strings,
    existing lists, and filters out empty values.
    
    Parameters:
        value: Input value to convert. Can be None, list, or string.
            - None: Returns None
            - List: Converts all elements to strings and strips whitespace
            - String: Splits by comma and strips whitespace from each part
        
    Returns:
        Optional[List[str]]: List of non-empty string values, or None if input is empty/invalid.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]

class KeywordSearchTool(Tool):
    """
    A keyword-based search tool that uses BM25 algorithm for text search in Weaviate.
    
    This tool provides a straightforward interface for performing keyword searches
    using the BM25 (Best Matching 25) ranking algorithm. BM25 is particularly
    effective for keyword-based queries as it considers both term frequency within
    documents and document frequency across the collection to provide relevant results.
    
    The tool supports property filtering, result customization, and flexible input
    formats for search parameters. It's ideal for use cases where exact keyword
    matching and term-based relevance are important.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute keyword-based search using BM25 algorithm.
        
        This method performs a text search on the specified Weaviate collection
        using the BM25 ranking algorithm. It searches for documents that contain
        the specified query terms and returns results ranked by relevance.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing search parameters
                - collection_name (str): Name of the Weaviate collection to search (required)
                - query (str): Text query containing keywords to search for (required)
                - limit (int): Maximum number of results to return (1-1000, default: 10)
                - where_filter (str/dict): JSON filter for document filtering (optional)
                    - String: JSON string containing filter criteria
                    - Dict: Direct filter dictionary
                - return_properties (str/list): Properties to return from results (optional)
                    - String: Comma-separated property names
                    - List: List of property names
                - search_properties (str/list): Properties to search within (optional)
                    - String: Comma-separated property names
                    - List: List of property names
                    - If not specified, searches all text properties
        
        Yields:
            ToolInvokeMessage: JSON messages containing search results or error information
            
        Search Algorithm:
            The tool uses BM25 (Best Matching 25) ranking algorithm which:
            - Considers term frequency (TF) within each document
            - Accounts for document frequency (DF) across the collection
            - Applies length normalization to prevent bias toward longer documents
            - Provides relevance scores that balance precision and recall
            
        Property Filtering:
            - return_properties: Controls which document properties are included in results
            - search_properties: Limits search to specific document properties
            - where_filter: Applies additional filtering criteria using Weaviate's query language
            
        Input Validation:
            - Collection name is required
            - Query text is required
            - Limit must be between 1 and 1000
            - Where filter must be valid JSON if provided
            - Property lists are parsed and validated
            
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
            limit_raw = tool_parameters.get('limit', 10)
            where_filter_raw = tool_parameters.get('where_filter')
            return_properties_in = tool_parameters.get('return_properties')
            search_properties_in = tool_parameters.get('search_properties')

            # Basic validation
            if not collection_name:
                yield self.create_json_message(create_error_response("Collection name is required"))
                return
            if not query:
                yield self.create_json_message(create_error_response("Query is required"))
                return

            # Parse limit
            try:
                limit = int(limit_raw)
            except (TypeError, ValueError):
                yield self.create_json_message(create_error_response("Limit must be an integer between 1 and 1000"))
                return
            if not validate_limit(limit):
                yield self.create_json_message(create_error_response("Limit must be between 1 and 1000"))
                return

            # Where filter: accept JSON string or dict
            where_filter = None
            if isinstance(where_filter_raw, str):
                s = where_filter_raw.strip()
                if s:
                    where_filter = safe_json_parse(s)
                    if where_filter is None or not validate_where_filter(where_filter):
                        yield self.create_json_message(create_error_response("Invalid where filter. Provide valid JSON"))
                        return
            elif isinstance(where_filter_raw, dict):
                where_filter = where_filter_raw
                if not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response("Invalid where filter JSON"))
                    return

            # Properties parsing (CSV or list)
            return_properties = _to_list(return_properties_in)
            search_properties = _to_list(search_properties_in)

            # Connect
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds['url'],
                api_key=creds.get('api_key'),
                timeout=60
            )

            try:
                results = client.text_search(
                    class_name=collection_name,
                    query=query,
                    limit=limit,
                    where_filter=where_filter,
                    return_properties=return_properties,
                    search_properties=search_properties
                )

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={
                            'results': [],
                            'count': 0,
                            'collection': collection_name,
                            'query': query,
                            'search_type': 'keyword'
                        },
                        message="No results found"
                    ))
                    return

                yield self.create_json_message(create_success_response(
                    data={
                        'results': results,
                        'count': len(results),
                        'collection': collection_name,
                        'query': query,
                        'search_type': 'keyword'
                    },
                    message=f"Found {len(results)} documents matching keywords"
                ))

            except Exception as e:
                logger.exception("Keyword search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))