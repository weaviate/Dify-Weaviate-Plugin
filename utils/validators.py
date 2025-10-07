"""
Validation Utilities Module for Weaviate Plugin

This module provides validation functions for various data types and structures
used throughout the Weaviate Dify plugin. It includes validators for URLs, API keys,
collection names, properties, vectors, filters, and other input parameters to ensure
data integrity and prevent errors during Weaviate operations.

Author: Weaviate Team
Version: 1.0.0
"""

from typing import Any, Dict, List, Union
import re
import json

VALID_DATA_TYPES = {
    "text", "int", "number", "boolean", "date",
    "uuid", "geoCoordinates", "blob"
}

def validate_weaviate_url(url: str) -> bool:
    """
    Validate Weaviate instance URL format.
    
    This function checks if the provided URL follows the correct format for
    Weaviate instance URLs, including protocol, hostname, optional port, and path.
    
    Args:
        url (str): URL string to validate
        
    Returns:
        bool: True if URL format is valid, False otherwise
    """
    pattern = r'^https?://[a-zA-Z0-9.-]+(:\d+)?(/.*)?$'
    return bool(url and re.match(pattern, url))

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format and presence.
    
    This function checks if the API key is present and not empty after stripping
    whitespace, ensuring it's a valid non-empty string.
    
    Args:
        api_key (str): API key string to validate
        
    Returns:
        bool: True if API key is valid, False otherwise
    """
    return bool(api_key and api_key.strip())

def validate_collection_name(name: str) -> bool:
    """
    Validate Weaviate collection name format.
    
    This function checks if the collection name follows Weaviate v4 naming conventions,
    which allow letters, numbers, and underscores, starting with a letter or underscore.
    
    Args:
        name (str): Collection name to validate
        
    Returns:
        bool: True if collection name format is valid, False otherwise
    """
    # v4 allows lowercase and underscores
    pattern = r'^[A-Za-z_][A-Za-z0-9_]*$'
    return bool(name and re.match(pattern, name))

def validate_properties(properties: List[Dict[str, Any]]) -> bool:
    """
    Validate collection property definitions.
    
    This function validates a list of property definitions to ensure they have
    the required structure and valid data types according to Weaviate specifications.
    
    Args:
        properties (List[Dict[str, Any]]): List of property definition dictionaries
        
    Returns:
        bool: True if all properties are valid, False otherwise
    """
    if not isinstance(properties, list) or not properties:
        return False
    for prop in properties:
        if not isinstance(prop, dict):
            return False
        n, t = prop.get("name"), prop.get("data_type")
        if not (isinstance(n, str) and n and re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', n)):
            return False
        if not (isinstance(t, str) and t in VALID_DATA_TYPES):
            return False
    return True

def validate_vector(vector: List[Union[int, float]], expected_dim: int = None) -> bool:
    """
    Validate vector data format and dimensions.
    
    This function checks if the provided vector is a valid list of numbers
    and optionally validates the dimension count matches the expected value.
    
    Args:
        vector (List[Union[int, float]]): Vector data to validate
        expected_dim (int, optional): Expected dimension count for validation
        
    Returns:
        bool: True if vector format is valid, False otherwise
    """
    if not isinstance(vector, list) or not vector:
        return False
    if not all(isinstance(x, (int, float)) for x in vector):
        return False
    if expected_dim and len(vector) != expected_dim:
        return False
    return True

def validate_where_filter(where_filter: Dict[str, Any]) -> bool:
    """
    Validate Weaviate where filter structure.
    
    This function checks if the where filter follows the correct Weaviate filter
    structure with proper operators and operands, and ensures it's JSON serializable.
    
    Args:
        where_filter (Dict[str, Any]): Where filter dictionary to validate
        
    Returns:
        bool: True if filter structure is valid, False otherwise
    """
    if not isinstance(where_filter, dict):
        return False
    try:
        json.dumps(where_filter)
    except (TypeError, ValueError):
        return False
    # minimal structural check
    if "operator" in where_filter and where_filter["operator"] in {"And", "Or", "Not"}:
        return "operands" in where_filter
    if "path" in where_filter and "operator" in where_filter:
        return True
    return False

def validate_limit(limit: Union[int, str], max_limit: int = 1000) -> bool:
    """
    Validate limit parameter for queries and operations.
    
    This function checks if the limit value is a valid integer within the
    acceptable range for Weaviate operations.
    
    Args:
        limit (Union[int, str]): Limit value to validate
        max_limit (int): Maximum allowed limit value
        
    Returns:
        bool: True if limit is valid, False otherwise
    """
    try:
        val = int(limit)
        return 1 <= val <= max_limit
    except (TypeError, ValueError):
        return False

def validate_alpha(alpha: Union[float, int, str]) -> bool:
    """
    Validate alpha parameter for hybrid search operations.
    
    This function checks if the alpha value is a valid number between 0.0 and 1.0,
    which is the required range for hybrid search weighting parameters.
    
    Args:
        alpha (Union[float, int, str]): Alpha value to validate
        
    Returns:
        bool: True if alpha is valid, False otherwise
    """
    try:
        val = float(alpha)
        return 0.0 <= val <= 1.0
    except (TypeError, ValueError):
        return False