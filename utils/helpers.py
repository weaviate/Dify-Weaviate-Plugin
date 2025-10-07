"""
Helper Utilities Module for Weaviate Plugin

This module provides utility functions for data processing, JSON handling, and response formatting
in the Weaviate Dify plugin. It includes functions for parsing, formatting, and transforming data
between different formats commonly used in vector database operations.

Author: Weaviate Team
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Union
import json
import logging

logger = logging.getLogger(__name__)

def format_search_results(results: List[Dict[str, Any]], include_metadata: bool = True) -> List[Dict[str, Any]]:
    """
    Format search results into a standardized structure.
    
    This function normalizes search results from Weaviate queries into a consistent format
    that includes UUID, properties, and optional metadata. It also converts distance scores
    to relevance scores for better user experience.
    
    Args:
        results (List[Dict[str, Any]]): Raw search results from Weaviate queries
        include_metadata (bool): Whether to include metadata in the formatted results
        
    Returns:
        List[Dict[str, Any]]: Formatted results with standardized structure
    """
    out: List[Dict[str, Any]] = []
    for r in results or []:
        item = {
            "uuid": r.get("uuid"),
            "properties": r.get("properties") or {},
        }
        if include_metadata:
            meta = r.get("metadata") or {}
            # Optional: normalize
            # If distance present, convert to relevance score in [0,1]
            if "distance" in meta and "relevance" not in meta:
                try:
                    d = float(meta["distance"])
                    meta["relevance"] = max(0.0, min(1.0, 1.0 - d))
                except Exception:
                    pass
            item["metadata"] = meta
        out.append(item)
    return out

def create_error_response(error_message: str, error_code: str = "WEAVIATE_ERROR", details: Any = None) -> Dict[str, Any]:
    """
    Create a standardized error response object.
    
    This function generates a consistent error response structure that can be used
    across all plugin tools to provide uniform error handling and reporting.
    
    Args:
        error_message (str): Human-readable error message describing what went wrong
        error_code (str): Machine-readable error code for programmatic handling
        details (Any, optional): Additional error details or context information
        
    Returns:
        Dict[str, Any]: Standardized error response dictionary
    """
    resp = {
        "success": False,
        "error": error_message,
        "error_code": error_code,
    }
    if details is not None:
        resp["details"] = details
    return resp

def create_success_response(data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
    """
    Create a standardized success response object.
    
    This function generates a consistent success response structure that can be used
    across all plugin tools to provide uniform success reporting.
    
    Args:
        data (Any): The actual data to return in the response
        message (str): Human-readable success message
        
    Returns:
        Dict[str, Any]: Standardized success response dictionary
    """
    return {
        "success": True,
        "data": data,
        "message": message,
    }

def safe_json_parse(value: Any, default: Any = None) -> Any:
    """
    Safely parse JSON from various input types.
    
    This function attempts to parse JSON from strings while handling dict/list inputs
    as-is. It provides robust error handling and returns a default value on failure.
    
    Args:
        value (Any): Input value to parse (string, dict, list, or other)
        default (Any): Default value to return if parsing fails
        
    Returns:
        Any: Parsed JSON object, original value if already dict/list, or default on failure
    """
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse JSON: %r", value)
            return default
    return default

def _parse_number(val: str) -> Optional[Union[int, float]]:
    """
    Parse a string value into a number (int or float).
    
    This internal helper function attempts to convert string values to appropriate
    numeric types while handling edge cases like NaN and infinity values.
    
    Args:
        val (str): String value to parse as a number
        
    Returns:
        Optional[Union[int, float]]: Parsed number or None if parsing fails
    """
    try:
        if isinstance(val, (int, float)):
            return val
        s = str(val).strip()
        if s.lower() in ("nan", "inf", "-inf"):
            return None
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        return float(s)
    except Exception:
        return None

def extract_properties_from_text(text: str) -> Dict[str, Any]:
    """
    Extract properties from text in key-value format.
    
    This function parses text containing key-value pairs separated by colons
    and converts them into a dictionary with appropriate type conversion.
    
    Args:
        text (str): Text containing key-value pairs separated by colons
        
    Returns:
        Dict[str, Any]: Dictionary of extracted properties with type conversion
    """
    props: Dict[str, Any] = {}
    if not text:
        return props
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip()
        if not key:
            continue
        low = val.lower()
        if low in ("true", "false"):
            props[key] = (low == "true")
            continue
        num = _parse_number(val)
        if num is not None:
            props[key] = num
        else:
            props[key] = val
    return props

def _value_field_for_python(value: Any) -> str:
    """
    Determine the correct Weaviate filter value field based on Python type.
    
    This internal helper function maps Python data types to the appropriate
    Weaviate filter value field names for query construction.
    
    Args:
        value (Any): Python value to determine the field type for
        
    Returns:
        str: Weaviate filter value field name (valueText, valueInt, etc.)
    """
    if isinstance(value, bool):
        return "valueBoolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "valueInt"
    if isinstance(value, float):
        return "valueNumber"
    # Fallback to text
    return "valueText"

def build_where_filter(conditions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a Weaviate where filter from simple conditions.
    
    This function converts a simple dictionary of field-value pairs into a
    proper Weaviate where filter structure with appropriate operators and value types.
    
    Args:
        conditions (Dict[str, Any]): Dictionary of field-value conditions
        
    Returns:
        Dict[str, Any]: Weaviate where filter structure or empty dict if no conditions
    """
    if not conditions:
        return {}

    def _single_cond(k: str, v: Any) -> Dict[str, Any]:
        val_key = _value_field_for_python(v)
        return {"path": [k], "operator": "Equal", val_key: v}

    items = list(conditions.items())
    if len(items) == 1:
        k, v = items[0]
        return _single_cond(k, v)

    return {
        "operator": "And",
        "operands": [_single_cond(k, v) for k, v in items],
    }

def csv_or_list_to_list(value: Any) -> Optional[List[str]]:
    """
    Convert CSV string or list to normalized list of strings.
    
    This function handles both comma-separated strings and existing lists,
    normalizing them into a clean list of non-empty string values.
    
    Args:
        value (Any): CSV string, list, or other value to convert
        
    Returns:
        Optional[List[str]]: Normalized list of strings or None if input is empty/invalid
    """
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return None
    return [p.strip() for p in s.split(",") if p.strip()]