"""
Schema Management Tool for Weaviate Plugin

This module provides comprehensive schema management capabilities for Weaviate
collections, including creation, deletion, inspection, and modification of
collection schemas and configurations.

The schema management tool enables users to define and manage the structure
of their Weaviate collections, including property definitions, vectorizer
configurations, and collection metadata. It supports various operations for
collection lifecycle management and schema inspection.

Classes:
    SchemaManagementTool: Main tool class for schema management operations

Constants:
    _ALLOWED_OPS: Set of allowed operations for the schema management tool
    _ALLOWED_VECTORIZERS: Set of supported vectorizer configurations
"""

from collections.abc import Generator
from typing import Any, List, Dict, Optional
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_collection_name, validate_properties
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

# Allowed operations for the schema management tool
_ALLOWED_OPS = {
    "list_collections",
    "create_collection",
    "delete_collection",
    "get_schema",
    "get_stats",
    "exists",
    "add_property",
    "update_config",
}

# Optional: restrict vectorizers we recognize; None/self_provided is default
_ALLOWED_VECTORIZERS = {"self_provided", "text2vec-openai", "text2vec-transformers"}

class SchemaManagementTool(Tool):
    """
    A comprehensive schema management tool for Weaviate collections.
    
    This tool provides a unified interface for managing Weaviate collection schemas,
    including creation, deletion, inspection, and modification operations. It supports
    various collection management tasks such as defining properties, configuring
    vectorizers, and managing collection metadata.
    
    The tool handles the complete lifecycle of Weaviate collections from creation
    to deletion, with support for schema inspection, property management, and
    configuration updates. It validates inputs and provides detailed feedback
    for all operations.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute schema management operations based on provided parameters.
        
        This is the main entry point for the schema management tool that processes
        different operations including collection creation, deletion, schema inspection,
        and configuration management. The method validates input parameters, establishes
        a Weaviate client connection, executes the requested operation, and returns
        appropriate responses.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing operation parameters
                - operation (str): The operation to perform. Must be one of:
                    - "list_collections": List all available collections
                    - "create_collection": Create a new collection with specified schema
                    - "delete_collection": Delete an existing collection
                    - "get_schema": Retrieve schema information for a collection
                    - "get_stats": Get statistics for a collection
                    - "exists": Check if a collection exists
                    - "add_property": Add a new property to an existing collection
                    - "update_config": Update collection configuration
                - collection_name (str): Name of the target collection (required for all operations except list_collections)
                - properties (str): JSON string containing property definitions for collection creation
                - vectorizer (str): Vectorizer configuration (optional, default: self_provided)
                - description (str): Collection description (optional)
                - multi_tenancy (bool): Enable multi-tenancy for the collection (optional)
                - property (str): JSON string containing property definition for add_property operation
                - config (str): JSON string containing configuration updates for update_config operation
        
        Yields:
            ToolInvokeMessage: JSON messages containing operation results or error information
            
        Operations:
            list_collections:
                Lists all available collections in the Weaviate instance.
                Returns collection names and count.
                
            create_collection:
                Creates a new collection with specified properties and configuration.
                Supports property definitions, vectorizer configuration, and metadata.
                Validates property structure and collection name format.
                Returns confirmation of successful creation.
                
            delete_collection:
                Deletes an existing collection and all its data.
                Performs permanent deletion with confirmation.
                Returns confirmation of successful deletion.
                
            get_schema:
                Retrieves the complete schema definition for a collection.
                Returns property definitions, data types, and configuration details.
                Useful for schema inspection and documentation.
                
            get_stats:
                Retrieves statistics and metadata for a collection.
                Returns document counts, size information, and collection metrics.
                Provides insights into collection usage and performance.
                
            exists:
                Checks whether a collection exists in the Weaviate instance.
                Returns boolean result with collection name confirmation.
                Useful for validation before operations.
                
            add_property:
                Adds a new property to an existing collection schema.
                Validates property definition format and data types.
                Returns confirmation of successful property addition.
                
            update_config:
                Updates collection configuration settings.
                Supports various configuration parameters and settings.
                Returns confirmation of successful configuration update.
        
        Vectorizer Support:
            The tool supports various vectorizer configurations:
            - self_provided: Use client-provided vectors (default)
            - text2vec-openai: OpenAI text embedding model
            - text2vec-transformers: Local transformer-based embeddings
            
        Property Definition:
            Properties must be defined as JSON objects with:
            - name: Property name (string)
            - data_type: Weaviate data type (string, int, float, boolean, etc.)
            - Additional configuration options as needed
            
        Input Validation:
            - Collection names must follow Weaviate naming conventions
            - Property definitions must include required fields
            - Vectorizer configurations must be from allowed list
            - JSON inputs are validated for proper format
            
        Error Handling:
            Comprehensive validation for:
            - Invalid operation names
            - Missing required parameters
            - Invalid collection names
            - Malformed property definitions
            - Unsupported vectorizer configurations
            - Weaviate client connection issues
            
        Returns:
            Generator[ToolInvokeMessage]: Stream of JSON messages containing:
                - Success responses with operation results and metadata
                - Error responses with detailed error messages
                - Status information and operation confirmations
        """
        try:
            op_raw = tool_parameters.get("operation", "")
            operation = (op_raw or "").strip().lower()

            collection_name = (tool_parameters.get("collection_name") or "").strip()
            properties_raw = tool_parameters.get("properties")
            vectorizer_raw = (tool_parameters.get("vectorizer") or "").strip()

            if operation not in _ALLOWED_OPS:
                yield self.create_json_message(create_error_response(
                    f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}"
                ))
                return

            if operation != "list_collections" and not collection_name:
                yield self.create_json_message(create_error_response(
                    "Collection name is required for this operation"
                ))
                return

            if collection_name and not validate_collection_name(collection_name):
                yield self.create_json_message(create_error_response(
                    "Invalid collection name. Use letters, digits, and underscores, starting with a letter or underscore (e.g., my_collection, UserProfiles)"
                ))
                return

            # Normalize vectorizer
            vectorizer = None
            if vectorizer_raw:
                v = vectorizer_raw.strip()
                # Map common aliases
                if v in {"none", "self", "self_provided"}:
                    vectorizer = None  # default to self-provided in client
                elif v in _ALLOWED_VECTORIZERS:
                    vectorizer = v
                else:
                    yield self.create_json_message(create_error_response(
                        f"Unsupported vectorizer '{v}'. Allowed: {sorted(_ALLOWED_VECTORIZERS)}"
                    ))
                    return

            # Connect
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds["url"],
                api_key=creds.get("api_key"),
                timeout=60,
            )

            try:
                # ---- list_collections ----
                if operation == "list_collections":
                    cols = client.list_collections()
                    yield self.create_json_message(create_success_response(
                        data={"collections": cols, "count": len(cols)},
                        message=f"Found {len(cols)} collections"
                    ))
                    return

                # ---- create_collection ----
                if operation == "create_collection":
                    if not properties_raw:
                        yield self.create_json_message(create_error_response(
                            "Properties are required for collection creation"
                        ))
                        return

                    props = safe_json_parse(properties_raw)
                    # Allow a single object or an array
                    if isinstance(props, dict):
                        props = [props]

                    if not validate_properties(props):
                        yield self.create_json_message(create_error_response(
                            "Invalid properties. Provide a JSON array of property objects with 'name' and 'data_type'"
                        ))
                        return

                    # Get optional parameters
                    description = (tool_parameters.get("description") or "").strip() or None
                    multi_tenancy = tool_parameters.get("multi_tenancy")

                    created = client.create_collection(
                        class_name=collection_name,
                        properties=props,
                        vectorizer=vectorizer,
                        vector_index_config=None,  # Can be exposed later if needed
                        description=description,
                        multi_tenancy=bool(multi_tenancy) if multi_tenancy is not None else None,
                    )
                    if created:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name},
                            message=f"Collection '{collection_name}' created successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to create collection '{collection_name}'"
                        ))
                    return

                # ---- delete_collection ----
                if operation == "delete_collection":
                    ok = client.delete_collection(collection_name)
                    if ok:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name},
                            message=f"Collection '{collection_name}' deleted successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to delete collection '{collection_name}'"
                        ))
                    return

                # ---- get_schema ----
                if operation == "get_schema":
                    schema = client.get_collection_schema(collection_name)
                    if schema:
                        yield self.create_json_message(create_success_response(
                            data={"schema": schema, "collection_name": collection_name},
                            message=f"Schema retrieved for '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to get schema for '{collection_name}'"
                        ))
                    return

                # ---- get_stats ----
                if operation == "get_stats":
                    stats = client.get_collection_stats(collection_name)
                    if stats:
                        yield self.create_json_message(create_success_response(
                            data={"stats": stats, "collection_name": collection_name},
                            message=f"Statistics retrieved for '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to get stats for '{collection_name}'"
                        ))
                    return

                # ---- exists ----
                if operation == "exists":
                    exists = client.collection_exists(collection_name)
                    yield self.create_json_message(create_success_response(
                        data={"exists": exists, "collection_name": collection_name},
                        message=f"Collection '{collection_name}' exists: {exists}"
                    ))
                    return

                # ---- add_property ----
                if operation == "add_property":
                    prop_str = (tool_parameters.get("property") or "").strip()
                    if not prop_str:
                        yield self.create_json_message(create_error_response(
                            "Property definition is required for add_property operation"
                        ))
                        return
                    
                    prop = safe_json_parse(prop_str)
                    if not isinstance(prop, dict) or "name" not in prop or "data_type" not in prop:
                        yield self.create_json_message(create_error_response(
                            "Invalid property format. Provide JSON with 'name' and 'data_type' fields"
                        ))
                        return
                    
                    success = client.add_property(collection_name, prop)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "property": prop["name"]},
                            message=f"Property '{prop['name']}' added to collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to add property '{prop['name']}' to collection '{collection_name}'"
                        ))
                    return

                # ---- update_config ----
                if operation == "update_config":
                    cfg_str = (tool_parameters.get("config") or "").strip()
                    if not cfg_str:
                        yield self.create_json_message(create_error_response(
                            "Config updates are required for update_config operation"
                        ))
                        return
                    
                    cfg = safe_json_parse(cfg_str)
                    if not isinstance(cfg, dict):
                        yield self.create_json_message(create_error_response(
                            "Invalid config format. Provide JSON object with configuration updates"
                        ))
                        return
                    
                    success = client.update_collection_config(collection_name, cfg)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "applied_config": cfg},
                            message=f"Configuration updated for collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to update configuration for collection '{collection_name}'"
                        ))
                    return

                # Fallback (shouldn't hit due to _ALLOWED_OPS)
                yield self.create_json_message(create_error_response(
                    f"Unsupported operation '{operation}'"
                ))

            except Exception as e:
                logger.exception("Schema management error")
                yield self.create_json_message(create_error_response(f"Operation failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))