"""
Data Management Tool for Weaviate Plugin

This module provides a comprehensive data management tool for interacting with Weaviate
vector database. It supports various CRUD operations including listing collections,
inserting, updating, deleting, and retrieving objects.

Classes:
    DataManagementTool: Main tool class for data management operations

Constants:
    _ALLOWED_OPS: Set of allowed operations for the tool
"""

from collections.abc import Generator
from typing import Any, List, Dict, Optional
import json
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

# Allowed operations for the data management tool
_ALLOWED_OPS = {"list_collections", "insert", "update", "delete", "get", "list_objects"}

class DataManagementTool(Tool):
    """
    A comprehensive data management tool for Weaviate vector database operations.
    
    This tool provides a unified interface for performing various data management
    operations on Weaviate collections including listing collections, inserting,
    updating, deleting, and retrieving objects. It supports both single and batch
    operations, multi-tenancy, and various filtering options.
    
    The tool inherits from the Dify Plugin Tool base class and implements the
    required _invoke method to handle tool execution.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """
    
    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute data management operations based on provided parameters.
        
        This is the main entry point for the tool that processes different operations
        including list_collections, insert, update, delete, get, and list_objects.
        The method validates input parameters, establishes a Weaviate client connection,
        executes the requested operation, and returns appropriate responses.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing operation parameters
                - operation (str): The operation to perform. Must be one of:
                    - "list_collections": List all available collections
                    - "insert": Insert one or more objects into a collection
                    - "update": Update an existing object
                    - "delete": Delete an object or objects by filter
                    - "get": Retrieve a specific object by UUID
                    - "list_objects": List objects from a collection with optional filtering
                - collection_name (str): Name of the target collection (required for all operations except list_collections)
                - object_data (str): JSON string containing object data (required for insert/update)
                - object_uuid (str): UUID of the object to operate on (required for get/update/delete)
                - return_properties (str): Comma-separated list of properties to return
                - where_filter (str): JSON string containing filter criteria for list_objects/delete
                - limit (int): Maximum number of objects to return (default: 100)
                - tenant (str): Tenant identifier for multi-tenant operations
                - include_vector (bool): Whether to include vector data in results (default: False)
                - return_additional (str): Comma-separated list of additional fields to return
                - mode (str): Insert mode - "single" or "batch" (default: "single")
                - batch_size (int): Number of objects per batch for batch insert (default: 100)
                - update_mode (str): Update mode - "merge", "replace", or "vector_only" (default: "merge")
                - dry_run (bool): Whether to perform a dry run for delete operations (default: False)
        
        Yields:
            ToolInvokeMessage: JSON messages containing operation results or error information
            
        Operations:
            list_collections:
                Lists all available collections in the Weaviate instance.
                Returns collection names and count.
                
            insert:
                Inserts one or more objects into the specified collection.
                Supports both single object and batch insertion modes.
                For batch mode, objects are processed in configurable batch sizes.
                Returns UUIDs of inserted objects or batch operation statistics.
                
            update:
                Updates an existing object in the specified collection.
                Supports three update modes:
                - merge: Merge new data with existing object properties
                - replace: Replace the entire object with new data
                - vector_only: Update only the vector data of the object
                Returns confirmation of successful update.
                
            delete:
                Deletes objects from the specified collection.
                Supports two deletion methods:
                - UUID-based: Delete a specific object by its UUID
                - Filter-based: Delete multiple objects matching filter criteria
                Supports dry-run mode to preview deletion without executing.
                Returns count of deleted objects.
                
            get:
                Retrieves a specific object by its UUID from the specified collection.
                Supports property filtering and additional field inclusion.
                Returns the complete object data or null if not found.
                
            list_objects:
                Lists objects from the specified collection with optional filtering.
                Supports property filtering, limit constraints, and field selection.
                Can include vector data and additional metadata fields.
                Returns array of matching objects with count information.
        
        Error Handling:
            The method includes comprehensive error handling for:
            - Invalid operation names
            - Missing required parameters
            - Invalid JSON data formats
            - Weaviate client connection issues
            - Operation-specific validation errors
            
        Client Management:
            Automatically establishes and manages Weaviate client connections.
            Ensures proper cleanup by disconnecting clients in finally blocks.
            Uses credentials from the runtime context for authentication.
        
        Returns:
            Generator[ToolInvokeMessage]: Stream of JSON messages containing:
                - Success responses with operation results and metadata
                - Error responses with detailed error messages
                - Status information and operation confirmations
        """
        try:
            # normalize inputs
            op_raw = tool_parameters.get("operation", "")
            operation = (op_raw or "").strip().lower()

            collection_name = (tool_parameters.get("collection_name") or "").strip()
            object_data_str = (tool_parameters.get("object_data") or "").strip()
            object_uuid = (tool_parameters.get("object_uuid") or "").strip()
            return_properties_str = (tool_parameters.get("return_properties") or "").strip()

            if operation not in _ALLOWED_OPS:
                yield self.create_json_message(create_error_response(
                    f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}"
                ))
                return

            if operation != "list_collections" and not collection_name:
                yield self.create_json_message(create_error_response(
                    "Collection name is required"
                ))
                return

            # credentials / client
            creds = self.runtime.credentials
            client = WeaviateClient(
                url=creds["url"],
                api_key=creds.get("api_key"),
                timeout=60,
            )

            try:
                # ---- list_collections ----
                if operation == "list_collections":
                    collections = client.list_collections()
                    yield self.create_json_message(create_success_response(
                        data={"collections": collections, "count": len(collections)},
                        message=f"Found {len(collections)} collections"
                    ))
                    return

                # parse optional projection
                return_properties = None
                if return_properties_str:
                    return_properties = [p.strip() for p in return_properties_str.split(",") if p.strip()]

                # ---- list_objects ----
                if operation == "list_objects":
                    where_filter = None
                    if tool_parameters.get("where_filter"):
                        where_filter = safe_json_parse(tool_parameters.get("where_filter"))
                        if not isinstance(where_filter, dict):
                            yield self.create_json_message(create_error_response(
                                "Invalid where_filter format. Provide valid JSON object"
                            ))
                            return
                    
                    limit = tool_parameters.get("limit", 100)
                    tenant = tool_parameters.get("tenant")
                    include_vector = tool_parameters.get("include_vector", False)
                    return_additional = None
                    if tool_parameters.get("return_additional"):
                        return_additional = [f.strip() for f in tool_parameters.get("return_additional").split(",") if f.strip()]
                    
                    objects = client.list_objects(
                        class_name=collection_name,
                        where_filter=where_filter,
                        limit=limit,
                        tenant=tenant,
                        return_properties=return_properties,
                        include_vector=include_vector,
                        return_additional=return_additional
                    )
                    
                    yield self.create_json_message(create_success_response(
                        data={"objects": objects, "count": len(objects), "collection": collection_name},
                        message=f"Retrieved {len(objects)} objects from collection '{collection_name}'"
                    ))
                    return

                # ---- insert ----
                if operation == "insert":
                    if not object_data_str:
                        yield self.create_json_message(create_error_response(
                            "Object data is required for insert operation"
                        ))
                        return

                    object_data = safe_json_parse(object_data_str)
                    if object_data is None:
                        yield self.create_json_message(create_error_response(
                            "Invalid object data format. Provide valid JSON (object or array of objects)"
                        ))
                        return

                    # allow single dict or list of dicts
                    if isinstance(object_data, dict):
                        payload = [object_data]
                    elif isinstance(object_data, list):
                        payload = object_data
                    else:
                        yield self.create_json_message(create_error_response(
                            "Object data must be a JSON object or an array of objects"
                        ))
                        return

                    mode = tool_parameters.get("mode", "single")
                    tenant = tool_parameters.get("tenant")
                    batch_size = tool_parameters.get("batch_size", 100)
                    
                    try:
                        if mode == "batch":
                            result = client.insert_objects_batch(
                                class_name=collection_name,
                                objects=payload,
                                tenant=tenant,
                                batch_size=batch_size
                            )
                            
                            yield self.create_json_message(create_success_response(
                                data={
                                    "inserted_count": result["inserted_count"],
                                    "failed_count": result["failed_count"],
                                    "errors": result["errors"],
                                    "collection": collection_name
                                },
                                message=f"Batch insert completed: {result['inserted_count']} inserted, {result['failed_count']} failed"
                            ))
                        else:
                            # Single mode (existing logic)
                            uuids = client.insert_objects(collection_name, payload)
                            if uuids:
                                yield self.create_json_message(create_success_response(
                                    data={"inserted_uuids": uuids, "count": len(uuids), "collection": collection_name},
                                    message=f"Successfully inserted {len(uuids)} objects"
                                ))
                            else:
                                yield self.create_json_message(create_error_response("Failed to insert objects - no UUIDs returned"))
                    except Exception as insert_error:
                        yield self.create_json_message(create_error_response(
                            f"Insert failed: {str(insert_error)}"
                        ))
                    return

                # ---- update ----
                if operation == "update":
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID is required for update operation"
                        ))
                        return
                    if not object_data_str:
                        yield self.create_json_message(create_error_response(
                            "Object data is required for update operation"
                        ))
                        return

                    object_data = safe_json_parse(object_data_str)
                    if not isinstance(object_data, dict):
                        yield self.create_json_message(create_error_response(
                            "Invalid object data format. Provide a JSON object"
                        ))
                        return

                    update_mode = tool_parameters.get("update_mode", "merge")
                    tenant = tool_parameters.get("tenant")
                    
                    try:
                        if update_mode == "replace":
                            success = client.replace_object(
                                class_name=collection_name,
                                uuid=object_uuid,
                                properties=object_data,
                                tenant=tenant
                            )
                        elif update_mode == "vector_only":
                            vector = object_data.get("vector")
                            if not vector:
                                yield self.create_json_message(create_error_response(
                                    "Vector data is required for vector_only update mode"
                                ))
                                return
                            success = client.update_vector(
                                class_name=collection_name,
                                uuid=object_uuid,
                                vector=vector,
                                tenant=tenant
                            )
                        else:  # merge mode (default)
                            success = client.update_object(collection_name, object_uuid, object_data)
                        
                        if success:
                            yield self.create_json_message(create_success_response(
                                data={"uuid": object_uuid, "collection": collection_name, "mode": update_mode},
                                message=f"Object updated successfully using {update_mode} mode"
                            ))
                        else:
                            yield self.create_json_message(create_error_response(f"Failed to update object using {update_mode} mode"))
                    except Exception as update_error:
                        yield self.create_json_message(create_error_response(
                            f"Update failed: {str(update_error)}"
                        ))
                    return

                # ---- delete ----
                if operation == "delete":
                    tenant = tool_parameters.get("tenant")
                    dry_run = tool_parameters.get("dry_run", False)
                    
                    # Check if we have a where filter instead of UUID
                    where_filter_str = tool_parameters.get("where_filter")
                    if where_filter_str:
                        where_filter = safe_json_parse(where_filter_str)
                        if not isinstance(where_filter, dict):
                            yield self.create_json_message(create_error_response(
                                "Invalid where_filter format. Provide valid JSON object"
                            ))
                            return
                        
                        result = client.delete_by_filter(
                            class_name=collection_name,
                            where_filter=where_filter,
                            tenant=tenant,
                            dry_run=dry_run
                        )
                        
                        if dry_run:
                            yield self.create_json_message(create_success_response(
                                data=result,
                                message=f"Dry run: would delete {result['would_delete_count']} objects"
                            ))
                        else:
                            yield self.create_json_message(create_success_response(
                                data=result,
                                message=f"Deleted {result['deleted_count']} objects, {result['failed_count']} failed"
                            ))
                        return
                    
                    # Original UUID-based delete
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID or where_filter is required for delete operation"
                        ))
                        return

                    success = client.delete_object(collection_name, object_uuid)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"uuid": object_uuid, "collection": collection_name},
                            message="Object deleted successfully"
                        ))
                    else:
                        yield self.create_json_message(create_error_response("Failed to delete object"))
                    return

                # ---- get ----
                if operation == "get":
                    if not object_uuid:
                        yield self.create_json_message(create_error_response(
                            "Object UUID is required for get operation"
                        ))
                        return

                    tenant = tool_parameters.get("tenant")
                    include_vector = tool_parameters.get("include_vector", False)
                    return_additional = None
                    if tool_parameters.get("return_additional"):
                        return_additional = [f.strip() for f in tool_parameters.get("return_additional").split(",") if f.strip()]

                    result = client.get_object(
                        class_name=collection_name, 
                        uuid=object_uuid, 
                        return_properties=return_properties
                    )
                    
                    if result:
                        yield self.create_json_message(create_success_response(
                            data={"object": result, "collection": collection_name},
                            message="Object retrieved successfully"
                        ))
                    else:
                        yield self.create_json_message(create_success_response(
                            data={"object": None, "collection": collection_name},
                            message="Object not found"
                        ))
                    return

                # fallback (shouldn't happen)
                yield self.create_json_message(create_error_response(
                    f"Unsupported operation '{operation}'"
                ))

            except Exception as e:
                logger.exception("Data management error")
                yield self.create_json_message(create_error_response(f"Operation failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))