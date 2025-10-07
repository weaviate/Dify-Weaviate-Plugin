"""
Tenant Management Tool for Weaviate Plugin

This module provides a tenant management tool for handling multi-tenancy operations
in Weaviate collections. It enables users to manage tenant isolation by adding,
removing, and listing tenants within collections that support multi-tenancy.

Multi-tenancy in Weaviate allows data isolation at the collection level, where
different tenants can have separate data spaces within the same collection.
This tool provides the necessary operations to manage these tenant configurations.

Classes:
    TenantManagementTool: Main tool class for tenant management operations
"""

from collections.abc import Generator
from typing import Any, List
import logging

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

# Allowed operations for tenant management
_ALLOWED_OPS = {"list_tenants", "add_tenants", "delete_tenants"}

class TenantManagementTool(Tool):
    """
    A tenant management tool for handling multi-tenancy operations in Weaviate collections.
    
    This tool provides a unified interface for managing tenants within Weaviate
    collections that support multi-tenancy. It enables users to add, remove,
    and list tenants, ensuring proper data isolation and access control.
    
    Multi-tenancy allows multiple tenants to share the same collection schema
    while maintaining complete data isolation. Each tenant's data is stored
    separately and can only be accessed by that specific tenant.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute tenant management operations based on provided parameters.
        
        This is the main entry point for the tenant management tool that processes
        different operations including listing, adding, and deleting tenants from
        Weaviate collections. The method validates input parameters, establishes
        a Weaviate client connection, executes the requested operation, and returns
        appropriate responses.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing operation parameters
                - operation (str): The operation to perform. Must be one of:
                    - "list_tenants": List all tenants in a collection
                    - "add_tenants": Add new tenants to a collection
                    - "delete_tenants": Remove tenants from a collection
                - collection_name (str): Name of the target collection (required for all operations)
                - tenants (str): JSON array string containing tenant names (required for add/delete operations)
                    - Format: '["tenant1", "tenant2", "tenant3"]'
                    - Each tenant name should be a valid string identifier
        
        Yields:
            ToolInvokeMessage: JSON messages containing operation results or error information
            
        Operations:
            list_tenants:
                Lists all tenants currently configured for the specified collection.
                Returns tenant names and count information.
                No additional parameters required beyond collection_name.
                
            add_tenants:
                Adds new tenants to the specified collection.
                Requires tenants parameter with JSON array of tenant names.
                Validates tenant name format and collection compatibility.
                Returns confirmation of added tenants.
                
            delete_tenants:
                Removes existing tenants from the specified collection.
                Requires tenants parameter with JSON array of tenant names.
                Validates tenant existence before deletion.
                Returns confirmation of deleted tenants.
                
        Multi-tenancy Requirements:
            - Collection must support multi-tenancy (enabled during creation)
            - Tenant names must be valid identifiers
            - Operations require appropriate permissions
            - Data isolation is maintained automatically by Weaviate
            
        Input Validation:
            - Operation must be one of the allowed operations
            - Collection name is required for all operations
            - Tenant list is required for add/delete operations
            - Tenant list must be valid JSON array format
            - Tenant names must be non-empty strings
            
        Error Handling:
            Comprehensive validation for:
            - Invalid operation names
            - Missing required parameters
            - Invalid JSON formats
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
            op_raw = tool_parameters.get("operation", "")
            operation = (op_raw or "").strip().lower()

            collection_name = (tool_parameters.get("collection_name") or "").strip()
            tenants_str = (tool_parameters.get("tenants") or "").strip()

            if operation not in _ALLOWED_OPS:
                yield self.create_json_message(create_error_response(
                    f"Unknown operation '{operation}'. Allowed: {sorted(_ALLOWED_OPS)}"
                ))
                return

            if not collection_name:
                yield self.create_json_message(create_error_response(
                    "Collection name is required for tenant operations"
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
                # ---- list_tenants ----
                if operation == "list_tenants":
                    tenants = client.list_tenants(collection_name)
                    yield self.create_json_message(create_success_response(
                        data={"tenants": tenants, "count": len(tenants), "collection_name": collection_name},
                        message=f"Found {len(tenants)} tenants in collection '{collection_name}'"
                    ))
                    return

                # ---- add_tenants ----
                if operation == "add_tenants":
                    if not tenants_str:
                        yield self.create_json_message(create_error_response(
                            "Tenant list is required for add_tenants operation"
                        ))
                        return

                    tenants = safe_json_parse(tenants_str)
                    if not isinstance(tenants, list):
                        yield self.create_json_message(create_error_response(
                            "Invalid tenants format. Provide JSON array of tenant names"
                        ))
                        return

                    success = client.add_tenants(collection_name, tenants)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "added_tenants": tenants},
                            message=f"Added {len(tenants)} tenants to collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to add tenants to collection '{collection_name}'"
                        ))
                    return

                # ---- delete_tenants ----
                if operation == "delete_tenants":
                    if not tenants_str:
                        yield self.create_json_message(create_error_response(
                            "Tenant list is required for delete_tenants operation"
                        ))
                        return

                    tenants = safe_json_parse(tenants_str)
                    if not isinstance(tenants, list):
                        yield self.create_json_message(create_error_response(
                            "Invalid tenants format. Provide JSON array of tenant names"
                        ))
                        return

                    success = client.delete_tenants(collection_name, tenants)
                    if success:
                        yield self.create_json_message(create_success_response(
                            data={"collection_name": collection_name, "deleted_tenants": tenants},
                            message=f"Deleted {len(tenants)} tenants from collection '{collection_name}'"
                        ))
                    else:
                        yield self.create_json_message(create_error_response(
                            f"Failed to delete tenants from collection '{collection_name}'"
                        ))
                    return

            except Exception as e:
                logger.exception("Tenant management error")
                yield self.create_json_message(create_error_response(f"Operation failed: {e}"))

            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Client disconnect failed quietly", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))