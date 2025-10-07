"""
Weaviate Client Utility Module

This module provides a high-level Python client for interacting with Weaviate vector database.
It abstracts the complexity of Weaviate's API and provides convenient methods for common operations
including collection management, data insertion, querying, and search operations.

Author: Weaviate Team
Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.classes.config import Property, DataType, Tokenization, Configure
from weaviate.classes.data import DataObject
from weaviate.exceptions import UnexpectedStatusCodeError

logger = logging.getLogger(__name__)

def _parse_endpoint(url: str) -> tuple:
    """
    Parse a Weaviate endpoint URL and extract connection parameters.
    
    This helper function takes a full HTTP(S) URL and extracts the necessary
    host, port, and security settings for both HTTP and gRPC connections.
    
    Args:
        url (str): Full Weaviate URL (e.g., "https://localhost:8080" or "http://weaviate.example.com")
        
    Returns:
        tuple: A tuple containing (http_host, http_port, http_secure, grpc_host, grpc_port, grpc_secure)
    """
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        # allow raw host
        host = url.replace("https://", "").replace("http://", "")
        return host, 443, True, host, 443, True
    http_secure = (p.scheme == "https")
    http_port = p.port or (443 if http_secure else 80)
    host = p.hostname or p.netloc

    grpc_secure = http_secure
    grpc_port = p.port or (443 if grpc_secure else 50051)
    return host, http_port, http_secure, host, grpc_port, grpc_secure

class WeaviateClient:
    """
    High-level client for interacting with Weaviate vector database.
    
    This client provides a simplified interface for common Weaviate operations including
    collection management, data insertion, querying, and various search methods. It handles
    connection management, error handling, and data serialization automatically.
    
    Attributes:
        url (str): The Weaviate instance URL
        api_key (Optional[str]): API key for authentication (if required)
        timeout (int): Connection timeout in seconds
        _client (Optional[weaviate.WeaviateClient]): Internal Weaviate client instance
    """
    
    def __init__(self, url: str, api_key: Optional[str] = None, timeout: int = 60):
        """
        Initialize the Weaviate client.
        
        Args:
            url (str): Weaviate instance URL (e.g., "https://localhost:8080")
            api_key (Optional[str]): API key for authentication. Required for cloud instances.
            timeout (int): Connection timeout in seconds. Defaults to 60.
            
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[weaviate.WeaviateClient] = None

    def connect(self) -> weaviate.WeaviateClient:
        """
        Establish connection to Weaviate instance.
        
        This method creates and returns a connection to the Weaviate instance.
        It automatically detects whether the instance is cloud-hosted or self-hosted
        and uses the appropriate connection method. The connection is cached and reused
        for subsequent operations.
        
        Returns:
            weaviate.WeaviateClient: Connected Weaviate client instance
            
        Raises:
            ConnectionError: If Weaviate instance is not ready or connection fails
        """
        if self._client is not None:
            return self._client

        # Parse the URL to extract host and port
        p = urlparse(self.url)
        host = p.hostname or self.url.replace("https://", "").replace("http://", "")
        
        # Check if it's a cloud instance
        if host.endswith(".weaviate.cloud"):
            # Use connect_to_weaviate_cloud for cloud instances
            auth = Auth.api_key(self.api_key) if self.api_key else None
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=auth,
            )
        else:
            # Use connect_to_custom for self-hosted instances
            http_secure = (p.scheme == "https")
            http_port = p.port or (443 if http_secure else 80)
            grpc_port = 50051  # Different port for gRPC
            
            auth = Auth.api_key(self.api_key) if self.api_key else None
            self._client = weaviate.connect_to_custom(
                http_host=host,
                http_port=http_port,
                http_secure=http_secure,
                grpc_host=host,
                grpc_port=grpc_port,
                grpc_secure=http_secure,
                auth_credentials=auth,
            )

        if not self._client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        return self._client

    def disconnect(self) -> None:
        """
        Disconnect from Weaviate instance and clean up resources.
        
        This method closes the connection to Weaviate and cleans up internal resources.
        It's safe to call this method multiple times. After disconnection, the client
        will automatically reconnect on the next operation.
        
        """
        try:
            if self._client:
                self._client.close()
        except Exception:
            logger.debug("Client close failed quietly", exc_info=True)
        finally:
            self._client = None

    # ---------- Collections / Schema ----------

    def list_collections(self) -> List[str]:
        """
        Retrieve a list of all collection names in the Weaviate instance.
        
        Returns:
            List[str]: List of collection names. Returns empty list on error.
        """
        try:
            client = self.connect()
            infos = client.collections.list_all()
            # list_all() may return list[str] or richer objects; normalize to names
            names = []
            for it in infos:
                name = getattr(it, "name", None) or str(it)
                names.append(name)
            return names
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def get_collection_schema(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the schema configuration for a specific collection.
        
        Args:
            class_name (str): Name of the collection to get schema for
            
        Returns:
            Optional[Dict[str, Any]]: Collection schema including properties and vectorizers.
                                     Returns None on error.
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            cfg = col.config.get()
            # cfg is a structured object; convert to dict if needed
            out = {
                "name": getattr(cfg, "name", class_name),
                "properties": [vars(p) for p in (getattr(cfg, "properties", []) or [])],
                "vectorizers": getattr(cfg, "vectorizers", None),
            }
            return out
        except Exception as e:
            logger.error(f"Error getting schema for {class_name}: {e}")
            return None

    def create_collection(
        self,
        class_name: str,
        properties: List[Dict[str, Any]],
        vectorizer: Optional[str] = None,
        vector_index_config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        multi_tenancy: Optional[bool] = None,
    ) -> bool:
        """
        Create a new collection with specified properties and configuration.
        
        Args:
            class_name (str): Name of the collection to create
            properties (List[Dict[str, Any]]): List of property definitions. Each dict should contain:
                - name (str): Property name
                - data_type (str): Data type (TEXT, INT, NUMBER, BOOLEAN, DATE, etc.)
                - tokenization (str, optional): Tokenization method
            vectorizer (Optional[str]): Vectorizer type ("text2vec-openai", "text2vec-transformers", etc.)
            vector_index_config (Optional[Dict[str, Any]]): Vector index configuration
            description (Optional[str]): Collection description
            multi_tenancy (Optional[bool]): Enable multi-tenancy for this collection
            
        Returns:
            bool: True if collection was created successfully, False otherwise
        """
        try:
            client = self.connect()
            props: List[Property] = []
            for p in properties:
                dt_raw = (p.get("data_type") or "text").upper()
                dt = getattr(DataType, dt_raw, DataType.TEXT)
                token = p.get("tokenization")
                tok = getattr(Tokenization, token.upper(), None) if token else None
                props.append(Property(name=p["name"], data_type=dt, tokenization=tok))

            if vectorizer and vectorizer.lower() == "text2vec-openai":
                vec_cfg = Configure.Vectorizer.text2vec_openai()
            elif vectorizer and vectorizer.lower() == "text2vec-transformers":
                vec_cfg = Configure.Vectorizer.text2vec_transformers()
            else:
                vec_cfg = Configure.Vectors.self_provided()

            # Multi-tenancy configuration
            multi_tenancy_cfg = None
            if multi_tenancy is not None:
                multi_tenancy_cfg = Configure.multi_tenancy(enabled=bool(multi_tenancy))

            client.collections.create(
                name=class_name,
                properties=props,
                vector_config=vec_cfg,
                description=description,
                multi_tenancy_config=multi_tenancy_cfg,
            )
            
            # If vector_index_config is provided, update it post-creation
            if vector_index_config:
                col = client.collections.use(class_name)
                col.config.update(vector_index_config=vector_index_config)
            
            return True
        except Exception as e:
            logger.error(f"Error creating collection {class_name}: {e}")
            return False

    def delete_collection(self, class_name: str) -> bool:
        """
        Delete a collection and all its data.
        
        Args:
            class_name (str): Name of the collection to delete
            
        Returns:
            bool: True if collection was deleted successfully, False otherwise
        """
        try:
            client = self.connect()
            if client.collections.exists(class_name):
                client.collections.delete(class_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {class_name}: {e}")
            return False

    def get_collection_stats(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a collection including object count.
        
        Args:
            class_name (str): Name of the collection to get stats for
            
        Returns:
            Optional[Dict[str, Any]]: Collection statistics including total_count.
                                     Returns None on error.
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            agg = col.aggregate.over_all(total_count=True)
            res = agg.do()
            total = getattr(res, "total_count", None)
            return {"class_name": class_name, "total_count": total}
        except Exception as e:
            logger.error(f"Error getting stats for {class_name}: {e}")
            return None

    def collection_exists(self, class_name: str) -> bool:
        """
        Check if a collection exists in the Weaviate instance.
        
        Args:
            class_name (str): Name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            client = self.connect()
            return client.collections.exists(class_name)
        except Exception as e:
            logger.error(f"Error checking if collection {class_name} exists: {e}")
            return False

    def add_property(self, class_name: str, prop: Dict[str, Any]) -> bool:
        """
        Add a new property to an existing collection.
        
        Args:
            class_name (str): Name of the collection to add property to
            prop (Dict[str, Any]): Property definition containing:
                - name (str): Property name
                - data_type (str): Data type (TEXT, INT, NUMBER, etc.)
                - tokenization (str, optional): Tokenization method
                
        Returns:
            bool: True if property was added successfully, False otherwise
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            dt_raw = (prop.get("data_type") or "text").upper()
            dt = getattr(DataType, dt_raw, DataType.TEXT)
            
            tok = None
            if prop.get("tokenization"):
                tok = getattr(Tokenization, prop["tokenization"].upper(), None)
            
            col.config.add_property(Property(
                name=prop["name"],
                data_type=dt,
                tokenization=tok
            ))
            return True
        except Exception as e:
            logger.error(f"Error adding property to {class_name}: {e}")
            return False

    def update_collection_config(self, class_name: str, cfg_updates: Dict[str, Any]) -> bool:
        """
        Update collection configuration (limited to updatable fields).
        
        Args:
            class_name (str): Name of the collection to update
            cfg_updates (Dict[str, Any]): Configuration updates to apply
            
        Returns:
            bool: True if configuration was updated successfully, False otherwise
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            # Only allow fields that Weaviate supports updating post-creation
            # Common ones: vector index params, description, etc.
            col.config.update(**cfg_updates)
            return True
        except Exception as e:
            logger.error(f"Error updating config for {class_name}: {e}")
            return False

    # ---------- Data Objects ----------

    def insert_objects(self, class_name: str, objects: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple objects into a collection.
        
        Args:
            class_name (str): Name of the collection to insert objects into
            objects (List[Dict[str, Any]]): List of objects to insert. Each object can contain:
                - id (str, optional): Custom UUID for the object
                - properties (Dict[str, Any]): Object properties
                - vector (List[float], optional): Custom vector for the object
                
        Returns:
            List[str]: List of UUIDs of successfully inserted objects

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            uuids_out: List[str] = []
            
            # Use individual insertions to get UUIDs back
            for i, obj in enumerate(objects):
                if not isinstance(obj, dict):
                    logger.warning(f"Skipping non-dict object at index {i}: {obj}")
                    continue
                    
                uid = obj.get("id")
                props = obj.get("properties") or obj  # allow raw property dicts
                vec = obj.get("vector")
                vec_payload = {"default": vec} if isinstance(vec, list) else None
                
                try:
                    # Insert individual object and get the UUID back
                    result = col.data.insert(
                        properties=props,
                        uuid=uid,
                        vector=vec_payload
                    )
                    
                    # The result IS the UUID object, convert it to string
                    inserted_uuid = str(result)
                    uuids_out.append(inserted_uuid)
                    logger.debug(f"Successfully inserted object {i+1}/{len(objects)} with UUID: {inserted_uuid}")
                    
                except Exception as obj_error:
                    logger.error(f"Failed to insert object {i+1}/{len(objects)}: {obj_error}")
                    logger.error(f"Object data: {obj}")
                    raise obj_error  # Re-raise to be caught by outer try-catch
            
            return uuids_out
        except Exception as e:
            logger.error(f"Error inserting objects to {class_name}: {e}")
            logger.error(f"Collection exists: {client.collections.exists(class_name) if 'client' in locals() else 'Unknown'}")
            raise e  # Re-raise the exception with details

    def update_object(self, class_name: str, uuid: str, properties: Dict[str, Any]) -> bool:
        """
        Update properties of an existing object.
        
        Args:
            class_name (str): Name of the collection containing the object
            uuid (str): UUID of the object to update
            properties (Dict[str, Any]): New properties to set
            
        Returns:
            bool: True if object was updated successfully, False otherwise

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            col.data.update(uuid=uuid, properties=properties)
            return True
        except UnexpectedStatusCodeError as e:
            if getattr(e, "status_code", None) == 404:
                return False
            logger.error(f"Update error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error updating object {uuid} in {class_name}: {e}")
            return False

    def delete_object(self, class_name: str, uuid: str) -> bool:
        """
        Delete an object by its UUID.
        
        Args:
            class_name (str): Name of the collection containing the object
            uuid (str): UUID of the object to delete
            
        Returns:
            bool: True if object was deleted successfully, False otherwise

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            col.data.delete_by_id(uuid)
            return True
        except UnexpectedStatusCodeError as e:
            if getattr(e, "status_code", None) == 404:
                return False
            logger.error(f"Delete error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting object {uuid} from {class_name}: {e}")
            return False

    def get_object(self, class_name: str, uuid: str, return_properties: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific object by its UUID.
        
        Args:
            class_name (str): Name of the collection containing the object
            uuid (str): UUID of the object to retrieve
            return_properties (Optional[List[str]]): Specific properties to return
            
        Returns:
            Optional[Dict[str, Any]]: Object data including UUID, properties, and metadata.
                                     Returns None if object not found.

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            obj = col.query.fetch_object_by_id(uuid, return_properties=return_properties)
            if obj is None:
                return None
            return {
                "uuid": str(getattr(obj, "uuid", uuid)),
                "properties": getattr(obj, "properties", None),
                "metadata": getattr(obj, "metadata", None),
            }
        except Exception as e:
            logger.error(f"Error getting object {uuid} from {class_name}: {e}")
            return None

    # ---------- Queries ----------

    def _build_where(self, where_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """
        Build a proper Weaviate Filter object from various input formats.
        
        This internal method converts simple dictionary filters into Weaviate Filter objects
        that can be used in queries.
        
        Args:
            where_filter (Optional[Dict[str, Any]]): Filter conditions
            
        Returns:
            Optional[Filter]: Weaviate Filter object or None
        """
        if not where_filter:
            return None
            
        # If it's already a Filter object, return as-is
        if hasattr(where_filter, 'to_dict'):
            return where_filter
            
        # Convert simple {field: value} format to Filter
        if len(where_filter) == 1:
            field, value = next(iter(where_filter.items()))
            return Filter.by_property(field).equal(value)
        
        # For multiple conditions, create an AND filter
        conditions = []
        for field, value in where_filter.items():
            conditions.append(Filter.by_property(field).equal(value))
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return Filter.and_filters(*conditions)

    def vector_search(
        self,
        class_name: str,
        query_vector: List[float],
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
        target_vector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in a collection.
        
        Args:
            class_name (str): Name of the collection to search
            query_vector (List[float]): Vector to search for similar objects
            limit (int): Maximum number of results to return (default: 10)
            where_filter (Optional[Dict[str, Any]]): Additional filter conditions
            return_properties (Optional[List[str]]): Specific properties to return
            target_vector (Optional[str]): Target vector name (default: "default")
            
        Returns:
            List[Dict[str, Any]]: List of similar objects with metadata including distance

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            res = col.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                filters=self._build_where(where_filter),
                return_properties=return_properties,
                return_metadata=MetadataQuery(distance=True),
                include_vector=False,
                target_vector=target_vector or "default",
            )
            return [
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", None),
                    "metadata": {"distance": getattr(getattr(o, "metadata", None), "distance", None)},
                }
                for o in (getattr(res, "objects", []) or [])
            ]
        except Exception as e:
            logger.error(f"Error performing vector search in {class_name}: {e}")
            return []

    def hybrid_search(
        self,
        class_name: str,
        query: str,
        query_vector: Optional[List[float]] = None,
        alpha: float = 0.7,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
        target_vector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining text and vector search.
        
        This method combines BM25 text search with vector similarity search for
        more comprehensive results. The alpha parameter controls the balance between
        text and vector search (0.0 = text only, 1.0 = vector only).
        
        Args:
            class_name (str): Name of the collection to search
            query (str): Text query for BM25 search
            query_vector (Optional[List[float]]): Vector for similarity search
            alpha (float): Balance between text and vector search (0.0-1.0, default: 0.7)
            limit (int): Maximum number of results to return (default: 10)
            where_filter (Optional[Dict[str, Any]]): Additional filter conditions
            return_properties (Optional[List[str]]): Specific properties to return
            target_vector (Optional[str]): Target vector name (default: "default")
            
        Returns:
            List[Dict[str, Any]]: List of search results with metadata including score

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            # If no query_vector provided, use text search only
            if query_vector is None:
                res = col.query.bm25(
                    query=query,
                    limit=limit,
                    filters=self._build_where(where_filter),
                    return_properties=return_properties,
                    include_vector=False,
                )
                return [
                    {
                        "uuid": str(getattr(o, "uuid", "")),
                        "properties": getattr(o, "properties", None),
                        "metadata": {"score": 1.0},  # Default score for text-only search
                    }
                    for o in (getattr(res, "objects", []) or [])
                ]
            
            # If no query provided, use vector search only
            if not query.strip():
                res = col.query.near_vector(
                    near_vector=query_vector,
                    limit=limit,
                    filters=self._build_where(where_filter),
                    return_properties=return_properties,
                    return_metadata=MetadataQuery(distance=True),
                    include_vector=False,
                    target_vector=target_vector or "default",
                )
                return [
                    {
                        "uuid": str(getattr(o, "uuid", "")),
                        "properties": getattr(o, "properties", None),
                        "metadata": {"distance": getattr(getattr(o, "metadata", None), "distance", None)},
                    }
                    for o in (getattr(res, "objects", []) or [])
                ]
            
            # Both query and vector provided - true hybrid search
            res = col.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=self._build_where(where_filter),
                return_properties=return_properties,
                return_metadata=MetadataQuery(score=True),
                target_vector=target_vector or "default",
            )
            return [
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", None),
                    "metadata": {"score": getattr(getattr(o, "metadata", None), "score", None)},
                }
                for o in (getattr(res, "objects", []) or [])
            ]
        except Exception as e:
            logger.error(f"Error performing hybrid search in {class_name}: {e}")
            return []

    def text_search(
        self,
        class_name: str,
        query: str,
        limit: int = 10,
        where_filter: Optional[Dict[str, Any]] = None,
        return_properties: Optional[List[str]] = None,
        search_properties: Optional[List[str]] = None,
        target_vector: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 text search in a collection.
        
        Args:
            class_name (str): Name of the collection to search
            query (str): Text query for BM25 search
            limit (int): Maximum number of results to return (default: 10)
            where_filter (Optional[Dict[str, Any]]): Additional filter conditions
            return_properties (Optional[List[str]]): Specific properties to return
            search_properties (Optional[List[str]]): Properties to search in
            target_vector (Optional[str]): Target vector name (default: "default")
            
        Returns:
            List[Dict[str, Any]]: List of search results

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            res = col.query.bm25(
                query=query,
                limit=limit,
                filters=self._build_where(where_filter),
                return_properties=return_properties,
                query_properties=search_properties,  # restrict BM25 fields if provided
                include_vector=False,
            )
            return [
                {
                    "uuid": str(getattr(o, "uuid", "")),
                    "properties": getattr(o, "properties", None),
                    "metadata": {},  # BM25 may not include score by default; add if you want with MetadataQuery
                }
                for o in (getattr(res, "objects", []) or [])
            ]
        except Exception as e:
            logger.error(f"Error performing text search in {class_name}: {e}")
            return []

    def insert_objects_batch(
        self, 
        class_name: str, 
        objects: List[Dict[str, Any]], 
        tenant: Optional[str] = None,
        batch_size: int = 100,
        consistency_level: Optional[str] = None,
        conflict_mode: str = "error"
    ) -> Dict[str, Any]:
        """
        Insert objects in batch with advanced options for better performance.
        
        Args:
            class_name (str): Name of the collection to insert objects into
            objects (List[Dict[str, Any]]): List of objects to insert
            tenant (Optional[str]): Tenant name for multi-tenant collections
            batch_size (int): Batch size for processing (default: 100)
            consistency_level (Optional[str]): Consistency level for the operation
            conflict_mode (str): How to handle conflicts ("error", "replace", "skip")
            
        Returns:
            Dict[str, Any]: Batch operation results including counts and errors

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            if tenant:
                col = col.with_tenant(tenant)
            
            inserted_count = 0
            failed_count = 0
            errors = []
            
            with col.data.batch.dynamic() as batch:
                for i, obj in enumerate(objects):
                    try:
                        uid = obj.get("id")
                        props = obj.get("properties") or obj
                        vec = obj.get("vector")
                        vec_payload = {"default": vec} if isinstance(vec, list) else None
                        
                        batch.add_object(
                            properties=props,
                            uuid=uid,
                            vector=vec_payload
                        )
                        inserted_count += 1
                        
                    except Exception as e:
                        failed_count += 1
                        errors.append({
                            "index": i,
                            "error": str(e),
                            "object": obj
                        })
            
            return {
                "inserted_count": inserted_count,
                "failed_count": failed_count,
                "errors": errors,
                "total_processed": len(objects)
            }
            
        except Exception as e:
            logger.error(f"Error in batch insert for {class_name}: {e}")
            return {
                "inserted_count": 0,
                "failed_count": len(objects),
                "errors": [{"index": 0, "error": str(e), "object": None}],
                "total_processed": len(objects)
            }

    def list_objects(
        self,
        class_name: str,
        where_filter: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        after: Optional[str] = None,
        sort: Optional[List[Dict[str, str]]] = None,
        tenant: Optional[str] = None,
        return_properties: Optional[List[str]] = None,
        include_vector: bool = False,
        return_additional: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List objects from a collection with filtering, pagination, and sorting.
        
        Args:
            class_name (str): Name of the collection to query
            where_filter (Optional[Dict[str, Any]]): Filter conditions
            limit (int): Maximum number of results to return (default: 100)
            after (Optional[str]): Cursor for pagination
            sort (Optional[List[Dict[str, str]]]): Sort configuration
            tenant (Optional[str]): Tenant name for multi-tenant collections
            return_properties (Optional[List[str]]): Specific properties to return
            include_vector (bool): Whether to include vector data
            return_additional (Optional[List[str]]): Additional metadata to return
            
        Returns:
            List[Dict[str, Any]]: List of objects with their properties and metadata

        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            if tenant:
                col = col.with_tenant(tenant)
            
            # Build metadata query
            metadata_fields = []
            if return_additional:
                metadata_fields.extend(return_additional)
            if include_vector:
                metadata_fields.append("vector")
            
            metadata_query = MetadataQuery(*metadata_fields) if metadata_fields else None
            
            # Build proper where filter using Filter class
            built_filters = self._build_where(where_filter)
            
            # Execute query with correct parameters
            result = col.query.fetch_objects(
                filters=built_filters,  # Changed from where=where_filter
                limit=limit,
                after=after,
                sort=sort,
                return_properties=return_properties,
                return_metadata=metadata_query,
                include_vector=include_vector
            )
            
            objects = []
            for obj in result.objects:
                obj_data = {
                    "uuid": str(obj.uuid),
                    "properties": obj.properties,
                    "metadata": {}
                }
                
                if obj.metadata:
                    if include_vector and hasattr(obj.metadata, 'vector'):
                        obj_data["vector"] = obj.metadata.vector
                    if return_additional:
                        for field in return_additional:
                            if hasattr(obj.metadata, field):
                                obj_data["metadata"][field] = getattr(obj.metadata, field)
                
                # Filter return properties if specified
                if return_properties:
                    filtered_properties = {}
                    for prop in return_properties:
                        if prop in obj.properties:
                            filtered_properties[prop] = obj.properties[prop]
                    obj_data["properties"] = filtered_properties
                
                objects.append(obj_data)
            
            return objects
            
        except Exception as e:
            logger.error(f"Error listing objects from {class_name}: {e}")
            return []

    def delete_by_filter(
        self,
        class_name: str,
        where_filter: Dict[str, Any],
        tenant: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Delete objects by filter with optional dry run.
        
        Args:
            class_name (str): Name of the collection to delete from
            where_filter (Dict[str, Any]): Filter conditions for objects to delete
            tenant (Optional[str]): Tenant name for multi-tenant collections
            dry_run (bool): If True, only count objects that would be deleted
            
        Returns:
            Dict[str, Any]: Deletion results including counts and dry run status
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            if tenant:
                col = col.with_tenant(tenant)
            
            if dry_run:
                # Count objects that would be deleted
                result = col.query.fetch_objects(where=where_filter, limit=1000)
                count = len(result.objects)
                return {
                    "deleted_count": 0,
                    "would_delete_count": count,
                    "dry_run": True
                }
            else:
                # Actually delete
                result = col.data.delete_many(where=where_filter)
                return {
                    "deleted_count": result.successful,
                    "failed_count": result.failed,
                    "dry_run": False
                }
                
        except Exception as e:
            logger.error(f"Error deleting objects by filter from {class_name}: {e}")
            return {
                "deleted_count": 0,
                "failed_count": 1,
                "error": str(e),
                "dry_run": dry_run
            }

    def replace_object(
        self,
        class_name: str,
        uuid: str,
        properties: Dict[str, Any],
        vector: Optional[List[float]] = None,
        tenant: Optional[str] = None
    ) -> bool:
        """
        Replace entire object (overwrite all properties).
        
        Args:
            class_name (str): Name of the collection containing the object
            uuid (str): UUID of the object to replace
            properties (Dict[str, Any]): New properties for the object
            vector (Optional[List[float]]): New vector for the object
            tenant (Optional[str]): Tenant name for multi-tenant collections
            
        Returns:
            bool: True if object was replaced successfully, False otherwise
            
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            if tenant:
                col = col.with_tenant(tenant)
            
            vec_payload = {"default": vector} if vector else None
            col.data.replace(
                uuid=uuid,
                properties=properties,
                vector=vec_payload
            )
            return True
            
        except Exception as e:
            logger.error(f"Error replacing object {uuid} in {class_name}: {e}")
            return False

    def update_vector(
        self,
        class_name: str,
        uuid: str,
        vector: List[float],
        tenant: Optional[str] = None
    ) -> bool:
        """
        Update only the vector of an object.
        
        Args:
            class_name (str): Name of the collection containing the object
            uuid (str): UUID of the object to update
            vector (List[float]): New vector for the object
            tenant (Optional[str]): Tenant name for multi-tenant collections
            
        Returns:
            bool: True if vector was updated successfully, False otherwise
            
        """
        try:
            client = self.connect()
            col = client.collections.use(class_name)
            
            if tenant:
                col = col.with_tenant(tenant)
            
            col.data.update(
                uuid=uuid,
                vector={"default": vector}
            )
            return True
            
        except Exception as e:
            logger.error(f"Error updating vector for object {uuid} in {class_name}: {e}")
            return False