# tools/generative_search.py
"""
Generative Search Tool for Weaviate Plugin

This module provides a comprehensive generative search tool that combines Weaviate vector
database search capabilities with Large Language Model (LLM) generation. It supports
Retrieval-Augmented Generation (RAG) patterns with various search modes and LLM providers.

The tool supports multiple search modes (BM25, vector search, hybrid), various LLM providers
(OpenAI, Anthropic), and includes support for multimodal inputs including text and images.

Classes:
    GenerativeSearchTool: Main tool class for generative search operations

Functions:
    _to_list: Convert various input types to a list of strings
    _parse_query_vector: Parse and validate query vectors from various input formats
    _parse_generative_config: Parse generative configuration from various input formats
    _parse_images: Parse and validate image inputs from various formats
    _validate_base64_image: Validate base64 encoded image data
    _resolve_llm_key: Resolve LLM API key from various sources
    _build_single_prompt: Build prompts for single document processing
    _build_grouped_prompt: Build prompts for grouped document processing
    _extract_images_from_properties: Extract base64 images from document properties
"""

from collections.abc import Generator
from typing import Any, List, Optional, Dict
import logging
import base64

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from utils.client import WeaviateClient
from utils.validators import validate_vector, validate_where_filter
from utils.helpers import create_error_response, create_success_response, safe_json_parse

logger = logging.getLogger(__name__)

# -------------------- small parsing helpers --------------------

def _to_list(value) -> Optional[List[str]]:
    """
    Convert various input types to a list of strings.
    
    This utility function normalizes different input formats (None, list, string)
    into a consistent list of strings format. It handles comma-separated strings,
    existing lists, and filters out empty values.
    
    Parameters:
        value: Input value to convert. Can be None, list, or string.
        
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

def _parse_query_vector(raw) -> Optional[List[float]]:
    """
    Parse and validate query vectors from various input formats.
    
    This function handles multiple input formats for query vectors including
    JSON arrays, comma-separated strings, and Python lists. It validates that
    all elements can be converted to floats and returns None for invalid inputs.
    
    Parameters:
        raw: Input vector data. Can be None, list, or string.
            - List: Direct list of numbers
            - String: JSON array string or comma-separated values
            - None: Returns None
            
    Returns:
        Optional[List[float]]: Parsed vector as list of floats, or None if invalid.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [float(x) for x in raw]
        except Exception:
            return None
    s = str(raw).strip()
    if not s:
        return None
    if s.startswith("["):
        arr = safe_json_parse(s)
        if isinstance(arr, list):
            try:
                return [float(x) for x in arr]
            except Exception:
                return None
        return None
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except Exception:
        return None

def _parse_generative_config(raw) -> Dict[str, Any]:
    """
    Parse generative configuration from various input formats.
    
    This function handles configuration objects for LLM generation parameters
    including temperature, max_tokens, top_p, and other generation settings.
    It accepts dictionaries directly or JSON strings that parse to dictionaries.
    
    Parameters:
        raw: Configuration input. Can be None, dict, or JSON string.
            - Dict: Direct configuration dictionary
            - String: JSON string that parses to dictionary
            - None: Returns empty dictionary
            
    Returns:
        Dict[str, Any]: Configuration dictionary with generation parameters.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        cfg = safe_json_parse(raw)
        return cfg if isinstance(cfg, dict) else {}
    return {}

def _parse_images(raw) -> Optional[List[str]]:
    """
    Parse and validate image inputs from various formats.
    
    This function handles image inputs for multimodal LLM generation.
    It supports base64 encoded images in various input formats including
    single strings, JSON arrays, and Python lists.
    
    Parameters:
        raw: Image input data. Can be None, list, or string.
            - List: List of image strings
            - String: Single image string or JSON array string
            - None: Returns None
            
    Returns:
        Optional[List[str]]: List of image strings, or None if no valid images found.
    """
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(img) for img in raw if str(img).strip()]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s.startswith("["):
            arr = safe_json_parse(s)
            if isinstance(arr, list):
                return [str(img) for img in arr if str(img).strip()]
        return [s]
    return None

def _validate_base64_image(img_str: str) -> bool:
    """
    Validate base64 encoded image data.
    
    This function checks if a string represents valid base64 encoded image data
    by attempting to decode it and verifying it produces non-empty binary data.
    
    Parameters:
        img_str: String to validate as base64 image data.
        
    Returns:
        bool: True if the string is valid base64 image data, False otherwise.
    """
    try:
        decoded = base64.b64decode(img_str, validate=True)
        return len(decoded) > 0
    except Exception:
        return False

def _resolve_llm_key(tool_params: dict, plugin_creds: dict, provider: str) -> Optional[str]:
    """
    Resolve LLM API key from various sources with priority order.
    
    This function implements a priority-based key resolution system:
    1. Per-call API key (highest priority)
    2. Plugin-level credentials for specific provider
    3. None if no valid key found
    
    Parameters:
        tool_params: Tool parameters dictionary containing per-call API key.
        plugin_creds: Plugin credentials dictionary containing provider keys.
        provider: LLM provider name (e.g., "openai", "anthropic").
        
    Returns:
        Optional[str]: Resolved API key string, or None if no valid key found.
    """
    # 1) Per-call key wins
    per_call = (tool_params.get("llm_api_key") or "").strip()
    if per_call:
        return per_call

    # 2) Plugin-level credentials next
    provider = (provider or "openai").lower()
    if provider == "openai":
        return (plugin_creds.get("openai_api_key") or "").strip()
    if provider == "anthropic":
        return (plugin_creds.get("anthropic_api_key") or "").strip()

    # add other providers here if you support them later
    return None

def _build_single_prompt(template: str, properties: Dict[str, Any]) -> str:
    """
    Build prompts for single document processing.
    
    This function creates prompts by replacing placeholders in a template
    with values from document properties. It uses simple string replacement
    with curly brace notation for placeholders.
    
    Parameters:
        template: Prompt template string with {property_name} placeholders.
        properties: Dictionary of document properties to substitute.
        
    Returns:
        str: Processed prompt with placeholders replaced by property values.
    """
    if not template:
        return ""
    result = template
    for k, v in (properties or {}).items():
        placeholder = f"{{{k}}}"
        if placeholder in result:
            result = result.replace(placeholder, str(v))
    return result

def _build_grouped_prompt(template: str, all_props: List[Dict[str, Any]], query: str = "") -> str:
    """
    Build prompts for grouped document processing.
    
    This function creates prompts for processing multiple documents together.
    It builds a context section from all document properties and substitutes
    {context} and {query} placeholders in the template.
    
    Parameters:
        template: Prompt template string with {context} and {query} placeholders.
        all_props: List of property dictionaries from all documents.
        query: Optional query string to substitute for {query} placeholder.
        
    Returns:
        str: Processed prompt with context and query placeholders replaced.
    """
    if not template:
        return ""
    # Build context block
    sections = []
    for idx, props in enumerate(all_props, 1):
        lines = []
        for k, v in (props or {}).items():
            if isinstance(v, (str, int, float)) and (str(v).strip() if isinstance(v, str) else True):
                lines.append(f"{k}: {v}")
        if lines:
            sections.append(f"Document {idx}:\n" + "\n".join(lines))
    context = "\n\n".join(sections)

    out = template
    if "{context}" in out:
        out = out.replace("{context}", context)
    else:
        out = f"{out}\n\nContext:\n{context}"
    if "{query}" in out:
        out = out.replace("{query}", query)
    return out

def _extract_images_from_properties(all_props: List[Dict[str, Any]], image_props: Optional[List[str]]) -> List[str]:
    """
    Extract base64 images from document properties.
    
    This function searches through document properties for base64 encoded images
    based on specified property names. It validates images and extracts them
    for use in multimodal LLM generation.
    
    Parameters:
        all_props: List of property dictionaries from all documents.
        image_props: List of property names that may contain images.
        
    Returns:
        List[str]: List of validated base64 image strings found in properties.
    """
    if not image_props:
        return []
    images: List[str] = []
    for props in all_props:
        for name in image_props:
            if name not in props:
                continue
            val = props[name]
            if isinstance(val, str) and len(val) > 100 and _validate_base64_image(val):
                images.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and len(item) > 100 and _validate_base64_image(item):
                        images.append(item)
    return images

# -------------------- main tool --------------------

class GenerativeSearchTool(Tool):
    """
    A comprehensive generative search tool that combines Weaviate vector database
    search capabilities with Large Language Model (LLM) generation for Retrieval-
    Augmented Generation (RAG) applications.
    
    This tool supports multiple search modes including BM25 keyword search, vector
    similarity search, and hybrid search combining both approaches. It integrates
    with various LLM providers (OpenAI, Anthropic) and supports multimodal inputs
    including text and images for enhanced generation capabilities.
    
    The tool implements client-side RAG patterns where it first retrieves relevant
    documents from Weaviate, builds context from the retrieved content, and then
    uses an LLM to generate responses based on the context and user query.
    
    Attributes:
        runtime: Runtime context containing credentials and configuration
    """

    def _invoke(self, tool_parameters: dict[str, Any]) -> Generator[ToolInvokeMessage]:
        """
        Execute generative search operations with LLM-based response generation.
        
        This is the main entry point for the generative search tool that performs
        document retrieval from Weaviate and generates responses using LLM providers.
        It supports various search modes, prompt templates, and multimodal inputs.
        
        Parameters:
            tool_parameters (dict[str, Any]): Dictionary containing search and generation parameters
                - collection_name (str): Name of the Weaviate collection to search
                - query (str): Text query for BM25 and hybrid search modes
                - query_vector (str/list): Vector for vector and hybrid search modes
                - limit (int): Maximum number of documents to retrieve (1-20, default: 5)
                - search_mode (str): Search mode - "bm25", "near_vector", or "hybrid" (default: "bm25")
                - alpha (float): Hybrid search weight (0.0-1.0, default: 0.7)
                - target_vector (str): Target vector name for BM25 search
                - search_properties (str/list): Properties to search in BM25 mode
                - where_filter (str/dict): JSON filter for document filtering
                - return_properties (str/list): Properties to return from documents
                - single_prompt (str): Template for processing each document individually
                - grouped_task (str): Template for processing all documents together
                - generative_config (str/dict): LLM generation parameters
                - images (str/list): Base64 encoded images for multimodal generation
                - image_properties (str/list): Document properties containing images
                - return_metadata (bool): Whether to return generation metadata
                - llm_provider (str): LLM provider - "openai" or "anthropic" (default: "openai")
                - llm_model (str): LLM model name (default: "gpt-4o-mini")
                - llm_api_key (str): LLM API key (optional, uses plugin credentials if not provided)
        
        Yields:
            ToolInvokeMessage: JSON messages containing search results and generated responses
            
        Search Modes:
            bm25: Keyword-based search using BM25 algorithm
                - Requires: query
                - Optional: search_properties, target_vector
                
            near_vector: Vector similarity search
                - Requires: query_vector
                - Uses cosine similarity for document ranking
                
            hybrid: Combines BM25 and vector search
                - Requires: query or query_vector (or both)
                - Uses alpha parameter to weight BM25 vs vector search
                
        Generation Modes:
            single_prompt: Process each retrieved document individually
                - Uses single_prompt template with document properties
                - Returns individual responses for each document
                
            grouped_task: Process all documents together
                - Uses grouped_task template with combined context
                - Returns single response based on all documents
                
        LLM Providers:
            openai: OpenAI GPT models with vision support
                - Supported models: gpt-4o, gpt-4o-mini, gpt-4-vision
                - Vision models required for image inputs
                
            anthropic: Anthropic Claude models
                - Supported models: Claude 3/3.5 variants
                - Vision models required for image inputs
                
        Error Handling:
            Comprehensive validation for:
            - Required parameters based on search mode
            - Vector format and validity
            - Image base64 encoding
            - LLM API key availability
            - Search mode compatibility
            
        Returns:
            Generator[ToolInvokeMessage]: Stream of JSON messages containing:
                - Search results with retrieved documents
                - Generated responses from LLM
                - Context sources and metadata
                - Error messages for failed operations
        """
        try:
            # -------- inputs --------
            collection = (tool_parameters.get("collection_name") or "").strip()
            query = (tool_parameters.get("query") or "").strip()
            query_vector_raw = tool_parameters.get("query_vector")
            limit_raw = tool_parameters.get("limit", 5)
            search_mode = (tool_parameters.get("search_mode") or "bm25").strip().lower()
            alpha_raw = tool_parameters.get("alpha", 0.7)
            target_vector = (tool_parameters.get("target_vector") or "").strip() or None
            search_properties = _to_list(tool_parameters.get("search_properties"))

            # generative
            single_prompt = (tool_parameters.get("single_prompt") or "").strip()
            grouped_task = (tool_parameters.get("grouped_task") or "").strip()
            generative_config = _parse_generative_config(tool_parameters.get("generative_config"))
            images = _parse_images(tool_parameters.get("images"))
            image_properties = _to_list(tool_parameters.get("image_properties"))
            return_metadata = bool(tool_parameters.get("return_metadata", False))

            # llm config
            llm_provider = (tool_parameters.get("llm_provider") or "openai").strip().lower()
            llm_model = (tool_parameters.get("llm_model") or "gpt-4o-mini").strip()

            # resolve key: per-call -> plugin creds -> none
            creds = self.runtime.credentials
            llm_api_key = _resolve_llm_key(tool_parameters, creds, llm_provider)

            where_filter_raw = tool_parameters.get("where_filter")
            return_properties_in = tool_parameters.get("return_properties")

            # -------- basic validation --------
            if not collection:
                yield self.create_json_message(create_error_response("Collection name is required"))
                return

            # parse vector BEFORE validating requirements
            query_vector = _parse_query_vector(query_vector_raw)

            # search_mode checks
            if search_mode not in {"bm25", "near_vector", "hybrid"}:
                yield self.create_json_message(create_error_response("search_mode must be one of: bm25, near_vector, hybrid"))
                return
            if search_mode == "bm25" and not query:
                yield self.create_json_message(create_error_response("Query is required for bm25 search"))
                return
            if search_mode == "near_vector" and not query_vector:
                yield self.create_json_message(create_error_response("query_vector is required for near_vector search"))
                return
            if search_mode == "hybrid" and not (query or query_vector):
                yield self.create_json_message(create_error_response("Provide at least one of query or query_vector for hybrid search"))
                return

            # Validate single/grouped task parameters
            if not single_prompt and not grouped_task:
                # Default grouped task if none provided
                grouped_task = (
                    "Answer the question using ONLY the context.\n\n"
                    "Question: {query}\n\nContext:\n{context}\n\nAnswer:"
                )
            elif single_prompt and grouped_task:
                yield self.create_json_message(create_error_response(
                    "Provide either single_prompt OR grouped_task, not both"
                ))
                return

            # limit
            try:
                limit = int(limit_raw)
            except Exception:
                yield self.create_json_message(create_error_response("limit must be an integer"))
                return
            if not (1 <= limit <= 20):
                yield self.create_json_message(create_error_response("limit must be between 1 and 20"))
                return

            # vector
            if query_vector is not None and not validate_vector(query_vector):
                yield self.create_json_message(create_error_response("query_vector must be a non-empty list of numbers"))
                return

            # alpha
            try:
                alpha = float(alpha_raw)
                if not (0.0 <= alpha <= 1.0):
                    raise ValueError
            except Exception:
                yield self.create_json_message(create_error_response("alpha must be a number between 0 and 1"))
                return

            # where filter
            where_filter = None
            if isinstance(where_filter_raw, str) and where_filter_raw.strip():
                where_filter = safe_json_parse(where_filter_raw.strip())
                if where_filter is None or not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response("Invalid where_filter JSON"))
                    return
            elif isinstance(where_filter_raw, dict):
                where_filter = where_filter_raw
                if not validate_where_filter(where_filter):
                    yield self.create_json_message(create_error_response("Invalid where_filter JSON"))
                    return

            # return props
            return_properties = _to_list(return_properties_in)

            # images sanity
            if images:
                for img in images:
                    if not _validate_base64_image(img):
                        yield self.create_json_message(create_error_response("One or more images are not valid base64"))
                        return
            
            # require LLM key for generation
            if not llm_api_key:
                yield self.create_json_message(create_error_response(
                    "No LLM API key provided. Add `llm_api_key` to the call or configure a provider key in the plugin settings."
                ))
                return

            # -------- connect --------
            creds = self.runtime.credentials
            client = WeaviateClient(url=creds["url"], api_key=creds.get("api_key"), timeout=60)

            try:
                # -------- Search for context --------
                if search_mode == "bm25":
                    results = client.text_search(
                        class_name=collection,
                        query=query,
                        limit=limit,
                        where_filter=where_filter,
                        return_properties=return_properties,
                        search_properties=search_properties,
                        target_vector=target_vector,
                    )
                    search_type = "bm25"
                elif search_mode == "near_vector":
                    results = client.vector_search(
                        class_name=collection,
                        query_vector=query_vector,
                        limit=limit,
                        where_filter=where_filter,
                        return_properties=return_properties,
                    )
                    search_type = "near_vector"
                else:
                    results = client.hybrid_search(
                        class_name=collection,
                        query=query,
                        query_vector=query_vector,
                        alpha=alpha,
                        limit=limit,
                        where_filter=where_filter,
                        return_properties=return_properties,
                    )
                    search_type = "hybrid"

                if not results:
                    yield self.create_json_message(create_success_response(
                        data={
                            "answer": None,
                            "collection": collection,
                            "query": query,
                            "search_mode": search_mode,
                            "search_type": search_type,
                            "context_sources": {"count": 0, "documents": [], "passages": []},
                        },
                        message="No relevant documents found"
                    ))
                    return

                # Build context
                def _props_to_text(doc: dict) -> str:
                    props = doc.get("properties") or {}
                    if not isinstance(props, dict):
                        return str(props)
                    # prefer common text fields
                    for key in ("content", "text", "body", "description"):
                        val = props.get(key)
                        if isinstance(val, str) and val.strip():
                            title = props.get("title")
                            if isinstance(title, str) and title.strip():
                                return f"Title: {title}\nContent: {val.strip()}"
                            return val.strip()
                    # fallback: join simple scalars
                    parts = []
                    if isinstance(props.get("title"), str) and props["title"].strip():
                        parts.append(f"Title: {props['title']}")
                    for k, v in props.items():
                        if k == "title":
                            continue
                        if isinstance(v, (str, int, float)) and (str(v).strip() if isinstance(v, str) else True):
                            parts.append(f"{k}: {v}")
                    return "\n".join(parts) if parts else str(props)

                context_passages = [t for t in (_props_to_text(d) for d in results) if t and t.strip()]
                context_text = "\n\n".join(context_passages)

                if not context_passages:
                    yield self.create_json_message(create_success_response(
                        data={
                            "answer": None,
                            "collection": collection,
                            "query": query,
                            "search_mode": search_mode,
                            "search_type": search_type,
                            "context_sources": {"count": len(results), "documents": results, "passages": []},
                            "note": "Retrieved documents had no usable text content",
                        },
                        message="Retrieved documents had no usable text content"
                    ))
                    return

                # Generate response using LLM
                props_list = [doc.get("properties", {}) for doc in results]
                all_images = (images or []) + _extract_images_from_properties(props_list, image_properties)
                
                # Handle single vs grouped tasks
                single_responses: List[Dict[str, Any]] = []
                grouped_response: Optional[Dict[str, Any]] = None
                gen_meta: Dict[str, Any] = {}
                
                if single_prompt:
                    # Process each document individually
                    for doc in results:
                        props = doc.get("properties") or {}
                        sp_text = _build_single_prompt(single_prompt, props)
                        if not sp_text:
                            continue
                        answer = self._generate_response_llm(
                            query=sp_text, context="", llm_provider=llm_provider, llm_model=llm_model,
                            llm_api_key=llm_api_key, generative_config=generative_config,
                            images=images, return_metadata=return_metadata
                        )
                        if isinstance(answer, dict):
                            single_responses.append({
                                "document": doc,
                                "generated_text": answer.get("text", ""),
                                "metadata": answer.get("metadata", {}),
                            })
                            gen_meta.update(answer.get("metadata", {}) or {})
                        else:
                            single_responses.append({"document": doc, "generated_text": str(answer), "metadata": {}})
                
                if grouped_task:
                    # Process all documents together
                    gp_text = _build_grouped_prompt(grouped_task, props_list, query)
                    g_ans = self._generate_response_llm(
                        query=gp_text, context=None, llm_provider=llm_provider, llm_model=llm_model,
                        llm_api_key=llm_api_key, generative_config=generative_config,
                        images=all_images, return_metadata=return_metadata
                    )
                    if isinstance(g_ans, dict):
                        grouped_response = {"generated_text": g_ans.get("text", ""), "metadata": g_ans.get("metadata", {})}
                        gen_meta.update(g_ans.get("metadata", {}) or {})
                    else:
                        grouped_response = {"generated_text": str(g_ans), "metadata": {}}

                # Determine the main answer
                if single_responses:
                    main_answer = single_responses[0]["generated_text"] if single_responses else None
                else:
                    main_answer = grouped_response["generated_text"] if grouped_response else None

                yield self.create_json_message(create_success_response(
                    data={
                        "answer": main_answer,
                        "collection": collection,
                        "query": query,
                        "search_mode": search_mode,
                        "search_type": search_type,
                        "single_responses": single_responses or None,
                        "grouped_response": grouped_response or None,
                        "context_sources": {"count": len(results), "documents": results, "passages": context_passages},
                        "metadata": {
                            "retrieved_documents": len(results),
                            "context_passages_used": len(context_passages),
                            "generation_metadata": gen_meta or None,
                        },
                    },
                    message=main_answer or "No generation performed"
                ))

            except Exception as e:
                logger.exception("Generative search error")
                yield self.create_json_message(create_error_response(f"Search failed: {e}"))
            finally:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Weaviate client disconnect failed", exc_info=True)

        except Exception as e:
            logger.exception("Tool execution error")
            yield self.create_json_message(create_error_response(f"Tool execution failed: {e}"))

    # -------------------- LLM wrapper for client_rag --------------------

    def _generate_response_llm(
        self,
        query: str,
        context: Optional[str],
        llm_provider: str,
        llm_model: str,
        llm_api_key: str,
        generative_config: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
        return_metadata: bool = False,
    ):
        """
        Generate responses using Large Language Model APIs with multimodal support.
        
        This method interfaces with various LLM providers (OpenAI, Anthropic) to generate
        text responses based on queries and context. It supports both text-only and
        multimodal generation with base64 encoded images.
        
        Parameters:
            query (str): The input query or prompt for generation.
            context (Optional[str]): Additional context to include in the prompt.
            llm_provider (str): LLM provider name ("openai" or "anthropic").
            llm_model (str): Specific model name to use for generation.
            llm_api_key (str): API key for the LLM provider.
            generative_config (Optional[Dict[str, Any]]): Generation parameters including:
                - temperature: Controls randomness (0.0-2.0, default: 0.2)
                - max_tokens: Maximum tokens to generate (default: 700)
                - top_p: Nucleus sampling parameter (0.0-1.0, default: 1.0)
                - frequency_penalty: Frequency penalty (default: 0.0)
                - presence_penalty: Presence penalty (default: 0.0)
            images (Optional[List[str]]): List of base64 encoded images for multimodal generation.
            return_metadata (bool): Whether to return generation metadata including token usage.
            
        Returns:
            Union[str, Dict[str, Any]]: Generated text response or dictionary with text and metadata.
            
        Supported Providers:
            openai: OpenAI GPT models
                - Text models: gpt-4, gpt-3.5-turbo, etc.
                - Vision models: gpt-4o, gpt-4o-mini, gpt-4-vision
                - Images must be base64 encoded with data:image/jpeg;base64, prefix
                
            anthropic: Anthropic Claude models
                - Text models: claude-3-sonnet, claude-3-haiku, etc.
                - Vision models: claude-3-opus, claude-3-sonnet, claude-3-5-sonnet
                - Images must be base64 encoded without data URL prefix
                
        Error Handling:
            - Validates model compatibility with image inputs
            - Handles API errors and network issues
            - Returns appropriate error messages for unsupported configurations
            
        Vision Requirements:
            - Images require vision-capable models
            - OpenAI: Models containing "gpt-4o" or "gpt-4-vision"
            - Anthropic: Models containing "claude-3" or "claude-3-5"
        """
        generative_config = generative_config or {}
        images = images or []

        # Build final prompt
        if context is None:
            prompt = query  # already contains context
        elif context:
            prompt = (
                "You are a helpful assistant. Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know.\n\n"
                f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:"
            )
        else:
            prompt = query

        temperature = float(generative_config.get("temperature", 0.2))
        max_tokens = int(generative_config.get("max_tokens", 700))
        top_p = float(generative_config.get("top_p", 1.0))
        freq_pen = float(generative_config.get("frequency_penalty", 0.0))
        pres_pen = float(generative_config.get("presence_penalty", 0.0))

        # ----- OpenAI -----
        if llm_provider == "openai":
            import openai as _openai  # deferred import
            client = _openai.OpenAI(api_key=llm_api_key)

            # Vision requirement
            if images and not any(m in llm_model.lower() for m in ("gpt-4o", "gpt-4-vision")):
                raise ValueError(f"Model '{llm_model}' does not support images. Use gpt-4o / gpt-4o-mini / gpt-4-vision.")

            # Build messages
            if images:
                content = [{"type": "text", "text": prompt}] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images
                ]
                messages = [{"role": "user", "content": content}]
            else:
                messages = [{"role": "user", "content": prompt}]

            resp = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
            )
            text = (resp.choices[0].message.content or "").strip()
            if return_metadata:
                return {
                    "text": text,
                    "metadata": {
                        "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                        "completion_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                        "total_tokens": getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
                        "model": llm_model,
                        "provider": "openai",
                    },
                }
            return text

        # ----- Anthropic -----
        if llm_provider == "anthropic":
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=llm_api_key)

            # Vision requirement
            if images and not any(m in llm_model.lower() for m in ("claude-3", "claude-3-5")):
                raise ValueError(f"Model '{llm_model}' does not support images. Use Claude 3/3.5 (e.g., claude-3-5-sonnet).")

            if images:
                content = [{"type": "text", "text": prompt}] + [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}}
                    for img in images
                ]
            else:
                content = [{"type": "text", "text": prompt}]

            resp = client.messages.create(
                model=llm_model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=[{"role": "user", "content": content}],
            )
            # resp.content is a list of blocks
            text = "".join(block.get("text", "") for block in (resp.content or [])) if isinstance(resp.content, list) else ""
            text = text.strip()
            if return_metadata:
                usage = getattr(resp, "usage", None)
                in_tok = getattr(usage, "input_tokens", 0) if usage else 0
                out_tok = getattr(usage, "output_tokens", 0) if usage else 0
                return {
                    "text": text,
                    "metadata": {
                        "prompt_tokens": in_tok,
                        "completion_tokens": out_tok,
                        "total_tokens": in_tok + out_tok,
                        "model": llm_model,
                        "provider": "anthropic",
                    },
                }
            return text

        # Fallback (unknown provider)
        return {"text": "", "metadata": {}} if return_metadata else ""