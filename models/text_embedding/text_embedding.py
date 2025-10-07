from typing import Any, List, Optional
import logging
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import MetadataQuery

from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult
from dify_plugin.entities.model.text_embedding import EmbeddingUsage

logger = logging.getLogger(__name__)

class WeaviateTextEmbeddingModel(TextEmbeddingModel):
    def _invoke(self, model: str, credentials: dict[str, Any], texts: list[str], user: str) -> TextEmbeddingResult:
        try:
            dimensions = self.model_schema.parameters.get('dimensions', 1536)
            vectorizer = self.model_schema.parameters.get('vectorizer', 'text2vec-openai')
            model_name = self.model_schema.parameters.get('model_name', 'text-embedding-ada-002')
            
            client = weaviate.connect_to_local(
                url=credentials['url'],
                auth_credentials=weaviate.AuthApiKey(api_key=credentials.get('api_key')) if credentials.get('api_key') else None,
                timeout_config=(5, 30)
            )
            
            try:
                # Use the first text from the input list
                text_to_embed = texts[0] if texts else ""
                temp_collection_name = f"temp_embedding_{hash(text_to_embed) % 10000}"
                
                collection_config = {
                    'properties': [
                        Property(
                            name='text',
                            data_type=DataType.TEXT,
                            description='Text to embed'
                        )
                    ],
                    'vectorizer_config': vectorizer,
                    'vector_index_config': Configure.VectorIndex.hnsw(
                        distance_metric=weaviate.classes.config.VectorDistances.COSINE
                    )
                }
                
                if vectorizer == 'text2vec-openai':
                    collection_config['vectorizer_config'] = weaviate.classes.config.Vectorizer.text2vec_openai(
                        model=model_name,
                        dimensions=dimensions
                    )
                elif vectorizer == 'text2vec-cohere':
                    collection_config['vectorizer_config'] = weaviate.classes.config.Vectorizer.text2vec_cohere(
                        model=model_name,
                        dimensions=dimensions
                    )
                elif vectorizer == 'text2vec-huggingface':
                    collection_config['vectorizer_config'] = weaviate.classes.config.Vectorizer.text2vec_huggingface(
                        model=model_name,
                        dimensions=dimensions
                    )
                elif vectorizer == 'text2vec-transformers':
                    collection_config['vectorizer_config'] = weaviate.classes.config.Vectorizer.text2vec_transformers(
                        model=model_name,
                        dimensions=dimensions
                    )
                elif vectorizer == 'text2vec-contextionary':
                    collection_config['vectorizer_config'] = weaviate.classes.config.Vectorizer.text2vec_contextionary(
                        vectorize_collection_name=False
                    )
                
                client.collections.create(temp_collection_name, **collection_config)
                
                collection = client.collections.get(temp_collection_name)
                collection.data.insert({'text': text_to_embed})
                
                result = collection.query.near_text(
                    query=text_to_embed,
                    limit=1,
                    return_metadata=MetadataQuery(distance=True)
                ).do()
                
                if result.objects:
                    embedding = result.objects[0].vector['default']
                    client.collections.delete(temp_collection_name)
                    client.close()
                    
                    return self.create_embedding_message(embedding)
                else:
                    client.collections.delete(temp_collection_name)
                    client.close()
                    raise Exception("Failed to generate embedding")
                    
            except Exception as e:
                try:
                    client.collections.delete(temp_collection_name)
                except:
                    pass
                client.close()
                raise e
                
        except Exception as e:
            logger.error(f"Text embedding error: {str(e)}")
            raise Exception(f"Failed to generate text embedding: {str(e)}")
    
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            url = credentials.get('url', '')
            api_key = credentials.get('api_key', '')
            
            if not url:
                raise ValueError("Weaviate URL is required")
            
            if api_key and not isinstance(api_key, str):
                raise ValueError("API key must be a string")
            
            client = weaviate.connect_to_local(
                url=url,
                auth_credentials=weaviate.AuthApiKey(api_key=api_key) if api_key else None,
                timeout_config=(5, 10)
            )
            client.close()
            
        except Exception as e:
            raise ValueError(f"Invalid Weaviate credentials: {str(e)}")
