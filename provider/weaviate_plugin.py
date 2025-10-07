from typing import Any
import logging

from dify_plugin import ToolProvider
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

# Prefer absolute import; your try/except fallback is fine if you want to keep it
from utils.validators import validate_weaviate_url, validate_api_key
from utils.client import WeaviateClient

logger = logging.getLogger(__name__)

class WeaviatePluginProvider(ToolProvider):
    """
    Weaviate plugin provider for Dify that validates credentials and manages connections
    to Weaviate vector database instances.
    """

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Validates Weaviate connection credentials by performing format checks and
        establishing a test connection to the Weaviate instance.
        
        This method performs the following validation steps:
        1. Validates URL format using regex patterns
        2. Validates API key format if provided
        3. Establishes a connection to the Weaviate instance
        4. Verifies the instance is ready and responsive
        5. Performs a test API call to validate authentication
        
        Args:
            credentials (dict[str, Any]): Dictionary containing connection credentials
                with the following keys:
                - url (str): Weaviate instance URL (e.g., https://your-instance.com)
                - api_key (str, optional): API key for authentication if required
                
        Raises:
            ToolProviderCredentialValidationError: If any validation step fails:
                - Invalid URL format
                - Invalid API key format
                - Connection failure
                - Instance not ready
                - Authentication failure
                
        Note:
            The method automatically handles connection cleanup and will disconnect
            the client after validation, even if an error occurs during the process.
        """
        url = (credentials.get("url") or "").strip()
        api_key = (credentials.get("api_key") or "").strip()

        # Basic format checks first (fail fast)
        if not validate_weaviate_url(url):
            raise ToolProviderCredentialValidationError(
                "Invalid Weaviate URL. Expected format like https://your-weaviate-instance.com[:port]"
            )
        if api_key and not validate_api_key(api_key):
            raise ToolProviderCredentialValidationError("Invalid API key value.")

        client = None
        try:
            # Use your v4-compliant wrapper (recommended)
            client = WeaviateClient(url=url, api_key=api_key, timeout=15)
            wc = client.connect()

            # Health/auth check:
            # 1) Ensure server is ready
            if not wc.is_ready():
                raise ToolProviderCredentialValidationError(
                    "Weaviate endpoint is reachable but not ready. Please try again later."
                )

            # 2) Simple authorized call (exercises API key permissions)
            #    If auth is wrong, this often raises an error we can surface.
            _ = client.list_collections()

        except ToolProviderCredentialValidationError:
            # Re-raise our own clear errors
            raise

        except Exception as e:
            # Normalize any lower-level errors to a user-friendly message
            msg = str(e) or repr(e)
            # Add a bit more guidance
            raise ToolProviderCredentialValidationError(
                f"Failed to connect to Weaviate at {url}. "
                f"Verify the URL is correct and the API key (if required) is valid. Details: {msg}"
            )
        finally:
            if client:
                try:
                    client.disconnect()
                except Exception:
                    logger.debug("Weaviate client disconnect failed quietly", exc_info=True)

    #########################################################################################
    # If OAuth is supported, uncomment the following functions.
    # Warning: please make sure that the sdk version is 0.4.2 or higher.
    #########################################################################################
    # def _oauth_get_authorization_url(self, redirect_uri: str, system_credentials: Mapping[str, Any]) -> str:
    #     """
    #     Generate the authorization URL for weaviate_plugin OAuth.
    #     """
    #     try:
    #         """
    #         IMPLEMENT YOUR AUTHORIZATION URL GENERATION HERE
    #         """
    #     except Exception as e:
    #         raise ToolProviderOAuthError(str(e))
    #     return ""
        
    # def _oauth_get_credentials(
    #     self, redirect_uri: str, system_credentials: Mapping[str, Any], request: Request
    # ) -> Mapping[str, Any]:
    #     """
    #     Exchange code for access_token.
    #     """
    #     try:
    #         """
    #         IMPLEMENT YOUR CREDENTIALS EXCHANGE HERE
    #         """
    #     except Exception as e:
    #         raise ToolProviderOAuthError(str(e))
    #     return dict()

    # def _oauth_refresh_credentials(
    #     self, redirect_uri: str, system_credentials: Mapping[str, Any], credentials: Mapping[str, Any]
    # ) -> OAuthCredentials:
    #     """
    #     Refresh the credentials
    #     """
    #     return OAuthCredentials(credentials=credentials, expires_at=-1)
