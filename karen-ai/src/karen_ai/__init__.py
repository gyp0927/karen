"""karen-ai: unified LLM API with provider configuration and streaming.

A Python port of `@earendil-works/pi-ai`'s core layer. See README.md for the
architecture map and scope notes.
"""

from .abort import AbortController, AbortSignal, abortable_sleep
from .errors import AbortError, ModelsError
from .event_stream import AssistantMessageEventStream, EventStream, create_assistant_message_event_stream
from .lazy import ProviderStreams, lazy_stream
from .models import (
    DELETE,
    CreateModelsOptions,
    CreateProviderOptions,
    Models,
    ModelsPublication,
    ModelsRefreshOptions,
    ModelsRefreshResult,
    Provider,
    RefreshModelsContext,
    calculate_cost,
    clamp_thinking_level,
    create_models,
    create_provider,
    get_supported_thinking_levels,
    has_api,
    models_are_equal,
)
from .models_store import InMemoryModelsStore, JsonFileModelsStore, ModelsStoreEntry
from .model_catalog import (
    catalog_available,
    catalog_generated_at,
    flatten_all_model_catalog,
    flatten_chat_model_catalog,
    flatten_classifier_model_catalog,
    flatten_image_model_catalog,
    list_catalog_provider_ids,
    load_catalog_groups,
)
from .utils.retry import (
    DEFAULT_MAX_AGENT_RETRY_DELAY_MS,
    ProviderHttpError,
    RetryCallbacks,
    RetryPolicy,
    is_retryable_assistant_error,
    retry_assistant_call,
    retry_delay_ms,
    retry_provider_request,
)
from .utils.validation import validate_tool_arguments, validate_tool_call
from .transcript import (
    collapse_system_messages,
    create_initial_system_message,
    get_current_system_message,
    get_current_system_prompt,
    get_current_tools,
    get_declared_tools,
    get_initial_system_message,
    get_tool_state_changes,
    has_non_additive_tool_changes,
    has_tool_redefinitions,
    normalize_context,
    resolve_transcript,
    resolve_transcript_tools,
    to_tool_declaration,
    without_initial_system_message,
)
from .types import *  # noqa: F403 — the type system is the public surface
from .types import (
    AnyModel,
    AssistantMessage,
    AssistantMessageEvent,
    ClassifierModel,
    Context,
    ImageModel,
    Message,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    Tool,
    ToolCall,
    TranscriptContext,
    Usage,
    get_model_type,
    is_model_type,
)
from .auth import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthResult,
    Credential,
    CredentialStore,
    InMemoryCredentialStore,
    JsonFileCredentialStore,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
    resolve_provider_auth,
)

__version__ = "0.1.0"
