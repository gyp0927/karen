"""Core type system for karen-ai, mirroring pi-ai's types.ts.

Data types are pydantic models. Python attribute names are snake_case; JSON
(field) names match pi-ai's camelCase wire format via aliases, so transcripts
and catalogs serialize identically.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from pydantic.alias_generators import to_camel
from typing_extensions import TypeAliasType

from .abort import AbortSignal

if TYPE_CHECKING:
    from .event_stream import AssistantMessageEventStream

# TypeAliasType makes pydantic emit a recursive $ref instead of expanding forever.
JsonValue = TypeAliasType(
    "JsonValue",
    Union[None, bool, int, float, str, List["JsonValue"], Dict[str, "JsonValue"]],
)
JsonObject = Dict[str, JsonValue]


class KarenBase(BaseModel):
    """Shared base: camelCase JSON aliases, populate by snake_case name."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# API / provider identifiers
# ---------------------------------------------------------------------------

KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
    "pi-messages",
]
# Custom api strings are allowed at runtime.
Api = str

KnownImageApi = Literal["openrouter-images"]
ImageApi = str

KnownClassifierApi = Literal["typesafe-system-one", "cloudflare-workers-ai-system-one"]
ClassifierApi = str

ProviderId = str

ToolChoice = Literal["auto", "none"]
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
ModelThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
# Maps pi thinking levels to provider/model-specific values.
# Missing keys use provider defaults; None marks a level as unsupported.
ThinkingLevelMap = Dict[str, Optional[str]]

ThinkingTokenBudgetField = Literal["thinking_token_budget", "thinking_budget", "thinking_budget_tokens"]


class ThinkingBudgets(KarenBase):
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: Optional[int] = None
    low: Optional[int] = None
    medium: Optional[int] = None
    high: Optional[int] = None


CacheRetention = Literal["none", "short", "long"]
Transport = Literal["sse", "websocket", "websocket-cached", "auto"]

# Provider-scoped environment overrides. Values take precedence over process env.
ProviderEnv = Dict[str, str]
# Header value None suppresses a provider/API default header with the same name.
ProviderHeaders = Dict[str, Optional[str]]
SessionAffinityFormat = Literal["openai", "openai-nosession", "openrouter"]


class ProviderResponse(KarenBase):
    status: int
    headers: Dict[str, str]


class ProviderRequestOptions(KarenBase):
    """Authentication, HTTP transport, and lifecycle callbacks shared by provider requests."""

    signal: Optional["AbortSignal"] = None
    api_key: Optional[str] = None
    env: Optional[ProviderEnv] = None
    on_payload: Optional[Callable[[Any, Any], Union[None, dict, Awaitable[Union[None, dict]]]]] = None
    on_response: Optional[Callable[[ProviderResponse, Any], Union[None, Awaitable[None]]]] = None
    headers: Optional[ProviderHeaders] = None
    timeout_ms: Optional[int] = None
    max_retries: Optional[int] = None
    max_retry_delay_ms: Optional[int] = None

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class StreamOptions(ProviderRequestOptions):
    on_provider_stream_event: Optional[Callable[[Any, Any], Union[None, Awaitable[None]]]] = None
    temperature: Optional[float] = None
    sampling_params: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    transport: Optional[Transport] = None
    cache_retention: Optional[CacheRetention] = None
    session_id: Optional[str] = None
    websocket_connect_timeout_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SimpleStreamOptions(StreamOptions):
    """Unified options with reasoning passed to stream_simple() and complete_simple()."""

    tool_choice: Optional[Any] = None
    reasoning: Optional[ThinkingLevel] = None
    deferred: Optional[Union[bool, Dict[str, Literal["15m", "1h", "24h"]]]] = None
    thinking_budgets: Optional[ThinkingBudgets] = None


class DeferredFetchOptions(ProviderRequestOptions):
    wait: Optional[int] = None


DeferredCancelOptions = ProviderRequestOptions


class DeferredHandle(KarenBase):
    provider: str
    model_id: str
    api: str
    id: str
    expires_at: Optional[int] = None
    poll_after_ms: Optional[int] = None
    data: Optional[JsonValue] = None


# ---------------------------------------------------------------------------
# Content blocks & messages
# ---------------------------------------------------------------------------


class TextContent(KarenBase):
    type: Literal["text"] = "text"
    text: str
    text_signature: Optional[str] = None


class ThinkingContent(KarenBase):
    type: Literal["thinking"] = "thinking"
    thinking: str
    thinking_signature: Optional[str] = None
    redacted: Optional[bool] = None


class ImageContent(KarenBase):
    type: Literal["image"] = "image"
    data: str  # base64 encoded image data
    mime_type: str  # e.g. "image/jpeg", "image/png"


class ToolCall(KarenBase):
    type: Literal["toolCall"] = "toolCall"
    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[str] = None
    namespace: Optional[str] = None

    # Streaming scratch buffers (never serialized; stripped when the block finalizes).
    _partial_json: Optional[str] = PrivateAttr(default=None)
    _partial_args: Optional[str] = PrivateAttr(default=None)
    _stream_index: Optional[int] = PrivateAttr(default=None)

    @property
    def partial_json(self) -> Optional[str]:
        return self._partial_json

    @partial_json.setter
    def partial_json(self, value: Optional[str]) -> None:
        self._partial_json = value

    @property
    def partial_args(self) -> Optional[str]:
        return self._partial_args

    @partial_args.setter
    def partial_args(self, value: Optional[str]) -> None:
        self._partial_args = value

    @property
    def stream_index(self) -> Optional[int]:
        return self._stream_index

    @stream_index.setter
    def stream_index(self, value: Optional[int]) -> None:
        self._stream_index = value


ContentBlock = Annotated[
    Union[TextContent, ThinkingContent, ImageContent, ToolCall],
    Field(discriminator="type"),
]

InputContentBlock = Annotated[
    Union[TextContent, ImageContent],
    Field(discriminator="type"),
]


class UsageCost(KarenBase):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(KarenBase):
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write1h: Optional[int] = None
    reasoning: Optional[int] = None
    total_tokens: int = 0
    cost: UsageCost = Field(default_factory=UsageCost)


StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]


class SystemMessage(KarenBase):
    """System instructions and tool declarations at one point in the transcript.

    The leading system message is the system prompt. Later system messages change it:
    `content` adds instructions, `sections` replace/remove named prompt sections,
    and `tools_added`/`tools_removed` change the tool set.
    """

    role: Literal["system"] = "system"
    content: Union[str, List[TextContent]] = ""
    sections: Optional[Dict[str, Optional[str]]] = None
    tools_added: Optional[List["Tool"]] = None
    tools_removed: Optional[List["ToolReference"]] = None
    timestamp: int = 0  # Unix timestamp in milliseconds


class UserMessage(KarenBase):
    role: Literal["user"] = "user"
    content: Union[str, List[InputContentBlock]]
    timestamp: int


class DiagnosticErrorInfo(KarenBase):
    name: Optional[str] = None
    message: str
    stack: Optional[str] = None
    code: Optional[Union[str, int]] = None


class AssistantMessageDiagnostic(KarenBase):
    type: str
    timestamp: int
    error: Optional[DiagnosticErrorInfo] = None
    details: Optional[Dict[str, Any]] = None


class AssistantMessage(KarenBase):
    role: Literal["assistant"] = "assistant"
    content: List[ContentBlock] = Field(default_factory=list)
    api: Api
    provider: ProviderId
    model: str
    response_model: Optional[str] = None
    response_id: Optional[str] = None
    provider_thinking_level: Optional[str] = None
    diagnostics: Optional[List[AssistantMessageDiagnostic]] = None
    usage: Usage = Field(default_factory=Usage)
    stop_reason: StopReason = "pending"
    deferred: Optional[DeferredHandle] = None
    error_message: Optional[str] = None
    raw_stop_reason: Optional[str] = None
    end_turn: Optional[bool] = None
    timestamp: int


class ToolResultMessage(KarenBase):
    role: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    tool_name: str
    content: List[InputContentBlock] = Field(default_factory=list)
    details: Optional[JsonValue] = None
    usage: Optional[Usage] = None
    is_error: bool = False
    timestamp: int


Message = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolResultMessage],
    Field(discriminator="role"),
]


# ---------------------------------------------------------------------------
# Images / classifiers (types only; implementations arrive with their APIs)
# ---------------------------------------------------------------------------


class ImagesContext(KarenBase):
    input: List[InputContentBlock]


ImagesStopReason = Literal["stop", "error", "aborted"]


class AssistantImages(KarenBase):
    api: ImageApi
    provider: ProviderId
    model: str
    output: List[InputContentBlock] = Field(default_factory=list)
    response_id: Optional[str] = None
    usage: Optional[Usage] = None
    stop_reason: ImagesStopReason
    error_message: Optional[str] = None
    timestamp: int


class ClassifierChoiceQuestion(KarenBase):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Dict[str, str]


class ClassifierScoreQuestion(KarenBase):
    type: Literal["score"] = "score"
    instructions: str
    criteria: List[str]


class ClassifierBoolQuestion(KarenBase):
    type: Literal["bool"] = "bool"
    instructions: str
    criteria: Dict[Literal["true", "false"], str]


ClassifierQuestion = Annotated[
    Union[ClassifierChoiceQuestion, ClassifierScoreQuestion, ClassifierBoolQuestion],
    Field(discriminator="type"),
]


class ClassifierContext(KarenBase):
    state: JsonObject
    questions: Dict[str, ClassifierQuestion]


class ClassifierChoiceAnswer(KarenBase):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float


class ClassifierScoreAnswer(KarenBase):
    type: Literal["score"] = "score"
    score: float
    confidence: float


class ClassifierBoolAnswer(KarenBase):
    type: Literal["bool"] = "bool"
    probability: float


ClassifierAnswer = Annotated[
    Union[ClassifierChoiceAnswer, ClassifierScoreAnswer, ClassifierBoolAnswer],
    Field(discriminator="type"),
]
ClassifierStopReason = Literal["stop", "error", "aborted"]


class ClassifierResult(KarenBase):
    api: ClassifierApi
    provider: ProviderId
    model: str
    answers: Dict[str, ClassifierAnswer]
    stop_reason: ClassifierStopReason
    error_message: Optional[str] = None
    timestamp: int


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

GrammarFormat = Literal["openai_lark", "openai_regex"]
GrammarVariants = Dict[str, str]


class JsonSchemaSampling(KarenBase):
    type: Literal["json_schema"] = "json_schema"
    strict: Literal["prefer", "require"]


class GrammarSampling(KarenBase):
    type: Literal["grammar"] = "grammar"
    variants: GrammarVariants


ConstrainedSamplingConfig = Annotated[
    Union[JsonSchemaSampling, GrammarSampling],
    Field(discriminator="type"),
]


class Tool(KarenBase):
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    constrained_sampling: Optional[Union[Literal[False], ConstrainedSamplingConfig]] = None


class ToolReference(KarenBase):
    name: str


SystemMessage.model_rebuild()


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


class Context(KarenBase):
    """Request input accepted by the public stream entry points.

    `system_prompt` and `tools` are shorthand for a leading system message;
    `normalize_context()` folds them into one before the request reaches a provider.
    """

    system_prompt: Optional[str] = None
    messages: List[Message]
    tools: Optional[List[Tool]] = None


class TranscriptContext(KarenBase):
    """Normalized request context passed to providers and API implementations.

    Only `normalize_context()` produces this type, so a raw `Context` cannot
    reach provider code by accident (mirrors pi-ai's branded type).
    """

    messages: List[Message]
    normalized: ClassVar[bool] = True


# ---------------------------------------------------------------------------
# Streaming event protocol
# ---------------------------------------------------------------------------


class StartEvent(KarenBase):
    type: Literal["start"] = "start"
    partial: AssistantMessage


class TextStartEvent(KarenBase):
    type: Literal["text_start"] = "text_start"
    content_index: int
    partial: AssistantMessage


class TextDeltaEvent(KarenBase):
    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class TextEndEvent(KarenBase):
    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ThinkingStartEvent(KarenBase):
    type: Literal["thinking_start"] = "thinking_start"
    content_index: int
    partial: AssistantMessage


class ThinkingDeltaEvent(KarenBase):
    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ThinkingEndEvent(KarenBase):
    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    partial: AssistantMessage


class ToolCallStartEvent(KarenBase):
    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int
    partial: AssistantMessage


class ToolCallDeltaEvent(KarenBase):
    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str
    partial: AssistantMessage


class ToolCallEndEvent(KarenBase):
    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage


class DoneEvent(KarenBase):
    type: Literal["done"] = "done"
    reason: Literal["stop", "length", "toolUse", "deferred"]
    message: AssistantMessage


class ErrorEvent(KarenBase):
    type: Literal["error"] = "error"
    reason: Literal["aborted", "error"]
    error: AssistantMessage


AssistantMessageEvent = Annotated[
    Union[
        StartEvent,
        TextStartEvent,
        TextDeltaEvent,
        TextEndEvent,
        ThinkingStartEvent,
        ThinkingDeltaEvent,
        ThinkingEndEvent,
        ToolCallStartEvent,
        ToolCallDeltaEvent,
        ToolCallEndEvent,
        DoneEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Model catalog types
# ---------------------------------------------------------------------------


class OpenRouterRouting(KarenBase):
    """OpenRouter provider routing preferences, sent as the `provider` request field."""

    allow_fallbacks: Optional[bool] = None
    require_parameters: Optional[bool] = None
    data_collection: Optional[Literal["deny", "allow"]] = None
    zdr: Optional[bool] = None
    enforce_distillable_text: Optional[bool] = None
    order: Optional[List[str]] = None
    only: Optional[List[str]] = None
    ignore: Optional[List[str]] = None
    quantizations: Optional[List[str]] = None
    sort: Optional[Union[str, Dict[str, Any]]] = None
    max_price: Optional[Dict[str, Any]] = None
    preferred_min_throughput: Optional[Union[int, float, Dict[str, Any]]] = None
    preferred_max_latency: Optional[Union[int, float, Dict[str, Any]]] = None


class VercelGatewayRouting(KarenBase):
    only: Optional[List[str]] = None
    order: Optional[List[str]] = None


class ChatTemplateKwargVar(KarenBase):
    var: Literal["thinking.enabled", "thinking.effort", "thinking.budget"] = Field(alias="$var")
    omit_when_off: Optional[bool] = None


ChatTemplateKwargValue = Union[str, int, float, bool, None, ChatTemplateKwargVar]


class OpenAICompletionsCompat(KarenBase):
    """Compatibility settings for OpenAI-compatible completions APIs.

    Use this to override URL-based auto-detection for custom providers.
    """

    model_config = ConfigDict(extra="allow")

    supports_store: Optional[bool] = None
    supports_developer_role: Optional[bool] = None
    supports_reasoning_effort: Optional[bool] = None
    supports_usage_in_streaming: Optional[bool] = None
    supports_finish_reason: Optional[bool] = None
    max_tokens_field: Optional[Literal["max_completion_tokens", "max_tokens"]] = None
    requires_tool_result_name: Optional[bool] = None
    requires_assistant_after_tool_result: Optional[bool] = None
    requires_thinking_as_text: Optional[bool] = None
    requires_reasoning_content_on_assistant_messages: Optional[bool] = None
    thinking_format: Optional[
        Literal[
            "openai",
            "openrouter",
            "deepseek",
            "together",
            "baseten",
            "zai",
            "qwen",
            "chat-template",
            "qwen-chat-template",
            "string-thinking",
            "ant-ling",
        ]
    ] = None
    chat_template_kwargs: Optional[Dict[str, ChatTemplateKwargValue]] = None
    chat_template_args: Optional[Dict[str, ChatTemplateKwargValue]] = None
    open_router_routing: Optional[OpenRouterRouting] = None
    vercel_gateway_routing: Optional[VercelGatewayRouting] = None
    zai_tool_stream: Optional[bool] = None
    thinking_token_budget_field: Optional[ThinkingTokenBudgetField] = None
    supports_thinking_token_budget: Optional[bool] = None
    supports_openai_grammar_tools: Optional[bool] = None
    supports_mid_convo_system_messages: Optional[bool] = None
    supports_mid_convo_tool_additions: Optional[bool] = None
    supports_strict_mode: Optional[bool] = None
    cache_control_format: Optional[Literal["anthropic"]] = None
    send_session_affinity_headers: Optional[bool] = None
    session_affinity_format: Optional[SessionAffinityFormat] = None
    supports_long_cache_retention: Optional[bool] = None
    vllm_priority: Optional[int] = None


class OpenAIResponsesCompat(KarenBase):
    model_config = ConfigDict(extra="allow")

    supports_developer_role: Optional[bool] = None
    supports_mid_convo_system_messages: Optional[bool] = None
    session_affinity_format: Optional[SessionAffinityFormat] = None
    supports_long_cache_retention: Optional[bool] = None
    supports_strict_mode: Optional[bool] = None
    supports_openai_grammar_tools: Optional[bool] = None
    supports_additional_tools: Optional[bool] = None
    supports_tool_search: Optional[bool] = None
    supports_explicit_prompt_cache_mode: Optional[bool] = None
    supports_max_output_tokens: Optional[bool] = None


class AnthropicAllowedFallbackModel(KarenBase):
    provider: ProviderId
    model: str
    cost: "ModelCost"


class AnthropicMessagesCompat(KarenBase):
    model_config = ConfigDict(extra="allow")

    supports_eager_tool_input_streaming: Optional[bool] = None
    supports_long_cache_retention: Optional[bool] = None
    send_session_affinity_headers: Optional[bool] = None
    session_affinity_format: Optional[Literal["openrouter"]] = None
    supports_cache_control_on_tools: Optional[bool] = None
    supports_temperature: Optional[bool] = None
    force_adaptive_thinking: Optional[bool] = None
    allow_empty_signature: Optional[bool] = None
    supports_strict_tools: Optional[bool] = None
    supports_mid_convo_effort: Optional[bool] = None
    supports_mid_convo_system_messages: Optional[bool] = None
    supports_mid_convo_tool_changes: Optional[bool] = None
    allowed_fallback_models: Optional[List[AnthropicAllowedFallbackModel]] = None


class BedrockCompat(KarenBase):
    model_config = ConfigDict(extra="allow")

    supports_strict_mode: Optional[bool] = None


class MistralConversationsCompat(KarenBase):
    model_config = ConfigDict(extra="allow")

    supports_mid_convo_system_messages: Optional[bool] = None


class ModelCostRates(KarenBase):
    input: float  # $/million tokens
    output: float  # $/million tokens
    cache_read: float  # $/million tokens
    cache_write: float  # $/million tokens


class ModelCostTier(ModelCostRates):
    input_tokens_above: int


class ModelCost(ModelCostRates):
    tiers: Optional[List[ModelCostTier]] = None


class ModelImageResizeOptions(KarenBase):
    max_width: Optional[int] = None
    max_height: Optional[int] = None
    max_bytes: Optional[int] = None
    jpeg_quality: Optional[int] = None


class ModelImageInputLimits(KarenBase):
    resize: Optional[ModelImageResizeOptions] = None
    max_per_message: Optional[int] = None
    max_per_request: Optional[int] = None


class ModelInputLimits(KarenBase):
    max_request_bytes: Optional[int] = None
    images: Optional[ModelImageInputLimits] = None


class BaseModel_(KarenBase):
    """Fields shared by every catalog entry, regardless of what you can do with it."""

    id: str
    name: str
    api: str
    provider: ProviderId
    base_url: str
    input: List[Literal["text", "image"]] = Field(default_factory=lambda: ["text"])
    input_limits: Optional[ModelInputLimits] = None
    cost: ModelCost
    headers: Optional[Dict[str, str]] = None


class Model(BaseModel_):
    """Chat model: usable with `stream()` and friends."""

    type: Optional[Literal["chat"]] = "chat"
    reasoning: bool = False
    thinking_level_map: Optional[ThinkingLevelMap] = None
    prompt_cache: Optional[Dict[Literal["short", "long"], int]] = None
    context_window: int = 0
    max_tokens: int = 0
    sampling_params: Optional[Dict[str, Any]] = None
    compat: Optional[Union[OpenAICompletionsCompat, OpenAIResponsesCompat, AnthropicMessagesCompat, BedrockCompat, MistralConversationsCompat]] = None

    @field_validator("compat", mode="before")
    @classmethod
    def _coerce_compat(cls, v: Any, info: Any) -> Any:
        # Compat shape is api-specific; when building from JSON we cannot know
        # the api reliably here, so accept dicts as OpenAICompletionsCompat only
        # via explicit construction elsewhere. Leave values untouched.
        return v


class ImageModel(BaseModel_):
    """Image-generation model: usable with `generate_images()` only."""

    type: Literal["image"] = "image"
    output: List[Literal["text", "image"]] = Field(default_factory=lambda: ["image"])


class ClassifierModel(BaseModel_):
    """Structured classifier model: usable with `classify()` only."""

    type: Literal["classifier"] = "classifier"
    context_window: int = 0


AnyModel = Union[Model, ImageModel, ClassifierModel]
ModelType = Literal["chat", "image", "classifier"]


def get_model_type(model: AnyModel) -> ModelType:
    t = getattr(model, "type", None)
    if t == "image":
        return "image"
    if t == "classifier":
        return "classifier"
    return "chat"


def is_model_type(model: AnyModel, model_type: ModelType) -> bool:
    return get_model_type(model) == model_type


# ---------------------------------------------------------------------------
# Stream function contracts (structural, see api/* and models.py)
# ---------------------------------------------------------------------------

StreamFunction = Callable[[Model, TranscriptContext, Optional[StreamOptions]], "AssistantMessageEventStream"]
SimpleStreamFunction = Callable[
    [Model, TranscriptContext, Optional[SimpleStreamOptions]], "AssistantMessageEventStream"
]
