"""Amazon Bedrock provider with a static flagship catalog.

Bedrock accepts a bearer token or the AWS credential chain; the login flow can
store a token/profile choice, and resolve also detects ambient AWS credentials
without copying them into the credential store.
"""

from __future__ import annotations

from typing import Optional

from ..api import bedrock_converse_stream_api
from ..auth.types import (
    ApiKeyAuth,
    ApiKeyCredential,
    ApiKeyResolveInput,
    AuthEventInfo,
    AuthInfoLink,
    AuthPromptSecret,
    AuthPromptSelect,
    AuthPromptText,
    AuthResult,
    AuthSelectOption,
    ModelAuth,
    ProviderAuth,
)
from ..errors import AbortError
from ..models import CreateProviderOptions, create_provider
from ..types import Model, ModelCost

BEDROCK_DEFAULT_BASE_URL = "https://bedrock-runtime.us-east-1.amazonaws.com"


def _model(
    id: str,
    name: str,
    *,
    input_cost: float,
    output_cost: float,
    cache_read: float = 0.0,
    cache_write: float = 0.0,
    context_window: int = 200_000,
    max_tokens: int = 64_000,
    reasoning: bool = False,
) -> Model:
    return Model(
        id=id,
        name=name,
        api="bedrock-converse-stream",
        provider="amazon-bedrock",
        base_url=BEDROCK_DEFAULT_BASE_URL,
        input=["text", "image"],
        cost=ModelCost(input=input_cost, output=output_cost, cache_read=cache_read, cache_write=cache_write),
        reasoning=reasoning,
        context_window=context_window,
        max_tokens=max_tokens,
    )


AMAZON_BEDROCK_MODELS = [
    _model(
        "anthropic.claude-sonnet-4-5",
        "Claude Sonnet 4.5 (Bedrock)",
        input_cost=3.0,
        output_cost=15.0,
        cache_read=0.3,
        cache_write=3.75,
        reasoning=True,
    ),
    _model(
        "anthropic.claude-opus-4-5",
        "Claude Opus 4.5 (Bedrock)",
        input_cost=15.0,
        output_cost=75.0,
        cache_read=1.5,
        cache_write=18.75,
        reasoning=True,
    ),
    _model(
        "anthropic.claude-haiku-4-5",
        "Claude Haiku 4.5 (Bedrock)",
        input_cost=1.0,
        output_cost=5.0,
        cache_read=0.1,
        cache_write=1.25,
        reasoning=True,
    ),
    _model(
        "anthropic.claude-3-7-sonnet",
        "Claude 3.7 Sonnet (Bedrock)",
        input_cost=3.0,
        output_cost=15.0,
        cache_read=0.3,
        cache_write=3.75,
        reasoning=True,
    ),
    _model(
        "anthropic.claude-3-5-haiku",
        "Claude 3.5 Haiku (Bedrock)",
        input_cost=0.8,
        output_cost=4.0,
        cache_read=0.08,
        cache_write=1.0,
    ),
    _model(
        "amazon.nova-pro-v1:0",
        "Amazon Nova Pro",
        input_cost=0.8,
        output_cost=3.2,
        context_window=300_000,
        max_tokens=10_000,
    ),
]


def _bedrock_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        signal = getattr(interaction, "signal", None)
        if signal is not None and signal.aborted:
            raise AbortError()
        method = await interaction.prompt(
            AuthPromptSelect(
                message="Select Amazon Bedrock authentication method:",
                options=[
                    AuthSelectOption(id="bearer-token", label="Bearer token"),
                    AuthSelectOption(id="aws-profile", label="AWS profile"),
                    AuthSelectOption(id="credential-chain", label="Existing AWS credential chain"),
                ],
            )
        )
        if signal is not None and signal.aborted:
            raise AbortError()
        if method == "bearer-token":
            key = await interaction.prompt(AuthPromptSecret(message="Enter Amazon Bedrock bearer token"))
            return ApiKeyCredential(key=key)
        interaction.notify(
            AuthEventInfo(
                message="Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials.",
                links=[
                    AuthInfoLink(
                        label="AWS credential provider chain",
                        url="https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html",
                    )
                ],
            )
        )
        if method == "aws-profile":
            profile = await interaction.prompt(AuthPromptText(message="Enter AWS profile name"))
            return ApiKeyCredential(env={"AWS_PROFILE": profile})
        if method != "credential-chain":
            raise ValueError(f"Unknown Amazon Bedrock auth method: {method}")
        await interaction.prompt(AuthPromptText(message="Configure AWS credentials, then press Enter to continue"))
        return ApiKeyCredential()

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        async def env(name: str) -> Optional[str]:
            input.signal.throw_if_aborted()
            value = await input.ctx.env(name)
            input.signal.throw_if_aborted()
            return value

        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        if await env("AWS_BEARER_TOKEN_BEDROCK"):
            return AuthResult(auth=ModelAuth(), source="AWS_BEARER_TOKEN_BEDROCK")
        credential_profile = (input.credential.env or {}).get("AWS_PROFILE") if input.credential else None
        if credential_profile or await env("AWS_PROFILE"):
            return AuthResult(
                auth=ModelAuth(),
                env=input.credential.env if input.credential else None,
                source="stored credential" if credential_profile else "AWS_PROFILE",
            )
        if (await env("AWS_ACCESS_KEY_ID")) and (await env("AWS_SECRET_ACCESS_KEY")):
            return AuthResult(auth=ModelAuth(), source="AWS access keys")
        if await env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
            return AuthResult(auth=ModelAuth(), source="ECS task role")
        if await env("AWS_CONTAINER_CREDENTIALS_FULL_URI"):
            return AuthResult(auth=ModelAuth(), source="ECS task role")
        if await env("AWS_WEB_IDENTITY_TOKEN_FILE"):
            return AuthResult(auth=ModelAuth(), source="web identity token")
        return None

    return ApiKeyAuth(name="AWS credentials or bearer token", login=login, resolve=resolve)


def amazon_bedrock_provider():
    return create_provider(
        CreateProviderOptions(
            id="amazon-bedrock",
            name="Amazon Bedrock",
            base_url=BEDROCK_DEFAULT_BASE_URL,
            auth=ProviderAuth(api_key=_bedrock_api_key_auth()),
            models=list(AMAZON_BEDROCK_MODELS),
            api=bedrock_converse_stream_api(),
        )
    )
