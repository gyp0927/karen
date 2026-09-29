"""Google Vertex AI provider with a static flagship catalog.

Vertex accepts an explicit API key or Application Default Credentials
(`gcloud auth application-default login`); ADC additionally requires project
and location env vars, which the adapter reads itself
(GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT, GOOGLE_CLOUD_LOCATION).
"""

from __future__ import annotations

from typing import Optional

from ..api import google_vertex_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, ApiKeyResolveInput, AuthResult, ModelAuth, ProviderAuth
from ..errors import AbortError
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider
from ..types import Model

GOOGLE_VERTEX_API_KEY_ENV = "VERTEX_API_KEY"
GOOGLE_VERTEX_BASE_URL = "https://{location}-aiplatform.googleapis.com"
# Marker stored in place of an API key when the user chooses ADC auth; the
# adapter falls back to Application Default Credentials when it sees it.
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"

def _vertex_api_key_auth() -> ApiKeyAuth:
    async def login(interaction):
        from ..auth.types import AuthPromptSecret

        if getattr(interaction, "signal", None) and interaction.signal.aborted:
            raise AbortError()
        key = await interaction.prompt(
            AuthPromptSecret(message="Enter Google Cloud API key (or 'gcp-vertex-credentials' to use ADC)")
        )
        return ApiKeyCredential(key=key)

    async def resolve(input: ApiKeyResolveInput) -> Optional[AuthResult]:
        input.signal.throw_if_aborted()
        if input.credential and input.credential.key:
            return AuthResult(
                auth=ModelAuth(api_key=input.credential.key), env=input.credential.env, source="stored credential"
            )
        api_key = await input.ctx.env(GOOGLE_VERTEX_API_KEY_ENV)
        input.signal.throw_if_aborted()
        if api_key:
            return AuthResult(auth=ModelAuth(api_key=api_key), source=GOOGLE_VERTEX_API_KEY_ENV)
        # ADC needs no API key at all; the adapter mints OAuth tokens itself.
        return AuthResult(auth=ModelAuth(api_key=GCP_VERTEX_CREDENTIALS_MARKER), source="application default credentials")

    return ApiKeyAuth(name="Google Cloud credentials", login=login, resolve=resolve)

def google_vertex_provider():
    return create_provider(
        CreateProviderOptions(
            id="google-vertex",
            name="Google Vertex AI",
            base_url=GOOGLE_VERTEX_BASE_URL,
            auth=ProviderAuth(api_key=_vertex_api_key_auth()),
            models=list(flatten_chat_model_catalog("google-vertex").values()),
            api=google_vertex_api(),
        )
    )
