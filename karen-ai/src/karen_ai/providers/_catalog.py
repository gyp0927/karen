"""Shared factory for catalog-backed providers, mirroring pi-ai's thin provider modules.

Most pi-ai providers are identical in shape: env-key auth, models from the
vendored catalog, one API implementation. This factory carries that shape so
each provider module only states its constants.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

from ..auth.helpers import env_api_key_auth
from ..auth.types import OAuthAuth, ProviderAuth
from ..lazy import ProviderStreams
from ..model_catalog import flatten_chat_model_catalog
from ..models import CreateProviderOptions, create_provider


def catalog_provider(
    *,
    id: str,
    name: str,
    key_name: str,
    env_vars: Sequence[str],
    api: Union[ProviderStreams, Dict[str, ProviderStreams]],
    base_url: Optional[str] = None,
    oauth: Optional[OAuthAuth] = None,
):
    """Build a provider whose models come from `providers/data/<id>.json`."""
    return create_provider(
        CreateProviderOptions(
            id=id,
            name=name,
            base_url=base_url,
            auth=ProviderAuth(api_key=env_api_key_auth(key_name, env_vars), oauth=oauth),
            models=list(flatten_chat_model_catalog(id).values()),
            api=api,
        )
    )
