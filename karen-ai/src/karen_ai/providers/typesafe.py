"""TypeSafe provider (classifier-only), mirroring typesafe.ts."""

from __future__ import annotations

from ..api import typesafe_system_one
from ..auth.helpers import env_api_key_auth
from ..auth.types import ProviderAuth
from ..model_catalog import flatten_classifier_model_catalog
from ..models import CreateProviderOptions, create_provider


def typesafe_provider():
    return create_provider(
        CreateProviderOptions(
            id="typesafe",
            name="TypeSafe",
            auth=ProviderAuth(api_key=env_api_key_auth("TypeSafe API key", ["TYPESAFE_API_KEY"])),
            models=list(flatten_classifier_model_catalog("typesafe").values()),
            classifiers={"typesafe-system-one": typesafe_system_one.classify},
        )
    )
