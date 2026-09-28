"""Images API registry, mirroring images-api-registry.ts.

Global registry dispatching image generation on `model.api`. Auth must be
passed explicitly via `options.api_key`; prefer `Models.generate_images()`,
which resolves provider auth.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional

from .types import AssistantImages, ImageModel, ImagesContext, ProviderRequestOptions

ImagesFunction = Callable[[ImageModel, ImagesContext, Optional[ProviderRequestOptions]], Awaitable[AssistantImages]]

_REGISTRY: Dict[str, ImagesFunction] = {}


def register_images_api_provider(api: str, generate_images: ImagesFunction) -> None:
    def guarded(model: ImageModel, context: ImagesContext, options: Optional[ProviderRequestOptions] = None):
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return generate_images(model, context, options)

    _REGISTRY[api] = guarded


def get_images_api_provider(api: str) -> Optional[ImagesFunction]:
    return _REGISTRY.get(api)


async def generate_images(
    model: ImageModel,
    context: ImagesContext,
    options: Optional[ProviderRequestOptions] = None,
) -> AssistantImages:
    """Global image generation dispatched on `model.api` through the registry."""
    provider = get_images_api_provider(model.api)
    if provider is None:
        raise ValueError(f"No API provider registered for api: {model.api}")
    return await provider(model, context, options)


def _register_builtins() -> None:
    from .api.openrouter_images import generate_images as openrouter_generate_images

    register_images_api_provider("openrouter-images", openrouter_generate_images)


_register_builtins()
