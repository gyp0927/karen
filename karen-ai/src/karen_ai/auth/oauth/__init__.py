"""OAuth login flows, mirroring pi-ai's auth/oauth/ package.

pi-ai loads these modules through bundler-opaque dynamic imports so browser
bundles never see Node-only code (callback servers, crypto). Python has no
bundler constraint, so this package imports its flows directly.

Flows:
- anthropic: PKCE + localhost callback (port 53692) with manual code fallback
- openrouter: PKCE + ephemeral callback, code exchanges for a permanent API key
- openai_codex: PKCE + localhost callback (port 1455), or device-code login
- github_copilot: device code against GitHub, then Copilot token + model policy
- kimi_coding: RFC 8628 device code against auth.kimi.com
- xai: RFC 8628 device code against auth.x.ai
- meta: RFC 8628 device code + Muse Code key mint
- radius: gateway OAuth (browser PKCE or device code) against a pi-messages gateway
"""

from .anthropic import anthropic_oauth
from .device_code import (
    DeviceCodePollResult,
    poll_complete,
    poll_failed,
    poll_oauth_device_code_flow,
    poll_pending,
    poll_slow_down,
)
from .github_copilot import github_copilot_oauth
from .kimi_coding import kimi_coding_oauth
from .meta import meta_oauth
from .oauth_page import oauth_error_html, oauth_success_html
from .openai_codex import openai_codex_oauth
from .openrouter import openrouter_oauth
from .pkce import generate_pkce
from .radius import create_radius_oauth, normalize_radius_gateway_url
from .xai import xai_oauth

__all__ = [
    "DeviceCodePollResult",
    "anthropic_oauth",
    "create_radius_oauth",
    "generate_pkce",
    "github_copilot_oauth",
    "kimi_coding_oauth",
    "meta_oauth",
    "normalize_radius_gateway_url",
    "oauth_error_html",
    "oauth_success_html",
    "openai_codex_oauth",
    "openrouter_oauth",
    "poll_complete",
    "poll_failed",
    "poll_oauth_device_code_flow",
    "poll_pending",
    "poll_slow_down",
    "xai_oauth",
]
