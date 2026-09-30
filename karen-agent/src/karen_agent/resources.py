"""Shared harness resource types (pi's `harness/types.ts` Skill/PromptTemplate/Resources)."""

from __future__ import annotations

from typing import List, Optional

from karen_ai.types import KarenBase

__all__ = ["Skill", "PromptTemplate", "Resources"]


class Skill(KarenBase):
    """A skill: named instructions the model can pull in on demand."""

    #: Stable skill name used for lookup and model-visible listings.
    name: str
    #: Short model-visible description of when to use the skill.
    description: str
    #: Full skill instructions.
    content: str
    #: Absolute path to the skill file. Used for model-visible location and resolving relative references.
    file_path: str
    #: Exclude this skill from model-visible skill lists while still allowing explicit application invocation.
    disable_model_invocation: Optional[bool] = None


class PromptTemplate(KarenBase):
    """A user-invocable prompt template."""

    #: Stable template name used for lookup or application command routing.
    name: str
    #: Optional description for command lists or autocomplete.
    description: Optional[str] = None
    #: Template content. Argument placeholders are formatted by ``format_prompt_template_invocation``.
    content: str


class Resources(KarenBase):
    """Prompt templates and skills available to a run."""

    #: Prompt templates available for explicit invocation.
    prompt_templates: Optional[List[PromptTemplate]] = None
    #: Skills available to the model and explicit skill invocation.
    skills: Optional[List[Skill]] = None
