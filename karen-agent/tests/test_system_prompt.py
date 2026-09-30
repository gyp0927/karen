"""format_skills_for_system_prompt (pi's system-prompt.ts)."""

from karen_agent.resources import Skill
from karen_agent.system_prompt import escape_xml, format_skills_for_system_prompt


def test_empty_and_all_hidden():
    assert format_skills_for_system_prompt([]) == ""
    hidden = Skill(name="h", description="d", content="c", file_path="/s/SKILL.md", disable_model_invocation=True)
    assert format_skills_for_system_prompt([hidden]) == ""


def test_renders_available_skills_block():
    skills = [
        Skill(name="review", description="Review code", content="c", file_path="/skills/review/SKILL.md"),
        Skill(name="hidden", description="x", content="c", file_path="/h/SKILL.md", disable_model_invocation=True),
    ]
    text = format_skills_for_system_prompt(skills)
    assert text == (
        "The following skills provide specialized instructions for specific tasks.\n"
        "Read the full skill file when the task matches its description.\n"
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.\n"
        "\n"
        "<available_skills>\n"
        "  <skill>\n"
        "    <name>review</name>\n"
        "    <description>Review code</description>\n"
        "    <location>/skills/review/SKILL.md</location>\n"
        "  </skill>\n"
        "</available_skills>"
    )


def test_escape_xml():
    assert escape_xml("""a & <b> "c" 'd'""") == "a &amp; &lt;b&gt; &quot;c&quot; &apos;d&apos;"
    skill = Skill(name="a<b", description="x & y", content="c", file_path="/p")
    text = format_skills_for_system_prompt([skill])
    assert "<name>a&lt;b</name>" in text
    assert "<description>x &amp; y</description>" in text
