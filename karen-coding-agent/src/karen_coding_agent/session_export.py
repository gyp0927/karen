"""Session export (pi's `core/session-export.ts` + `core/export-html/`).

Two exporters:

- **JSONL** (`serialize_session_branch` / `export_session_to_jsonl`) writes the
  current branch in pi's importable v3 wire format — a `{"type":"session",
  "version":3,...}` header followed by the branch's entries re-parented into a
  single linear chain — so a karen session can be re-opened by pi.
- **HTML** (`export_session_to_html` / `export_html_from_file`) writes one
  self-contained `.html` report: the session data is base64-embedded and
  rendered client-side by the template's built-in markdown subset renderer —
  no CDN scripts, so the report opens offline and with no network requests
  (pi inlines vendored marked/highlight.js instead). Only the `leafId` branch
  path is rendered, like pi's `getPath`. pi sources the theme from its TUI
  theme system; karen ships a neutral dark default until the TUI/theme
  milestone lands.
"""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

# Tools rendered directly by the HTML template (pi's `TEMPLATE_RENDERED_TOOLS`).
# karen's built-in tool set; extension/custom tools would be pre-rendered by a
# tool renderer, which arrives with the extension milestone.
_TEMPLATE_RENDERED_TOOLS = frozenset(["bash", "powershell", "read", "write", "edit", "ls", "grep", "find"])

#: pi's importable session line format version.
PI_SESSION_VERSION = 3

# -- neutral default export palette (pi derives this from the TUI theme) --------
_PAGE_BG = "#1a1b1e"
_CARD_BG = "#232428"
_INFO_BG = "#2e2b20"
_TEXT = "#e6e6e9"
_MUTED = "#9a9aa2"
_USER_BG = "#343541"
_ASSISTANT_BG = "#26272b"
_ACCENT = "#7aa2f7"
_TOOL_BG = "#1f2124"
_TOOL_BORDER = "#3a3b40"
_ERROR = "#f7768e"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonl_header(session_id: str, cwd: str, timestamp: str) -> Dict[str, Any]:
    """pi's v3 export header."""
    return {"type": "session", "version": PI_SESSION_VERSION, "id": session_id, "timestamp": timestamp, "cwd": cwd}


# ---------------------------------------------------------------------------
# JSONL export
# ---------------------------------------------------------------------------


def serialize_session_branch(
    session_id: str,
    cwd: str,
    branch_entries: Sequence[Any],
    create_trailing_entries: Optional[Callable[[Optional[str], str], Sequence[Any]]] = None,
) -> str:
    """Serialize the current branch (and optional trailing entries) as pi-importable JSONL.

    `branch_entries` are the branch's entries oldest-first as JSON-able dicts
    (karen's `to_jsonable(entry)`). Each is re-parented to the previous one so
    the export is one linear chain, exactly like pi's `serializeSessionBranch`.
    """
    timestamp = _iso_now()
    entries: List[Any] = [_jsonl_header(session_id, cwd, timestamp)]
    parent_id: Optional[str] = None
    for entry in branch_entries:
        data = dict(entry)
        data["parentId"] = parent_id
        entries.append(data)
        parent_id = data.get("id")
    if create_trailing_entries is not None:
        entries.extend(create_trailing_entries(parent_id, timestamp))
    return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n"


def export_session_to_jsonl(
    session_id: str,
    cwd: str,
    branch_entries: Sequence[Any],
    output_path: Optional[str] = None,
    create_trailing_entries: Optional[Callable[[Optional[str], str], Sequence[Any]]] = None,
) -> str:
    """Write the branch as pi-importable JSONL; returns the file path."""
    if output_path is None:
        stamp = _iso_now().replace(":", "-").replace(".", "-")
        output_path = os.path.join(cwd, f"session-{stamp}.jsonl")
    output_path = os.path.abspath(os.path.expanduser(output_path))
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(serialize_session_branch(session_id, cwd, branch_entries, create_trailing_entries))
    return output_path


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------


def _generate_html(session_data: Dict[str, Any]) -> str:
    payload = base64.b64encode(json.dumps(session_data, ensure_ascii=False).encode("utf-8")).decode("utf-8")
    return _TEMPLATE.replace("__PAGE_BG__", _PAGE_BG).replace("__CARD_BG__", _CARD_BG).replace(
        "__INFO_BG__", _INFO_BG
    ).replace("__TEXT__", _TEXT).replace("__MUTED__", _MUTED).replace("__USER_BG__", _USER_BG).replace(
        "__ASSISTANT_BG__", _ASSISTANT_BG
    ).replace("__ACCENT__", _ACCENT).replace("__TOOL_BG__", _TOOL_BG).replace(
        "__TOOL_BORDER__", _TOOL_BORDER
    ).replace("__ERROR__", _ERROR).replace("__SESSION_DATA__", payload)


def _branch_path_entries(entries: Sequence[Any], leaf_id: Optional[str]) -> List[Any]:
    """Root -> leaf path through `parentId` links (pi's `getPath`).

    pi embeds every entry and walks the path in the browser (its report has a
    branch-tree sidebar); karen has no sidebar yet, so the path is resolved at
    export time. Entries on abandoned branches stay in the session file but
    never reach the report. An unknown leaf falls back to every entry, so a
    foreign or hand-edited file still exports something.
    """
    by_id = {entry.get("id"): entry for entry in entries if isinstance(entry, dict) and entry.get("id")}
    current = by_id.get(leaf_id) if leaf_id else None
    if current is None:
        return list(entries)
    path: List[Any] = []
    seen = set()
    while current is not None and current.get("id") not in seen:
        seen.add(current.get("id"))
        path.append(current)
        parent_id = current.get("parentId")
        if not parent_id or parent_id == current.get("id"):
            break
        current = by_id.get(parent_id)
    path.reverse()
    return path or list(entries)


def export_session_to_html(
    header: Dict[str, Any],
    entries: Sequence[Any],
    leaf_id: Optional[str],
    output_path: Optional[str] = None,
    cwd: Optional[str] = None,
    system_prompt: Optional[str] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Write the session as a self-contained HTML report; returns the path.

    `header`/`entries` are karen's native JSON-able shapes (the JSONL session
    file's own format); the report renders the `leaf_id` branch path.
    """
    session_data = {
        "header": header,
        "entries": _branch_path_entries(entries, leaf_id),
        "leafId": leaf_id,
        "systemPrompt": system_prompt,
        "tools": list(tools) if tools else None,
    }
    html = _generate_html(session_data)

    if output_path is None:
        base_dir = cwd or os.getcwd()
        basename = str(header.get("id") or f"session-{int(time.time())}")
        output_path = os.path.join(base_dir, f"karen-session-{basename}.html")
    output_path = os.path.abspath(os.path.expanduser(output_path))
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html)
    return output_path


def read_session_records(input_path: str) -> Dict[str, Any]:
    """Read a karen session file into `{header, entries, leaf_id}`.

    Handles both on-disk shapes:

    - **karen's native storage log** (what `~/.karen/sessions/*.jsonl` holds):
      a `{"kind":"header"}` line, `{"kind":"value"}` writes — the branch tip
      lives in the `pi.branch.tip` namespace — and `{"kind":"entry"}` records
      batched one JSON array per line.
    - **pi's importable v3 format** (what `export_session_to_jsonl` writes, and
      what pi's own exports look like): a `{"type":"session"}` header followed
      by entry objects carrying `parentId`/`isLeaf`.

    Entries come back in `seq` order (native) or file order, which is what the
    branch walk needs.
    """
    header: Dict[str, Any] = {}
    entries: List[Any] = []
    leaf_id: Optional[str] = None
    tips: Dict[str, Any] = {}
    with open(input_path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for item in record if isinstance(record, list) else [record]:
                if not isinstance(item, dict):
                    continue
                kind = item.get("kind")
                if kind == "header":
                    header = item
                    continue
                if kind == "value":
                    if item.get("namespace") == "pi.branch.tip":
                        tips[item.get("key")] = item.get("value")
                    continue
                if kind == "entry":
                    entries.append(item)
                    continue
                # pi's export format: a typed header, then bare entry objects.
                rtype = item.get("type")
                if rtype == "session":
                    header = item
                elif rtype == "leaf" or rtype == "branchTip":
                    leaf_id = item.get("id") or item.get("leafId") or leaf_id
                else:
                    entries.append(item)
                    if item.get("isLeaf"):
                        leaf_id = item.get("id")

    entries.sort(key=lambda entry: entry.get("seq") or 0)
    if leaf_id is None:
        # The last tip write for any lane wins; a session file has one lane.
        leaf_id = tips.get("main") or (list(tips.values())[-1] if tips else None)
    if leaf_id is None and entries:
        leaf_id = entries[-1].get("id")
    return {"header": header, "entries": entries, "leaf_id": leaf_id}


def export_html_from_file(input_path: str, output_path: Optional[str] = None) -> str:
    """Export an existing karen JSONL session file to HTML (standalone).

    Reads either the native storage log or a v3 export (see
    `read_session_records`) and renders the `leaf_id` branch.
    """
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    session = read_session_records(input_path)
    if output_path is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"karen-session-{base}.html")

    return export_session_to_html(
        session["header"],
        session["entries"],
        session["leaf_id"],
        output_path=output_path,
        cwd=os.path.dirname(input_path),
    )


# Self-contained HTML template. The session data is injected as base64 JSON and
# rendered client-side by the small built-in renderer below — no CDN scripts,
# so the file opens offline and makes no network requests. pi vendors marked +
# highlight.js into the document (`export-html/vendor/`); karen ships a
# markdown subset and no syntax highlighting instead of a 165KB JS blob, and
# renders the `leafId` branch path exactly like pi's `getPath`.
_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>karen session</title>
<style>
:root{
  --pageBg:__PAGE_BG__; --cardBg:__CARD_BG__; --infoBg:__INFO_BG__;
  --text:__TEXT__; --muted:__MUTED__; --userBg:__USER_BG__; --assistantBg:__ASSISTANT_BG__;
  --accent:__ACCENT__; --toolBg:__TOOL_BG__; --toolBorder:__TOOL_BORDER__; --error:__ERROR__;
}
*{box-sizing:border-box}
body{margin:0;background:var(--pageBg);color:var(--text);font:14px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.container{max-width:860px;margin:0 auto;padding:24px 16px 80px}
header.top{margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid var(--toolBorder)}
header.top h1{font-size:18px;margin:0 0 6px}
header.top .meta{color:var(--muted);font-size:12px}
.msg{background:var(--cardBg);border:1px solid var(--toolBorder);border-radius:10px;padding:12px 14px;margin:12px 0}
.msg .role{display:inline-block;font-size:11px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-bottom:8px;padding:2px 8px;border-radius:6px;background:var(--toolBg)}
.msg.user .role{background:var(--userBg);color:var(--text)}
.msg.assistant .role{background:var(--assistantBg);color:var(--accent)}
.msg pre{white-space:pre-wrap;word-break:break-word;background:var(--toolBg);border:1px solid var(--toolBorder);border-radius:8px;padding:10px;overflow-x:auto}
.msg code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px}
.msg p{margin:8px 0}
.msg img{max-width:100%;border-radius:8px;margin:8px 0}
.msg ul,.msg ol{padding-left:22px}
.msg h1,.msg h2,.msg h3,.msg h4,.msg h5,.msg h6{margin:14px 0 6px;line-height:1.3}
.msg h1{font-size:19px}.msg h2{font-size:17px}.msg h3{font-size:15px}
.msg h4,.msg h5,.msg h6{font-size:14px;color:var(--muted)}
.msg blockquote{margin:8px 0;padding:4px 12px;border-left:3px solid var(--toolBorder);color:var(--muted)}
.msg a{color:var(--accent)}
.msg hr{border:0;border-top:1px solid var(--toolBorder);margin:14px 0}
details.tool{background:var(--toolBg);border:1px solid var(--toolBorder);border-radius:8px;margin:8px 0}
details.tool>summary{cursor:pointer;padding:8px 12px;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:var(--accent);list-style:none}
details.tool>summary::-webkit-details-marker{display:none}
details.tool>summary::before{content:"\\25B8";margin-right:8px;color:var(--muted)}
details.tool[open]>summary::before{content:"\\25BE"}
details.tool .body{padding:0 12px 12px}
details.tool .lbl{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin:8px 0 4px}
details.tool pre{margin:0}
.tool-error{color:var(--error)}
.msg.bash-msg{background:var(--toolBg)}
.bash .cmd{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:var(--accent);word-break:break-word}
.bash pre{margin:8px 0 0}
.bash .note{color:var(--muted);font-size:12px;margin-top:6px}
.bash.error{border-left:3px solid var(--error)}
.bash.error .note{color:var(--error)}
.sys{background:var(--infoBg);border-radius:8px;padding:10px 12px;color:var(--muted);font-size:12.5px;margin:12px 0}
.entry-kind{color:var(--muted);font-size:11px}
</style>
</head>
<body>
<div class="container" id="app">Rendering&hellip;</div>
<script>
// atob yields Latin-1 code units, so the UTF-8 bytes it produced have to be
// decoded back before parsing (pi's template.js does the same) — otherwise
// every non-ASCII string in the transcript renders as mojibake.
const DATA = JSON.parse(new TextDecoder("utf-8").decode(Uint8Array.from(atob("__SESSION_DATA__"),(c)=>c.charCodeAt(0))));
const esc = (s)=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
// Inline markdown subset: fenced code, inline code, headings, lists, quotes,
// rules, bold/italic, links. Input is escaped first, then transformed, so
// session text can never inject markup.
// Fenced code is extracted line-wise, the way CommonMark reads it: a fence
// opens only at the start of a line (up to three spaces of indent), closes on a
// fence-only line, and an unterminated fence runs to the end of the text. Every
// extraction leaves a whole-line placeholder, so the block is always emitted and
// no sentinel can survive into a paragraph.
function extractFences(text,blocks){
  const lines=String(text??"").split("\\n");
  const out=[];
  let i=0;
  while(i<lines.length){
    const open=lines[i].match(/^ {0,3}```([^`]*)$/);
    if(!open){ out.push(lines[i]); i++; continue; }
    const lang=(open[1].trim().split(/\\s+/)[0]||"");
    const body=[];
    i++;
    while(i<lines.length&&!/^ {0,3}```\\s*$/.test(lines[i])){ body.push(lines[i]); i++; }
    if(i<lines.length) i++; // consume the closing fence
    blocks.push('<pre><code'+(lang?' class="lang-'+esc(lang)+'"':'')+'>'+esc(body.join("\\n"))+"</code></pre>");
    out.push("\\x00"+(blocks.length-1)+"\\x00");
  }
  return out.join("\\n");
}
function md(text){
  const blocks=[];
  let work=extractFences(text,blocks);
  const inline=[];
  work=work.replace(/`([^`\\n]+)`/g,(m,code)=>{ inline.push("<code>"+esc(code)+"</code>"); return "\\x01"+(inline.length-1)+"\\x01"; });
  work=esc(work);
  work=work.replace(/\\[([^\\]]+)\\]\\((https?:[^)\\s]+)\\)/g,'<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  work=work.replace(/\\*\\*([^*]+)\\*\\*/g,"<strong>$1</strong>");
  work=work.replace(/(^|[\\s(])\\*([^*\\n]+)\\*/g,"$1<em>$2</em>");
  const out=[]; let list=null;
  const closeList=()=>{ if(list){ out.push("</"+list+">"); list=null; } };
  for(const rawLine of work.split("\\n")){
    const line=rawLine.replace(/\\s+$/,"");
    if(!line.trim()){ closeList(); continue; }
    const code=line.match(/^\\x00(\\d+)\\x00$/);
    if(code){ closeList(); out.push(blocks[+code[1]]); continue; }
    const heading=line.match(/^(#{1,6})\\s+(.*)$/);
    if(heading){ closeList(); const lv=heading[1].length; out.push("<h"+lv+">"+heading[2]+"</h"+lv+">"); continue; }
    if(/^(-{3,}|\\*{3,}|_{3,})$/.test(line.trim())){ closeList(); out.push("<hr>"); continue; }
    const quote=line.match(/^&gt;\\s?(.*)$/);
    if(quote){ closeList(); out.push("<blockquote>"+quote[1]+"</blockquote>"); continue; }
    const ul=line.match(/^\\s*[-*+]\\s+(.*)$/);
    if(ul){ if(list!=="ul"){ closeList(); out.push("<ul>"); list="ul"; } out.push("<li>"+ul[1]+"</li>"); continue; }
    const ol=line.match(/^\\s*\\d+[.)]\\s+(.*)$/);
    if(ol){ if(list!=="ol"){ closeList(); out.push("<ol>"); list="ol"; } out.push("<li>"+ol[1]+"</li>"); continue; }
    closeList();
    out.push("<p>"+line+"</p>");
  }
  closeList();
  return out.join("").replace(/\\x01(\\d+)\\x01/g,(m,i)=>inline[+i]);
}
function contentBlocks(content){
  if(typeof content==="string") return [{type:"text",text:content}];
  if(!Array.isArray(content)) return [];
  return content;
}
function renderBlock(b){
  if(!b||typeof b!=="object") return "";
  const t=b.type;
  if(t==="text") return md(b.text??"");
  if(t==="thinking") return '<div class="sys"><span class="entry-kind">thinking</span>'+md(b.thinking??b.text??"")+"</div>";
  if(t==="image"){ const mt=b.mimeType||b.mime_type||"image/png"; return '<img src="data:'+esc(mt)+";base64,"+esc(b.data??"")+'">'; }
  if(t==="toolCall"||t==="tool_call"){ return renderToolCall(b); }
  return "<pre>"+esc(JSON.stringify(b,null,2))+"</pre>";
}
function renderBash(m){
  const cancelled=!!m.cancelled;
  const exitCode=m.exitCode??m.exit_code;
  const isErr=cancelled||(exitCode!==undefined&&exitCode!==null&&exitCode!==0);
  let html='<div class="bash'+(isErr?" error":"")+'"><div class="cmd">$ '+esc(m.command??"")+"</div>";
  if(m.output) html+="<pre>"+esc(m.output)+"</pre>";
  if(m.truncated){
    const full=m.fullOutputPath??m.full_output_path;
    html+='<div class="note">(truncated'+(full?" &middot; full output: "+esc(full):"")+")</div>";
  }
  if(cancelled) html+='<div class="note">(cancelled)</div>';
  else if(exitCode!==undefined&&exitCode!==null&&exitCode!==0) html+='<div class="note">(exit '+esc(exitCode)+")</div>";
  return html+"</div>";
}
function renderToolCall(b){
  const name=b.name||b.toolName||b.tool_name||"tool";
  const args=b.arguments??b.args??b.input??{};
  return '<details class="tool"><summary>&#9654; '+esc(name)+'</summary><div class="body"><div class="lbl">arguments</div><pre>'+esc(JSON.stringify(args,null,2))+"</pre></div></details>";
}
function renderToolResult(b){
  const name=b.toolName||b.tool_name||"result";
  const isErr=b.isError||b.is_error;
  let inner="";
  const c=b.content;
  if(Array.isArray(c)){ inner=c.map(renderBlock).join(""); }
  else inner="<pre>"+esc(typeof c==="string"?c:JSON.stringify(c,null,2))+"</pre>";
  return '<details class="tool"><summary>'+esc(name)+(isErr?' <span class="tool-error">(error)</span>':"")+'</summary><div class="body">'+inner+"</div></details>";
}
function renderMessage(m){
  const role=m.role||"message";
  if(role==="toolResult"||role==="tool_result"){
    // always return a node: callers appendChild() the result
    const wrap=document.createElement("div");
    wrap.innerHTML=renderToolResult(m);
    return wrap;
  }
  if(role==="bashExecution"||role==="bash_execution"){
    // Side-channel bash runs carry no content blocks (pi's template.js renders
    // these explicitly too), so the command and its output are the message.
    const div=document.createElement("div");
    div.className="msg bash-msg";
    div.innerHTML='<span class="role">bash</span>'+renderBash(m);
    return div;
  }
  const div=document.createElement("div");
  div.className="msg "+({user:"user",assistant:"assistant",system:"system"}[role]||"");
  let html='<span class="role">'+esc(role)+"</span>";
  if(role==="system"){ div.className="sys"; html=md(m.text??m.content??""); div.innerHTML=html; return div; }
  for(const b of contentBlocks(m.content)) html+=renderBlock(b);
  div.innerHTML=html;
  return div;
}
// Entries are already the leafId branch path, ordered root -> leaf (computed
// at export time, like pi's `getPath`); entries on abandoned branches stay in
// the session file but never reach the report.
function render(){
  const app=document.getElementById("app");
  app.innerHTML="";
  const h=DATA.header||{};
  const top=document.createElement("header");
  top.className="top";
  top.innerHTML="<h1>karen session</h1>"+'<div class="meta">'+esc(h.id??"")+(h.cwd?" &middot; "+esc(h.cwd):"")+((h.createdAt??h.timestamp)?" &middot; "+esc(new Date(h.createdAt??h.timestamp).toLocaleString()):"")+"</div>";
  app.appendChild(top);
  if(DATA.systemPrompt){ const s=document.createElement("div"); s.className="sys"; s.innerHTML='<span class="entry-kind">system prompt</span>'+md(DATA.systemPrompt); app.appendChild(s); }
  for(const e of (DATA.entries||[])){
    const m=e.message||e;
    if(m&&(m.role||m.type==="message")) app.appendChild(renderMessage(m));
    else if(e.type==="compaction"||e.type==="branch_summary"||e.type==="branchSummary"){
      const s=document.createElement("div"); s.className="sys";
      s.innerHTML='<span class="entry-kind">'+esc(e.type)+"</span>"+md(e.summary??e.text??JSON.stringify(e));
      app.appendChild(s);
    }
  }
}
render();
</script>
</body>
</html>
"""
