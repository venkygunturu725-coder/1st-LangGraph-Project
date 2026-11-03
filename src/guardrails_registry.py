"""Single source of truth for static NeMo guardrails + dynamic .md guardrails."""
import re
from pathlib import Path
from typing import TypedDict

CUSTOM_MD_DIR = Path(__file__).resolve().parent / "config" / "custom_guardrails"
CUSTOM_MD_DIR.mkdir(parents=True, exist_ok=True)
from rag_guardrails import index_guardrail_file, remove_guardrail_from_index


class Guardrail(TypedDict, total=False):
    label: str
    description: str
    flow: str          # Colang flow name in config/rails.co (empty for .md files)
    refusal: str       # Fixed refusal message
    cost: str
    locked: bool       # True = always runs, user cannot deselect it
    internal: bool
    custom_md: bool    # True = user-uploaded .md guardrail file
    filename: str      # Original .md filename


GUARDRAILS: dict[str, Guardrail] = {
    "prompt_injection": {
        "label": "Prompt injection",
        "description": "Blocks attempts to override or reveal instructions",
        "flow": "check prompt injection",
        "refusal": "Security violation: Prompt injection attempt detected. Request blocked.",
        "cost": "regex only, no LLM call",
        "locked": True,
        "internal": False,
    },
    "route_request": {
        "label": "Request routing",
        "description": "Decides which tools the request justifies",
        "flow": "route request",
        "refusal": "",
        "cost": "regex only, no LLM call",
        "locked": True,
        "internal": True,
    },
    "sre_scope": {
        "label": "SRE scope",
        "description": "Refuses off-topic requests with a fixed message",
        "flow": "check sre scope",
        "refusal": (
            "I am restricted to SRE operational tasks: creating Jira bug tickets and "
            "reading Slack notifications. I cannot assist with general queries."
        ),
        "cost": "one small LLM call, only when no task is detected",
        "locked": False,
        "internal": False,
    },
}


def _safe_md_filename(filename: str) -> str:
    name = Path(filename).name
    if not name.lower().endswith(".md"):
        name += ".md"
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _md_id(filename: str) -> str:
    return f"md:{filename}"


def load_custom_md_guardrails() -> dict[str, Guardrail]:
    """Load all user-uploaded .md files from config/custom_guardrails/."""
    custom: dict[str, Guardrail] = {}
    for path in sorted(CUSTOM_MD_DIR.glob("*.md")):
        gid = _md_id(path.name)
        first_line = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            cleaned = line.lstrip("#").strip()
            if cleaned:
                first_line = cleaned[:80]
                break
        custom[gid] = {
            "label": path.stem.replace("_", " ").replace("-", " ").title(),
            "description": first_line or f"Custom policy from {path.name}",
            "flow": "",
            "refusal": f"Request blocked by custom policy ({path.name}).",
            "cost": "checked in-prompt (0 extra LLM calls)",
            "locked": False,
            "internal": False,
            "custom_md": True,
            "filename": path.name,
        }
    return custom


def all_guardrails() -> dict[str, Guardrail]:
    """Combined dictionary of static NeMo guardrails + uploaded .md guardrails."""
    return {**GUARDRAILS, **load_custom_md_guardrails()}


def save_custom_md_guardrail(filename: str, content: str) -> dict:
    """Save a user-uploaded .md file, chunk it, and embed it in the vector DB."""
    safe_name = _safe_md_filename(filename)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Uploaded file '{filename}' is empty.")

    full_text = "\n".join(lines)
    target = CUSTOM_MD_DIR / safe_name
    target.write_text(full_text, encoding="utf-8")
    gid = _md_id(safe_name)

    try:
        index_guardrail_file(safe_name, full_text, gid)
    except Exception:
        # CHANGED. Design rationale: the file on disk and the vector index must agree.
        # Before, a failed index (e.g. missing Supabase table) left the .md on disk, so
        # the UI listed it as an active guardrail while nothing was actually enforced.
        # Rolling the file back makes the upload all-or-nothing; the exception is
        # re-raised so api.py can report the real reason to the UI.
        target.unlink(missing_ok=True)
        raise

    return {"id": gid, **load_custom_md_guardrails()[gid]}


def delete_custom_md_guardrail(guardrail_id: str) -> bool:
    """Delete a user-uploaded .md guardrail from disk and the vector DB."""
    if not guardrail_id.startswith("md:"):
        raise ValueError("Only custom .md guardrails can be deleted.")

    safe_name = _safe_md_filename(guardrail_id.removeprefix("md:"))
    target = CUSTOM_MD_DIR / safe_name

    # CHANGED. Design rationale: always purge the index, even when the file is already
    # gone from disk. Otherwise chunks from an earlier failed delete or a manual file
    # removal stay in Supabase and keep matching searches for a guardrail nobody sees.
    remove_guardrail_from_index(guardrail_id)

    if target.exists():
        target.unlink()
        return True
    return False


# def get_custom_md_contents(ids: list[str]) -> str:
#     """Return the combined markdown text for all selected `md:*` guardrail IDs."""
#     blocks: list[str] = []
#     for gid in ids:
#         if gid.startswith("md:"):
#             safe_name = _safe_md_filename(gid.removeprefix("md:"))
#             path = CUSTOM_MD_DIR / safe_name
#             if path.exists():
#                 blocks.append(f"[{safe_name}]\n{path.read_text(encoding='utf-8')}")
#     return "\n\n".join(blocks)


def list_guardrails() -> list[dict]:
    """What a UI or CLI shows in the picker (the '/' menu). Internal steps are hidden."""
    combined = all_guardrails()
    return [{"id": gid, **g} for gid, g in combined.items() if not g.get("internal")]


def resolve_guardrails(selected: list[str] | None) -> list[str]:
    """Turn a user selection into the effective, ordered list of guardrail ids."""
    combined = all_guardrails()
    requested = set(selected or [])

    unknown = requested - combined.keys()
    if unknown:
        raise ValueError(f"Unknown guardrail id(s): {sorted(unknown)}")

    return [
        gid for gid, g in combined.items()
        if g.get("locked") or gid in requested
    ]


def flows_for(ids: list[str]) -> tuple[str, ...]:
    """NeMo flow names for static guardrail ids (skips custom .md files which have flow='')."""
    combined = all_guardrails()
    return tuple(
        combined[gid]["flow"]
        for gid in ids
        if gid in combined and combined[gid].get("flow")
    )


def visible(ids: list[str]) -> list[str]:
    """Ids a person should see ('Checked by'): internal plumbing filtered out."""
    combined = all_guardrails()
    return [gid for gid in ids if gid in combined and not combined[gid].get("internal")]