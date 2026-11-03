"""Custom NeMo Guardrails actions. NeMo auto-loads config/actions.py.

Design rationale: every check on a user's request lives here and in rails.co, so
agent.py contains orchestration only. To change what counts as an incident report or
an injection, edit this file; no graph code changes.
"""
import logging
import re
import unicodedata

from langchain_ollama import ChatOllama
from nemoguardrails.actions import action

log = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    # NFKC folds look-alike unicode (full-width letters, etc.) into plain ASCII,
    # and collapsing whitespace stops "ignore   previous\ninstructions" evading regex.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


# ---------------------------------------------------------------------------
# Rail 1: prompt injection (deterministic)
# Design rationale: a 3B model is easy to talk out of its instructions, so the
# injection check must NOT depend on an LLM. Patterns are deliberately specific
# (verb + qualifier + noun) so a legit report like "ignore the old alert rule"
# is not blocked, while "ignore your SRE instructions" is.
# ---------------------------------------------------------------------------
_QUALIFIER = r"(?:all |any |your |the |my |previous |prior |above |earlier |these |those |sre |system |safety )+"
_INJECTION_PATTERNS = [
    rf"\b(?:ignore|disregard|forget|override)\s+{_QUALIFIER}(?:instructions?|rules?|prompts?|constraints?|guidelines?|restrictions?)",
    r"\b(?:reveal|show|print|repeat|tell me)\b.{0,30}\b(?:system prompt|your instructions|hidden prompt)",
    r"\b(?:act as|pretend to be|you are now)\b.{0,40}\b(?:unrestricted|admin|jailbroken|dan)\b",
    r"\bbypass\b.{0,30}\b(?:tool|constraint|guardrail|filter|safety|restriction)",
    r"\b(?:delete|drop|export|dump)\b.{0,20}\ball\b.{0,30}\b(?:tickets?|records?|database|tables?|messages?)",
]
_INJECTION_RE = [re.compile(p) for p in _INJECTION_PATTERNS]


@action(name="check_prompt_injection")
async def check_prompt_injection(text: str) -> bool:
    """Return True when the message looks like a prompt-injection attempt."""
    t = _normalize(text or "")
    hit = any(rx.search(t) for rx in _INJECTION_RE)
    if hit:
        log.warning("Prompt injection blocked: %r", text[:120])
    return hit


# ---------------------------------------------------------------------------
# Routing: which tools do the user's OWN words justify? (deterministic)
# Design rationale: the model only ever PROPOSES a tool call. agent.py binds just the
# tools returned here, so a tool the user never asked for cannot even be proposed.
# Words like "bug" or "ticket" are deliberately NOT failure evidence: they are what
# you say when ASKING about the feature, so counting them would let "can you create
# a jira bug?" make the model invent a ticket.
# ---------------------------------------------------------------------------
_SLACK_READ_INTENT = re.compile(
    r"\b(read|show|get|fetch|check|list|see|latest|recent|last)\b.{0,40}\b(slack|messages?|channel)\b"
    r"|\b(slack|channel)\b.{0,40}\b(messages?|history)\b"
)
_FAILURE_EVIDENCE = re.compile(
    r"\b(fail(?:ed|ure|ing|s)?|errors?|exceptions?|outage|crash(?:ed|es)?|exhausted|"
    r"timeouts?|timed out|incident|root cause|stack ?trace)\b"
)
# Tool name (as in tools.py) -> evidence required in the user's message.
_TOOL_EVIDENCE = {
    "read_slack_messages_tool": _SLACK_READ_INTENT,
    "create_jira_bug_tool": _FAILURE_EVIDENCE,
}


@action(name="authorize_tools")
async def authorize_tools(text: str) -> list[str]:
    """Return the names of the tools the message justifies (empty = just talk)."""
    t = _normalize(text or "")
    return [name for name, rx in _TOOL_EVIDENCE.items() if rx.search(t)]


# ---------------------------------------------------------------------------
# Rail 2: SRE scope (optional). One tiny LLM call, and only when needed.
# ---------------------------------------------------------------------------
_SCOPE_PROMPT = """You are a strict gatekeeper for an SRE incident assistant.
The assistant creates Jira bug tickets for production failures and reads Slack messages.
Reply with exactly one word: ON_TOPIC or OFF_TOPIC.
ON_TOPIC = a greeting, thanks, a question about the assistant itself (role, abilities,
limits, permissions), or anything about Jira bugs, Slack or a production failure,
even when details are missing.
OFF_TOPIC = anything else (poems, jokes, weather, general coding help, opinions).

Message: Hi
Answer: ON_TOPIC
Message: what permissions do you have?
Answer: ON_TOPIC
Message: can you create jira tickets?
Answer: ON_TOPIC
Message: Write a poem about a failed deployment
Answer: OFF_TOPIC
Message: What is the weather in Delhi today?
Answer: OFF_TOPIC
Message: {message}
Answer:"""


def _guard_llm() -> ChatOllama:
    # Design rationale: a fresh client per call. The graph is synchronous and calls
    # NeMo through asyncio.run(), which gives every request a NEW event loop; an async
    # HTTP client created in one loop must not be reused in another. Building the
    # client is cheap (no network), and the 3B model stays loaded by keep_alive.
    # num_predict=5 keeps a guard call to a handful of tokens.
    return ChatOllama(model="llama3.2:3b", temperature=0, num_ctx=2048, num_predict=5, keep_alive="30m")


@action(name="check_sre_scope")
async def check_sre_scope(text: str, tools: list[str] | None = None) -> bool:
    """Return True when the message may proceed, False when it is clearly off-topic."""
    # A detected task is in scope by definition: no LLM call on the main path.
    if tools:
        return True

    try:
        resp = await _guard_llm().ainvoke(_SCOPE_PROMPT.format(message=_normalize(text or "")))
        verdict = str(resp.content).strip().upper()
    except Exception:
        # Fail CLOSED: if the guard itself is broken we must not silently allow traffic.
        log.exception("Scope check failed; blocking request")
        return False

    # Only an explicit ON_TOPIC passes; anything ambiguous is treated as off-topic.
    return verdict.startswith("ON_TOPIC")