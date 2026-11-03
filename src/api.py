"""HTTP layer over the guarded LangGraph agent.

Endpoints:
  GET    /guardrails                  -> registry (static NeMo + uploaded .md files)
  POST   /guardrails/upload           -> upload .md guardrail files
  DELETE /guardrails/{guardrail_id}   -> delete an uploaded .md guardrail file
  POST   /run                         -> start a run with the user's guardrail selection
  POST   /approve                     -> resume a run paused at the Jira approval step
  GET    /                            -> the built React UI (ui/dist), when it exists
"""
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent import graph, warm_up
from guardrails_registry import (
    delete_custom_md_guardrail,
    list_guardrails,
    resolve_guardrails,
    save_custom_md_guardrail,
    visible,
)

log = logging.getLogger(__name__)
UI_DIST = Path(__file__).resolve().parent / "ui" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    warm_up()
    yield


app = FastAPI(title="SRE Guarded Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Schemas ----------------
class RunRequest(BaseModel):
    message: str = Field(min_length=1)
    enabled_guardrails: list[str] = []
    # Optional: allow sending new .md files directly inside the /run payload as well
    uploaded_md_files: dict[str, str] = {}


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]


class RunResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "blocked", "pending_approval"]
    reply: str | None = None
    pending: dict | None = None
    enabled_guardrails: list[str] = []
    trace: list[str] = []


# ---------------- Helpers ----------------
def _drive(graph_input, thread_id: str) -> RunResponse:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    pending, trace = None, []

    start = time.time()
    for update in graph.stream(graph_input, config=config, stream_mode="updates"):
        for node_name, output in update.items():
            if node_name == "__interrupt__":
                pending = output[0].value
                continue
            trace.append(node_name)

    values = graph.get_state(config).values
    messages = values.get("messages", [])
    reply = messages[-1].content if messages else None

    if pending:
        status = "pending_approval"
    elif values.get("blocked"):
        status = "blocked"
    else:
        status = "completed"

    log.info("thread=%s status=%s %.1fs", thread_id, status, time.time() - start)
    return RunResponse(
        thread_id=thread_id,
        status=status,
        reply=reply,
        pending=pending,
        enabled_guardrails=visible(values.get("enabled_guardrails", [])),
        trace=trace,
    )


_resuming: set[str] = set()
_resuming_lock = threading.Lock()


# ---------------- Routes ----------------
@app.get("/guardrails")
def get_guardrails():
    """Returns both static NeMo rails and user-uploaded .md guardrails."""
    return list_guardrails()


@app.post("/guardrails/upload")
def upload_guardrails(files: list[UploadFile] = File(...)):
    """Upload one or more .md files to become active guardrails."""
    added = []
    for f in files:
        if not (f.filename or "").lower().endswith(".md"):
            raise HTTPException(status_code=422, detail=f"Only .md files are allowed (got '{f.filename}')")
        content = f.file.read().decode("utf-8", errors="replace")
        try:
            item = save_custom_md_guardrail(f.filename, content)
            added.append(item)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            # CHANGED. Design rationale: indexing talks to Supabase and Ollama, so it can
            # fail for reasons unrelated to the file. Without this, the UI only saw a bare
            # "500 Internal Server Error"; now it gets the real reason, and `check()` in
            # api.ts already shows `detail` to the user. 502 = an upstream service failed.
            log.exception("Indexing failed for %s", f.filename)
            raise HTTPException(status_code=502, detail=f"Could not index '{f.filename}': {e}")
    return {"added": added, "guardrails": list_guardrails()}

@app.delete("/guardrails/{guardrail_id:path}")
def remove_guardrail(guardrail_id: str):
    """Delete a custom .md guardrail file so it is no longer checked."""
    try:
        deleted = delete_custom_md_guardrail(guardrail_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Guardrail '{guardrail_id}' not found")
    return {"deleted": guardrail_id, "guardrails": list_guardrails()}


@app.post("/run", response_model=RunResponse)
def run(req: RunRequest):
    selected = list(req.enabled_guardrails)

    # If the UI sent inline .md files with the request, save & enable them automatically
    for filename, content in req.uploaded_md_files.items():
        saved = save_custom_md_guardrail(filename, content)
        if saved["id"] not in selected:
            selected.append(saved["id"])

    try:
        resolve_guardrails(selected)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    thread_id = str(uuid.uuid4())
    return _drive(
        {
            "messages": [("user", req.message)],
            "blocked": False,
            "enabled_guardrails": selected,
        },
        thread_id,
    )


@app.post("/approve", response_model=RunResponse)
def approve(req: ApproveRequest):
    with _resuming_lock:
        if req.thread_id in _resuming:
            raise HTTPException(status_code=409, detail="This approval is already being processed")
        _resuming.add(req.thread_id)
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        snapshot = graph.get_state(config)

        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Unknown thread_id")
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="This run is not waiting for approval")

        return _drive(Command(resume=req.decision), req.thread_id)
    finally:
        with _resuming_lock:
            _resuming.discard(req.thread_id)


if UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")