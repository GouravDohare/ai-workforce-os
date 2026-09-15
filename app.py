import os
import re
import json
import html
import time
import uuid
import hmac
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from openai import OpenAI

# ============================================================
# AI Workforce OS v0.3.12
# Clean rebuild based on lessons from v0.2 -> v0.3.11.
# ============================================================

APP_VERSION = "0.3.12"
SCHEMA_VERSION = "312-1"

BASE_DIR = Path(__file__).resolve().parent
DB = BASE_DIR / os.getenv("WORKFORCE_DB", "workforce_v0312.db")

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))

DEFAULT_GOAL_BUDGET = float(os.getenv("DEFAULT_GOAL_BUDGET_USD", "5.0"))
MAX_GOAL_BUDGET = float(os.getenv("MAX_GOAL_BUDGET_USD", "50.0"))
MAX_TASKS = int(os.getenv("MAX_TASKS_PER_GOAL", "10"))
MAX_AGENTS = int(os.getenv("MAX_AGENT_INSTANCES_PER_GOAL", "8"))
MAX_TASK_WORKERS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))
MAX_GOAL_WORKERS = int(os.getenv("MAX_CONCURRENT_GOALS", "1"))
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "2"))

ENABLE_WEB_RESEARCH = os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true"
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "").strip()

# Approximate public API prices for local accounting only.
# Override these with environment variables if needed.
MODEL_PRICES = {
    "gpt-5-mini": (0.25, 2.00),
}
INPUT_PRICE_PER_MTOK = float(os.getenv("OPENAI_INPUT_PRICE_PER_MTOK", "0.25"))
OUTPUT_PRICE_PER_MTOK = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_MTOK", "2.0"))

app = FastAPI(title="AI Workforce OS", version=APP_VERSION)

goal_executor = ThreadPoolExecutor(max_workers=MAX_GOAL_WORKERS)
task_executor = ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS)
goal_futures = {}
future_lock = threading.Lock()


# ----------------------------
# Database
# ----------------------------

def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return str(uuid.uuid4())


def esc(value):
    return html.escape(str(value or ""))


def db():
    conn = sqlite3.connect(DB, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def fetch(sql, args=()):
    conn = db()
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


def fetch_one(sql, args=()):
    rows = fetch(sql, args)
    return rows[0] if rows else None


def write(sql, args=()):
    conn = db()
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def log_event(run_id=None, goal_id=None, task_id=None, kind="info",
              message="", payload=None):
    run = fetch_one("SELECT goal_id FROM runs WHERE id=?", (run_id,)) if run_id else None
    goal = goal_id or (run["goal_id"] if run else None)
    company = fetch_one("SELECT id FROM companies ORDER BY created_at LIMIT 1")
    write(
        """INSERT INTO events
           (id,company_id,run_id,goal_id,task_id,kind,message,payload_json,created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            uid(),
            company["id"] if company else None,
            run_id,
            goal,
            task_id,
            kind,
            message,
            json.dumps(payload or {}, ensure_ascii=False),
            now(),
        ),
    )


def init_db():
    conn = db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS companies(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_profiles(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            instructions TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS goals(
            id TEXT PRIMARY KEY,
            company_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            criteria TEXT NOT NULL,
            budget REAL NOT NULL,
            spent REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL,
            current_run_id TEXT,
            plan_json TEXT,
            final_output TEXT,
            replan_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL,
            parent_run_id TEXT,
            run_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            started_at TEXT,
            ended_at TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_instances(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            profile_id TEXT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            instructions TEXT NOT NULL,
            capabilities_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tasks(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            plan_task_id TEXT NOT NULL,
            agent_instance_id TEXT NOT NULL,
            title TEXT NOT NULL,
            instructions TEXT NOT NULL,
            required INTEGER NOT NULL DEFAULT 1,
            requires_web INTEGER NOT NULL DEFAULT 0,
            phase TEXT NOT NULL DEFAULT 'work',
            status TEXT NOT NULL,
            output TEXT,
            structured_output_json TEXT,
            confidence REAL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 2,
            error_type TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS task_dependencies(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            upstream_task_id TEXT NOT NULL,
            downstream_task_id TEXT NOT NULL,
            condition TEXT NOT NULL DEFAULT 'completed',
            required INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS task_attempts(
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            attempt_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            model TEXT,
            started_at TEXT,
            ended_at TEXT,
            latency_ms INTEGER,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost REAL NOT NULL DEFAULT 0,
            error_type TEXT,
            error_message TEXT,
            strategy_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_calls(
            id TEXT PRIMARY KEY,
            run_id TEXT,
            task_id TEXT,
            purpose TEXT NOT NULL,
            model TEXT NOT NULL,
            tool_mode TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            latency_ms INTEGER,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cost REAL NOT NULL DEFAULT 0,
            request_id TEXT,
            response_status TEXT,
            incomplete_reason TEXT,
            error_type TEXT,
            error_message TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS budget_ledger(
            id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL,
            run_id TEXT,
            task_id TEXT,
            amount REAL NOT NULL,
            kind TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evidence(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            task_id TEXT,
            claim TEXT NOT NULL,
            source_title TEXT,
            source_url TEXT,
            publisher TEXT,
            retrieved_at TEXT NOT NULL,
            snippet TEXT,
            confidence REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifacts(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            name TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            content TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifact_dependencies(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            downstream_task_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS handoffs(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            from_task_id TEXT NOT NULL,
            to_task_id TEXT NOT NULL,
            summary TEXT NOT NULL,
            evidence_ids_json TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            unknowns_json TEXT NOT NULL,
            confidence REAL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS evaluations(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            score REAL,
            passed INTEGER,
            checks_json TEXT NOT NULL,
            failures_json TEXT NOT NULL,
            recommendations_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS contradictions(
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            claim_a TEXT NOT NULL,
            claim_b TEXT NOT NULL,
            evidence_a_json TEXT NOT NULL,
            evidence_b_json TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL,
            resolution TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events(
            id TEXT PRIMARY KEY,
            company_id TEXT,
            run_id TEXT,
            goal_id TEXT,
            task_id TEXT,
            kind TEXT NOT NULL,
            message TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id);
        CREATE INDEX IF NOT EXISTS idx_deps_downstream ON task_dependencies(downstream_task_id);
        CREATE INDEX IF NOT EXISTS idx_events_goal ON events(goal_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);
        CREATE INDEX IF NOT EXISTS idx_model_calls_run ON model_calls(run_id);
        """
    )

    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('app_version',?)",
        (APP_VERSION,),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
        (SCHEMA_VERSION,),
    )

    if not conn.execute("SELECT id FROM companies LIMIT 1").fetchone():
        cid = uid()
        t = now()
        conn.execute(
            "INSERT INTO companies(id,name,created_at,updated_at) VALUES(?,?,?,?)",
            (cid, "My AI Company", t, t),
        )

        profiles = [
            ("CEO", "ceo", "Plan objectives, delegate work, identify capability gaps, and define verification.",
             ["planning", "delegation", "orchestration"]),
            ("Research Specialist", "research", "Research external facts and evidence; separate facts, assumptions, and unknowns.",
             ["web_research", "source_verification", "market_research"]),
            ("Engineering Analyst", "engineering", "Analyze technical feasibility, architecture, constraints, and risks.",
             ["technical_analysis", "architecture", "risk"]),
            ("Product & UX Specialist", "product", "Analyze customers, workflows, product scope, adoption, and UX.",
             ["customer", "product", "ux"]),
            ("Procurement Analyst", "procurement", "Analyze suppliers, vendors, costs, tooling, and commercial constraints.",
             ["procurement", "vendors", "commercial"]),
            ("Data Analyst", "data", "Perform calculations, metrics, quantitative checks, and data sanity checks.",
             ["data", "metrics", "quantitative"]),
            ("Quality & Risk Reviewer", "qa", "Challenge unsupported claims, omissions, contradictions, and weak evidence.",
             ["qa", "risk", "verification"]),
            ("Executive Report Writer", "report", "Synthesize verified work into a decision-ready answer without inventing evidence.",
             ["synthesis", "report", "decision_support"]),
        ]
        for name, role, instr, caps in profiles:
            conn.execute(
                """INSERT INTO agent_profiles
                   (id,name,role,instructions,capabilities_json,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (uid(), name, role, instr, json.dumps(caps), t, t),
            )
    conn.commit()
    conn.close()

    # Never leave active work pretending to be alive after a process restart.
    conn = db()
    stale_runs = conn.execute(
        "SELECT id,goal_id FROM runs WHERE status IN ('running','queued')"
    ).fetchall()
    for r in stale_runs:
        conn.execute(
            "UPDATE runs SET status='interrupted',ended_at=?,reason=? WHERE id=?",
            (now(), "Process restarted; active worker state was lost.", r["id"]),
        )
        conn.execute(
            "UPDATE tasks SET status='interrupted',error_type='process_restart',error_message=?,updated_at=? "
            "WHERE run_id=? AND status IN ('running','queued')",
            ("Worker process restarted before task completion.", now(), r["id"]),
        )
        conn.execute(
            "UPDATE agent_instances SET status='interrupted',updated_at=? WHERE run_id=? AND status IN ('running','queued')",
            (now(), r["id"]),
        )
        conn.execute(
            "UPDATE goals SET status='interrupted',updated_at=? WHERE id=? AND status IN ('running','queued')",
            (now(), r["goal_id"]),
        )
    conn.commit()
    conn.close()


def company_row():
    return fetch_one("SELECT * FROM companies ORDER BY created_at LIMIT 1")


def profile_for_role(role):
    return fetch_one("SELECT * FROM agent_profiles WHERE role=? LIMIT 1", (role,))


# ----------------------------
# Auth
# ----------------------------

def authorized(request):
    if not APP_ACCESS_TOKEN:
        return True
    auth = request.headers.get("Authorization", "")
    cookie = request.cookies.get("wf_auth", "")
    expected = hashlib.sha256(APP_ACCESS_TOKEN.encode()).hexdigest()
    if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:], APP_ACCESS_TOKEN):
        return True
    if cookie and hmac.compare_digest(cookie, expected):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if APP_ACCESS_TOKEN and request.url.path not in {"/health", "/login"}:
            if not authorized(request):
                return HTMLResponse("Unauthorized. Use /login.", status_code=401)
        return await call_next(request)


app.add_middleware(AuthMiddleware)


@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """<!doctype html><html><body style="font-family:system-ui;max-width:520px;margin:60px auto">
    <h1>AI Workforce OS</h1><form method="post" action="/login">
    <input name="token" type="password" placeholder="Access token" style="width:100%;padding:12px">
    <button style="margin-top:12px;padding:12px 20px">Sign in</button></form></body></html>"""


@app.post("/login")
def login(token: str = Form(...)):
    if not APP_ACCESS_TOKEN or not hmac.compare_digest(token, APP_ACCESS_TOKEN):
        return HTMLResponse("Invalid token.", status_code=401)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        "wf_auth",
        hashlib.sha256(token.encode()).hexdigest(),
        httponly=True,
        samesite="lax",
    )
    return response


# ----------------------------
# OpenAI integration
# ----------------------------

def get_client():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=key, timeout=OPENAI_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)


def model_for(purpose):
    env_map = {
        "planner": "OPENAI_PLANNER_MODEL",
        "research": "OPENAI_RESEARCH_MODEL",
        "qa": "OPENAI_QA_MODEL",
        "report": "OPENAI_REPORT_MODEL",
        "evaluator": "OPENAI_EVALUATOR_MODEL",
    }
    return os.getenv(env_map.get(purpose, ""), "") or DEFAULT_MODEL


def usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0
    return int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)


def cost_for(model, input_tokens, output_tokens):
    ip, op = MODEL_PRICES.get(model, (INPUT_PRICE_PER_MTOK, OUTPUT_PRICE_PER_MTOK))
    return (input_tokens / 1_000_000.0) * ip + (output_tokens / 1_000_000.0) * op


def response_text(resp):
    text = (getattr(resp, "output_text", "") or "").strip()
    if text:
        return text

    chunks = []
    for item in getattr(resp, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            value = getattr(part, "text", None)
            if value:
                chunks.append(str(value))
    return "\n".join(chunks).strip()


def response_status(resp):
    return str(getattr(resp, "status", "") or "")


def incomplete_reason(resp):
    detail = getattr(resp, "incomplete_details", None)
    return str(getattr(detail, "reason", "") or "") if detail else ""


def classify_error(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text:
        return "transient_timeout"
    if "ratelimit" in name or "429" in text or "rate limit" in text:
        return "transient_rate_limit"
    if any(x in name for x in ("connection", "apierror", "internalserver")):
        return "transient_provider"
    if any(x in text for x in ("502", "503", "504", "connection reset", "temporarily unavailable")):
        return "transient_provider"
    if "auth" in name or "401" in text or "403" in text:
        return "permanent_auth"
    if "badrequest" in name or "bad request" in text or "400" in text:
        return "permanent_request"
    return "unknown"


def parse_json_object(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Model returned empty output.")
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
        if raw.lower().startswith("json\n"):
            raw = raw[5:].lstrip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Structured JSON parse failed: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("Structured response must be a JSON object.")
    return obj


def extract_urls(text):
    urls = []
    for url in re.findall(r"https?://[^\s\]\)>,\"']+", text or ""):
        url = url.rstrip(".")
        if url not in urls:
            urls.append(url)
    return urls


def extract_citations(resp, fallback_text=""):
    found = []

    def add(title="", url="", publisher="", snippet=""):
        if not url:
            return
        if any(x["url"] == url for x in found):
            return
        found.append({
            "title": title or url,
            "url": url,
            "publisher": publisher or "",
            "snippet": snippet or "",
        })

    # Current Responses API exposes web-search annotations in message content.
    for item in getattr(resp, "output", []) or []:
        for part in getattr(item, "content", []) or []:
            for ann in getattr(part, "annotations", []) or []:
                url = getattr(ann, "url", None) or getattr(ann, "source_url", None)
                title = getattr(ann, "title", None) or ""
                if url:
                    add(title=title, url=url)

    # Some SDK/API versions expose sources on tool call action.
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", "") in ("web_search_call", "web_search_preview_call"):
            action = getattr(item, "action", None)
            for source in getattr(action, "sources", []) or []:
                add(
                    title=getattr(source, "title", "") or "",
                    url=getattr(source, "url", "") or "",
                    publisher=getattr(source, "publisher", "") or "",
                )

    for url in extract_urls(fallback_text):
        add(url=url)

    return found


def call_model(
    *,
    run_id=None,
    task_id=None,
    purpose="task",
    instructions="",
    input_text="",
    use_web=False,
    max_output_tokens=6000,
):
    model = model_for(purpose)
    started = time.time()
    call_id = uid()
    started_at = now()

    write(
        """INSERT INTO model_calls
           (id,run_id,task_id,purpose,model,tool_mode,started_at,created_at,response_status)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            call_id,
            run_id,
            task_id,
            purpose,
            model,
            "web_search" if use_web else "none",
            started_at,
            started_at,
            "started",
        ),
    )

    log_event(
        run_id=run_id,
        task_id=task_id,
        kind="model_start",
        message=f"Model call started: {purpose}",
        payload={"model": model, "web": bool(use_web), "max_output_tokens": max_output_tokens},
    )

    try:
        client = get_client()
        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": input_text,
            "max_output_tokens": max_output_tokens,
        }

        # IMPORTANT:
        # Do not attach reasoning_effort to web-search requests.
        # We learned from v0.3.11 that the deployed API/model combination rejects it.
        if use_web:
            if not ENABLE_WEB_RESEARCH:
                raise RuntimeError("Web research is disabled by configuration.")
            kwargs["tools"] = [{"type": "web_search"}]

        resp = client.responses.create(**kwargs)
        elapsed = int((time.time() - started) * 1000)
        text = response_text(resp)
        status = response_status(resp)
        reason = incomplete_reason(resp)
        in_tok, out_tok = usage(resp)
        cost = cost_for(model, in_tok, out_tok)

        if status == "incomplete":
            raise RuntimeError(
                f"Model response incomplete: reason={reason or 'unknown'}; "
                f"max_output_tokens={max_output_tokens}"
            )
        if status not in ("completed", ""):
            raise RuntimeError(f"Model response status={status}.")
        if not text:
            raise RuntimeError("Model returned empty output.")

        write(
            """UPDATE model_calls SET ended_at=?,latency_ms=?,input_tokens=?,output_tokens=?,
               cost=?,request_id=?,response_status=?,incomplete_reason=? WHERE id=?""",
            (
                now(),
                elapsed,
                in_tok,
                out_tok,
                cost,
                str(getattr(resp, "id", "") or ""),
                status or "completed",
                reason,
                call_id,
            ),
        )
        if run_id:
            write(
                "INSERT INTO budget_ledger(id,goal_id,run_id,task_id,amount,kind,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (
                    uid(),
                    fetch_one("SELECT goal_id FROM runs WHERE id=?", (run_id,))["goal_id"],
                    run_id,
                    task_id,
                    cost,
                    "model_call",
                    now(),
                ),
            )

        log_event(
            run_id=run_id,
            task_id=task_id,
            kind="model_ok",
            message=f"Model call completed: {purpose}",
            payload={
                "model": model,
                "latency_ms": elapsed,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost": cost,
                "request_id": str(getattr(resp, "id", "") or ""),
            },
        )
        return {
            "text": text,
            "response": resp,
            "model": model,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost": cost,
            "latency_ms": elapsed,
        }

    except Exception as exc:
        elapsed = int((time.time() - started) * 1000)
        etype = classify_error(exc)
        write(
            """UPDATE model_calls SET ended_at=?,latency_ms=?,response_status='error',
               error_type=?,error_message=? WHERE id=?""",
            (now(), elapsed, etype, str(exc)[:4000], call_id),
        )
        log_event(
            run_id=run_id,
            task_id=task_id,
            kind="model_error",
            message=f"Model call failed: {purpose}: {str(exc)[:500]}",
            payload={"error_type": etype, "model": model, "latency_ms": elapsed},
        )
        raise


# ----------------------------
# Planning
# ----------------------------

PLANNER_INSTRUCTIONS = """
You are the CEO/orchestrator of an AI workforce.

Your job is to turn a human objective into the smallest sufficient, dependency-aware
workforce. Do not create agents just for decoration.

Return ONLY one JSON object with:
{
  "agents": [
    {"name": "...", "role": "...", "instructions": "...", "capabilities": ["..."]}
  ],
  "tasks": [
    {
      "id": "T1",
      "title": "...",
      "instructions": "...",
      "agent_role": "...",
      "depends_on": [],
      "required": true,
      "requires_web": false,
      "phase": "work|qa|report",
      "max_attempts": 2
    }
  ],
  "verification": {
    "checks": ["..."],
    "pass_requirements": ["..."]
  }
}

Rules:
- Maximum 8 agents and 10 tasks.
- Every task must have exactly one agent_role.
- Dependencies must refer to task IDs.
- A task requiring external/current factual research must set requires_web=true.
- QA tasks should depend on the research/analysis they review.
- Report tasks should depend on all required upstream work.
- Avoid circular dependencies.
- State facts requiring validation as checks rather than inventing them.
- Design for a useful answer, not for maximum agent count.
"""

def fallback_plan(title, description):
    return {
        "agents": [
            {
                "name": "Research Specialist",
                "role": "research",
                "instructions": "Research externally verifiable facts and distinguish facts from assumptions and unknowns.",
                "capabilities": ["web_research", "source_verification"],
            },
            {
                "name": "Quality & Risk Reviewer",
                "role": "qa",
                "instructions": "Review evidence quality, unsupported claims, contradictions, omissions, and uncertainty.",
                "capabilities": ["qa", "verification"],
            },
            {
                "name": "Executive Report Writer",
                "role": "report",
                "instructions": "Synthesize verified work into a decision-ready answer without inventing evidence.",
                "capabilities": ["synthesis", "report"],
            },
        ],
        "tasks": [
            {
                "id": "T1",
                "title": "External research",
                "instructions": f"Research the objective using external sources. Objective: {description}",
                "agent_role": "research",
                "depends_on": [],
                "required": True,
                "requires_web": True,
                "phase": "work",
                "max_attempts": 2,
            },
            {
                "id": "T2",
                "title": "Source and quality verification",
                "instructions": "Review T1 for evidence quality, unsupported claims, contradictions, and omissions.",
                "agent_role": "qa",
                "depends_on": ["T1"],
                "required": True,
                "requires_web": False,
                "phase": "qa",
                "max_attempts": 2,
            },
            {
                "id": "T3",
                "title": "Executive report",
                "instructions": "Produce a decision-ready answer using only verified upstream work.",
                "agent_role": "report",
                "depends_on": ["T1", "T2"],
                "required": True,
                "requires_web": False,
                "phase": "report",
                "max_attempts": 2,
            },
        ],
        "verification": {
            "checks": ["Requested objective addressed", "Evidence and uncertainty represented"],
            "pass_requirements": ["Required tasks completed", "No critical verification failure"],
        },
    }


def validate_plan(plan):
    if not isinstance(plan, dict):
        raise ValueError("Planner output is not an object.")

    agents = plan.get("agents")
    tasks = plan.get("tasks")
    if not isinstance(agents, list) or not agents:
        raise ValueError("Planner returned no agents.")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Planner returned no tasks.")

    agents = agents[:MAX_AGENTS]
    tasks = tasks[:MAX_TASKS]
    roles = set()

    clean_agents = []
    for a in agents:
        role = str(a.get("role", "")).strip().lower()
        if not role:
            continue
        roles.add(role)
        clean_agents.append({
            "name": str(a.get("name") or role.title())[:120],
            "role": role[:80],
            "instructions": str(a.get("instructions") or "")[:2000],
            "capabilities": [str(x) for x in (a.get("capabilities") or [])][:12],
        })

    ids = set()
    clean_tasks = []
    for i, t in enumerate(tasks, 1):
        tid = str(t.get("id") or f"T{i}").strip()
        if tid in ids:
            tid = f"T{i}"
        ids.add(tid)

        role = str(t.get("agent_role") or "").strip().lower()
        if not role:
            raise ValueError(f"Task {tid} has no agent_role.")
        if role not in roles:
            # Create a minimal role definition rather than silently assigning work.
            clean_agents.append({
                "name": role.title(),
                "role": role,
                "instructions": "Execute the assigned task according to its contract.",
                "capabilities": [],
            })
            roles.add(role)

        deps = []
        for dep in t.get("depends_on") or []:
            dep = str(dep)
            if dep != tid and dep not in deps:
                deps.append(dep)

        clean_tasks.append({
            "id": tid,
            "title": str(t.get("title") or tid)[:180],
            "instructions": str(t.get("instructions") or "")[:5000],
            "agent_role": role,
            "depends_on": deps,
            "required": bool(t.get("required", True)),
            "requires_web": bool(t.get("requires_web", False)),
            "phase": str(t.get("phase") or "work")[:30],
            "max_attempts": max(1, min(int(t.get("max_attempts", 2) or 2), 3)),
        })

    # Remove dependencies on nonexistent task IDs.
    valid_ids = {t["id"] for t in clean_tasks}
    for t in clean_tasks:
        t["depends_on"] = [d for d in t["depends_on"] if d in valid_ids]

    # Basic cycle detection.
    graph = {t["id"]: set(t["depends_on"]) for t in clean_tasks}
    visiting, visited = set(), set()

    def visit(node):
        if node in visiting:
            raise ValueError("Planner produced a circular dependency.")
        if node in visited:
            return
        visiting.add(node)
        for dep in graph.get(node, set()):
            visit(dep)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)

    return {
        "agents": clean_agents[:MAX_AGENTS],
        "tasks": clean_tasks,
        "verification": plan.get("verification") or {},
    }


def build_plan(goal):
    try:
        result = call_model(
            run_id=goal["current_run_id"],
            purpose="planner",
            instructions=PLANNER_INSTRUCTIONS,
            input_text=(
                f"Objective title: {goal['title']}\n"
                f"Objective description: {goal['description']}\n"
                f"Success criteria: {goal['criteria']}\n"
                f"Budget: ${goal['budget']:.2f}\n"
            ),
            use_web=False,
            max_output_tokens=6000,
        )
        plan = validate_plan(parse_json_object(result["text"]))
        log_event(
            run_id=goal["current_run_id"],
            goal_id=goal["id"],
            kind="planner_ok",
            message="CEO plan created.",
            payload={"task_count": len(plan["tasks"]), "agent_count": len(plan["agents"])},
        )
        return plan
    except Exception as exc:
        log_event(
            run_id=goal["current_run_id"],
            goal_id=goal["id"],
            kind="planner_fallback",
            message=f"Planner failed; using objective-specific safe plan: {type(exc).__name__}: {str(exc)[:300]}",
        )
        return validate_plan(fallback_plan(goal["title"], goal["description"]))


# ----------------------------
# Workforce execution
# ----------------------------

def create_instances(run_id, plan):
    instances = {}
    for agent in plan["agents"]:
        profile = profile_for_role(agent["role"])
        instance_id = uid()
        t = now()
        write(
            """INSERT INTO agent_instances
               (id,run_id,profile_id,name,role,instructions,capabilities_json,status,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                instance_id,
                run_id,
                profile["id"] if profile else None,
                agent["name"],
                agent["role"],
                agent["instructions"],
                json.dumps(agent["capabilities"]),
                "ready",
                t,
                t,
            ),
        )
        instances[agent["role"]] = instance_id
    return instances


def create_tasks(run_id, plan, instances):
    mapping = {}
    for task in plan["tasks"]:
        agent_id = instances[task["agent_role"]]
        task_id = uid()
        t = now()
        write(
            """INSERT INTO tasks
               (id,run_id,plan_task_id,agent_instance_id,title,instructions,required,
                requires_web,phase,status,max_attempts,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                task_id,
                run_id,
                task["id"],
                agent_id,
                task["title"],
                task["instructions"],
                1 if task["required"] else 0,
                1 if task["requires_web"] else 0,
                task["phase"],
                "queued",
                task["max_attempts"],
                t,
                t,
            ),
        )
        mapping[task["id"]] = task_id

    for task in plan["tasks"]:
        for dep in task["depends_on"]:
            if dep not in mapping:
                continue
            write(
                """INSERT INTO task_dependencies
                   (id,run_id,upstream_task_id,downstream_task_id,condition,required,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    uid(),
                    run_id,
                    mapping[dep],
                    mapping[task["id"]],
                    "completed",
                    1,
                    now(),
                ),
            )
    return mapping


def dependency_state(task):
    deps = fetch(
        "SELECT * FROM task_dependencies WHERE downstream_task_id=?",
        (task["id"],),
    )
    if not deps:
        return "ready", []

    blocked = []
    for dep in deps:
        upstream = fetch_one("SELECT * FROM tasks WHERE id=?", (dep["upstream_task_id"],))
        if not upstream:
            blocked.append("missing_upstream")
            continue
        if dep["condition"] == "completed":
            if upstream["status"] == "completed":
                continue
            if upstream["status"] in ("failed", "blocked", "interrupted"):
                blocked.append(upstream["status"])
            else:
                return "waiting", []
    if blocked:
        return "blocked", blocked
    return "ready", []


def dependency_context(task):
    deps = fetch(
        """SELECT t.* FROM task_dependencies d
           JOIN tasks t ON t.id=d.upstream_task_id
           WHERE d.downstream_task_id=?""",
        (task["id"],),
    )
    parts = []
    evidence = []
    for d in deps:
        parts.append(
            f"UPSTREAM {d['plan_task_id']} / {d['title']} / status={d['status']}\n"
            f"{(d['output'] or '')[:9000]}"
        )
        evidence.extend(fetch(
            "SELECT * FROM evidence WHERE task_id=? ORDER BY created_at",
            (d["id"],),
        ))
    if not parts:
        return "No upstream task outputs.", evidence
    return "\n\n".join(parts), evidence


def record_evidence(run_id, task_id, citations, confidence=0.7):
    ids = []
    for c in citations:
        eid = uid()
        write(
            """INSERT INTO evidence
               (id,run_id,task_id,claim,source_title,source_url,publisher,retrieved_at,snippet,confidence,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eid,
                run_id,
                task_id,
                "Source supporting task findings.",
                c.get("title", ""),
                c.get("url", ""),
                c.get("publisher", ""),
                now(),
                c.get("snippet", ""),
                confidence,
                now(),
            ),
        )
        ids.append(eid)
    return ids


def execute_one_task(task):
    task_id = task["id"]
    run_id = task["run_id"]
    run = fetch_one("SELECT * FROM runs WHERE id=?", (run_id,))
    goal = fetch_one("SELECT * FROM goals WHERE id=?", (run["goal_id"],))
    agent = fetch_one("SELECT * FROM agent_instances WHERE id=?", (task["agent_instance_id"],))

    state, reasons = dependency_state(task)
    if state != "ready":
        if state == "blocked":
            write(
                "UPDATE tasks SET status='blocked',error_type='dependency_blocked',error_message=?,updated_at=? WHERE id=?",
                (json.dumps(reasons), now(), task_id),
            )
        return False

    # Claim task atomically enough for this single-process worker.
    write(
        "UPDATE tasks SET status='running',attempt_count=attempt_count+1,updated_at=? WHERE id=? AND status='queued'",
        (now(), task_id),
    )
    current = fetch_one("SELECT * FROM tasks WHERE id=?", (task_id,))
    if current["status"] != "running":
        return False

    attempt_number = current["attempt_count"]
    started = time.time()
    attempt_id = uid()
    write(
        """INSERT INTO task_attempts
           (id,task_id,attempt_number,status,model,started_at,strategy_json,created_at)
           VALUES(?,?,?,?,?,?,?,?)""",
        (
            attempt_id,
            task_id,
            attempt_number,
            "running",
            model_for("research" if task["requires_web"] else "task"),
            now(),
            json.dumps({"web": bool(task["requires_web"])}, ensure_ascii=False),
            now(),
        ),
    )
    log_event(
        run_id=run_id,
        task_id=task_id,
        kind="task_start",
        message=f"{agent['name']} started {task['title']}",
    )

    upstream_text, upstream_evidence = dependency_context(task)

    system = f"""
You are the {agent['name']} in an AI workforce.
Role: {agent['role']}.
Instructions: {agent['instructions']}

You are executing one task inside a governed workflow.
Never invent sources, data, citations, or completed work.
Clearly distinguish:
- sourced facts
- analysis/inference
- assumptions
- unknowns
- recommended validation

Return a useful, substantive answer, not a status message.
"""

    user = f"""
GOAL:
{goal['title']}

DESCRIPTION:
{goal['description']}

SUCCESS CRITERIA:
{goal['criteria']}

YOUR TASK:
{task['instructions']}

UPSTREAM WORK:
{upstream_text}

Produce the best work product possible within the task scope.
"""

    try:
        purpose = "research" if task["requires_web"] else (
            "qa" if task["phase"] == "qa" else
            "report" if task["phase"] == "report" else
            "task"
        )
        result = call_model(
            run_id=run_id,
            task_id=task_id,
            purpose=purpose,
            instructions=system,
            input_text=user,
            use_web=bool(task["requires_web"]),
            max_output_tokens=8000 if task["requires_web"] else (
                7000 if task["phase"] == "report" else 6000
            ),
        )

        text = result["text"]
        citations = extract_citations(result["response"], text) if task["requires_web"] else []
        evidence_ids = record_evidence(run_id, task_id, citations, 0.75) if citations else []

        confidence = 0.75 if text else 0.0
        write(
            """UPDATE tasks SET status='completed',output=?,confidence=?,
               error_type=NULL,error_message=NULL,updated_at=? WHERE id=?""",
            (text, confidence, now(), task_id),
        )

        cost = result["cost"]
        write(
            """UPDATE task_attempts SET status='completed',ended_at=?,latency_ms=?,
               input_tokens=?,output_tokens=?,cost=? WHERE id=?""",
            (
                now(),
                int((time.time() - started) * 1000),
                result["input_tokens"],
                result["output_tokens"],
                cost,
                attempt_id,
            ),
        )

        log_event(
            run_id=run_id,
            task_id=task_id,
            kind="task_complete",
            message=f"{agent['name']} completed {task['title']}",
            payload={"confidence": confidence, "evidence_count": len(evidence_ids)},
        )
        return True

    except Exception as exc:
        etype = classify_error(exc)
        retriable = etype.startswith("transient_") and attempt_number < current["max_attempts"]
        final_status = "queued" if retriable else "failed"

        write(
            """UPDATE tasks SET status=?,error_type=?,error_message=?,updated_at=? WHERE id=?""",
            (final_status, etype, str(exc)[:4000], now(), task_id),
        )
        write(
            """UPDATE task_attempts SET status='failed',ended_at=?,latency_ms=?,
               error_type=?,error_message=? WHERE id=?""",
            (
                now(),
                int((time.time() - started) * 1000),
                etype,
                str(exc)[:4000],
                attempt_id,
            ),
        )
        log_event(
            run_id=run_id,
            task_id=task_id,
            kind="task_retry" if retriable else "task_failed",
            message=(
                f"{task['title']} will retry: {etype}"
                if retriable else
                f"{task['title']} failed: {etype}"
            ),
        )
        return False


def run_evaluator(run_id):
    goal = get_goal_from_run(run_id)
    tasks = fetch("SELECT * FROM tasks WHERE run_id=? ORDER BY created_at", (run_id,))
    required = [t for t in tasks if t["required"]]
    completed = [t for t in required if t["status"] == "completed"]
    failed = [t for t in required if t["status"] in ("failed", "blocked", "interrupted")]

    if failed:
        write(
            """INSERT INTO evaluations
               (id,run_id,stage,status,score,passed,checks_json,failures_json,recommendations_json,created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                uid(), run_id, "pre_report", "not_run", 0.0, 0,
                json.dumps(["All required tasks completed"]),
                json.dumps([f"{t['title']}: {t['status']}" for t in failed]),
                json.dumps(["Retry or replan failed required work."]),
                now(),
            ),
        )
        log_event(run_id=run_id, kind="evaluation_skip", message="Evaluator not run: required work incomplete.")
        return False

    # Evaluator is deliberately lightweight and deterministic at this stage.
    checks = {
        "required_tasks_completed": len(completed) == len(required),
        "report_exists": any(t["phase"] == "report" and t["output"] for t in tasks),
        "evidence_present_for_web_tasks": all(
            fetch_one("SELECT id FROM evidence WHERE task_id=? LIMIT 1", (t["id"],))
            for t in tasks if t["requires_web"]
        ),
    }
    passed = all(checks.values())
    score = sum(1 for x in checks.values() if x) / max(1, len(checks))

    write(
        """INSERT INTO evaluations
           (id,run_id,stage,status,score,passed,checks_json,failures_json,recommendations_json,created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (
            uid(), run_id, "pre_report", "completed", score, 1 if passed else 0,
            json.dumps(checks),
            json.dumps([k for k,v in checks.items() if not v]),
            json.dumps([] if passed else ["Review failed checks before accepting the run."]),
            now(),
        ),
    )
    log_event(
        run_id=run_id,
        kind="evaluation_complete",
        message=f"Evaluator completed: {'passed' if passed else 'failed'}",
        payload={"score": score, "checks": checks},
    )
    return passed


def assemble_final(run_id):
    goal = get_goal_from_run(run_id)
    tasks = fetch(
        "SELECT * FROM tasks WHERE run_id=? ORDER BY phase,created_at",
        (run_id,),
    )
    report_tasks = [t for t in tasks if t["phase"] == "report" and t["status"] == "completed" and t["output"]]
    if not report_tasks:
        return None

    report = report_tasks[-1]["output"]
    evidence = fetch("SELECT * FROM evidence WHERE run_id=? ORDER BY created_at", (run_id,))

    if evidence:
        sources = "\n\nSources captured by the research task:\n"
        for e in evidence:
            sources += f"- {e['source_title'] or e['source_url']}: {e['source_url']}\n"
        report += sources

    return report


def execute_run(run_id):
    run = fetch_one("SELECT * FROM runs WHERE id=?", (run_id,))
    if not run:
        return

    goal = fetch_one("SELECT * FROM goals WHERE id=?", (run["goal_id"],))
    write("UPDATE runs SET status='running',started_at=? WHERE id=?", (now(), run_id))
    write("UPDATE goals SET status='running',updated_at=? WHERE id=?", (now(), goal["id"]))
    log_event(run_id=run_id, goal_id=goal["id"], kind="run_start", message="Workforce run started.")

    try:
        plan = build_plan(fetch_one("SELECT * FROM goals WHERE id=?", (goal["id"],)))
        write(
            "UPDATE goals SET plan_json=?,updated_at=? WHERE id=?",
            (json.dumps(plan, ensure_ascii=False, indent=2), now(), goal["id"]),
        )

        instances = create_instances(run_id, plan)
        create_tasks(run_id, plan, instances)

        # Dependency-aware scheduler. Independent ready tasks can run concurrently.
        while True:
            tasks = fetch("SELECT * FROM tasks WHERE run_id=? ORDER BY created_at", (run_id,))
            unfinished = [t for t in tasks if t["status"] in ("queued", "running")]
            if not unfinished:
                break

            ready = []
            for t in tasks:
                if t["status"] != "queued":
                    continue
                state, _ = dependency_state(t)
                if state == "ready":
                    ready.append(t)

            if ready:
                futures = [task_executor.submit(execute_one_task, dict(t)) for t in ready[:MAX_TASK_WORKERS]]
                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        log_event(run_id=run_id, kind="scheduler_error", message=str(exc)[:500])
                continue

            # No runnable work remains.
            if any(t["status"] == "running" for t in tasks):
                time.sleep(0.2)
                continue

            # Remaining queued tasks are blocked by failed dependencies.
            for t in tasks:
                if t["status"] == "queued":
                    state, reasons = dependency_state(t)
                    if state == "blocked":
                        write(
                            "UPDATE tasks SET status='blocked',error_type='dependency_blocked',error_message=?,updated_at=? WHERE id=?",
                            (json.dumps(reasons), now(), t["id"]),
                        )
            break

        evaluation_passed = run_evaluator(run_id)
        final_output = assemble_final(run_id)
        required = fetch(
            "SELECT * FROM tasks WHERE run_id=? AND required=1",
            (run_id,),
        )
        all_required_done = all(t["status"] == "completed" for t in required)

        if all_required_done and evaluation_passed and final_output:
            status = "completed"
            final = final_output
        else:
            status = "incomplete"
            final = (
                "INCOMPLETE: required work was not verified as ready. "
                "No unsupported final answer was accepted."
            )

        write(
            "UPDATE goals SET status=?,final_output=?,updated_at=? WHERE id=?",
            (status, final, now(), goal["id"]),
        )
        write(
            "UPDATE runs SET status=?,ended_at=? WHERE id=?",
            (status, now(), run_id),
        )
        log_event(
            run_id=run_id,
            goal_id=goal["id"],
            kind="run_complete",
            message=f"Run finished with status={status}.",
        )

    except Exception as exc:
        etype = classify_error(exc)
        write(
            "UPDATE goals SET status='failed',final_output=?,updated_at=? WHERE id=?",
            (f"RUN FAILED: {etype}: {str(exc)[:2000]}", now(), goal["id"]),
        )
        write(
            "UPDATE runs SET status='failed',ended_at=?,reason=? WHERE id=?",
            (now(), str(exc)[:2000], run_id),
        )
        log_event(
            run_id=run_id,
            goal_id=goal["id"],
            kind="run_failed",
            message=f"Run failed: {etype}: {str(exc)[:500]}",
        )


def start_goal(goal_id):
    goal = fetch_one("SELECT * FROM goals WHERE id=?", (goal_id,))
    if not goal:
        return None

    run_number = int(
        fetch_one(
            "SELECT COALESCE(MAX(run_number),0)+1 AS n FROM runs WHERE goal_id=?",
            (goal_id,),
        )["n"]
    )
    run_id = uid()
    write(
        """INSERT INTO runs
           (id,goal_id,parent_run_id,run_number,status,created_at)
           VALUES(?,?,?,?,?,?)""",
        (run_id, goal_id, goal["current_run_id"], run_number, "queued", now()),
    )
    write(
        "UPDATE goals SET current_run_id=?,status='queued',updated_at=? WHERE id=?",
        (run_id, now(), goal_id),
    )
    future = goal_executor.submit(execute_run, run_id)
    with future_lock:
        goal_futures[run_id] = future
    return run_id


# ----------------------------
# HTTP/UI
# ----------------------------

def page_shell(title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)} - AI Workforce OS</title>
<style>
body{{margin:0;background:#f5f6f8;color:#111;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:28px 18px 80px}}
.card{{background:white;border:1px solid #e0e3e8;border-radius:16px;padding:20px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.03)}}
h1{{font-size:30px;margin:4px 0 18px}} h2{{font-size:23px;margin:0 0 14px}}
label{{display:block;font-weight:650;margin:10px 0 6px}}
input,textarea{{box-sizing:border-box;width:100%;border:1px solid #ccd1d8;border-radius:10px;padding:11px;font:inherit}}
textarea{{min-height:110px;resize:vertical}}
button,.button{{display:inline-block;border:0;border-radius:10px;background:#111;color:#fff;padding:11px 16px;font-weight:650;text-decoration:none;cursor:pointer}}
button.secondary,.button.secondary{{background:#e9ebef;color:#111}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}
@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
pre{{white-space:pre-wrap;overflow:auto;background:#f6f7f9;border-radius:10px;padding:14px;font-size:13px;line-height:1.5}}
.status{{font-weight:700}} .muted{{color:#626975}} .ok{{color:#087443}} .bad{{color:#a51d2d}}
.row{{padding:12px 0;border-bottom:1px solid #eceef1}}
.small{{font-size:13px}} .pill{{display:inline-block;background:#eef0f3;border-radius:999px;padding:3px 8px;font-size:12px;margin-right:5px}}
</style></head><body><main>{body}</main></body></html>"""


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model": DEFAULT_MODEL,
        "db": DB.name,
    }


@app.get("/diagnostics/generation")
def diagnostic_generation():
    try:
        result = call_model(
            purpose="diagnostic",
            instructions="Reply with exactly: OK",
            input_text="Return the exact string OK and nothing else.",
            use_web=False,
            max_output_tokens=4096,
        )
        return {
            "status": "ok",
            "message": "Real Responses API generation succeeded.",
            "version": APP_VERSION,
            "model": result["model"],
            "output": result["text"],
            "latency_ms": result["latency_ms"],
            "max_output_tokens": 4096,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "version": APP_VERSION,
                "model": DEFAULT_MODEL,
            },
        )


@app.get("/diagnostics/web")
def diagnostic_web():
    try:
        result = call_model(
            purpose="diagnostic",
            instructions=(
                "Use web search to answer the user's request. "
                "Give a short factual answer and cite the sources you used."
            ),
            input_text="What is the OpenAI Responses API?",
            use_web=True,
            max_output_tokens=8000,
        )
        citations = extract_citations(result["response"], result["text"])
        return {
            "status": "ok",
            "message": "Real Responses API web search succeeded.",
            "version": APP_VERSION,
            "model": result["model"],
            "output": result["text"],
            "citations": citations,
            "latency_ms": result["latency_ms"],
            "max_output_tokens": 8000,
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "version": APP_VERSION,
                "model": DEFAULT_MODEL,
            },
        )


@app.get("/", response_class=HTMLResponse)
def home():
    goals = fetch("SELECT * FROM goals ORDER BY created_at DESC LIMIT 20")
    items = []
    for g in goals:
        items.append(
            f"""<div class="row">
            <a href="/goals/{g['id']}"><strong>{esc(g['title'])}</strong></a>
            <div class="small muted">Status: {esc(g['status'])} | Budget: ${g['budget']:.2f} | Spent: ${g['spent']:.4f}</div>
            </div>"""
        )
    body = f"""
    <div class="card">
      <h1>AI Workforce OS <span class="pill">v{APP_VERSION}</span></h1>
      <p class="muted">Governed multi-agent orchestration: plan, execute, verify, report.</p>
      <form method="post" action="/goals">
        <label>Objective</label>
        <input name="title" placeholder="e.g. Research the global industrial automation market" required>
        <label>Description</label>
        <textarea name="description" placeholder="What should the workforce accomplish?"></textarea>
        <label>Success criteria</label>
        <textarea name="criteria" placeholder="What must be true for the result to be accepted?"></textarea>
        <label>Budget (USD, optional)</label>
        <input name="budget" type="number" min="0.10" max="{MAX_GOAL_BUDGET}" step="0.01" placeholder="System default ${DEFAULT_GOAL_BUDGET:.2f}">
        <br><br><button type="submit">Create goal</button>
      </form>
    </div>
    <div class="card"><h2>Recent goals</h2>{''.join(items) or '<p class="muted">No goals yet.</p>'}</div>
    """
    return page_shell("Dashboard", body)


@app.post("/goals")
def create_goal(
    title: str = Form(...),
    description: str = Form(""),
    criteria: str = Form(""),
    budget: str = Form(""),
):
    try:
        parsed = float(budget) if budget.strip() else DEFAULT_GOAL_BUDGET
    except ValueError:
        parsed = DEFAULT_GOAL_BUDGET
    parsed = max(0.10, min(parsed, MAX_GOAL_BUDGET))

    c = company_row()
    gid = uid()
    t = now()
    desc = description.strip() or title.strip()
    crit = criteria.strip() or "Address the objective accurately; distinguish facts, assumptions, and unknowns."
    write(
        """INSERT INTO goals
           (id,company_id,title,description,criteria,budget,spent,status,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (gid, c["id"], title.strip(), desc, crit, parsed, 0.0, "created", t, t),
    )
    log_event(goal_id=gid, kind="goal_created", message=f"Human assigned objective: {title.strip()}")
    start_goal(gid)
    return RedirectResponse(f"/goals/{gid}", status_code=303)


@app.post("/goals/{goal_id}/retry")
def retry_goal(goal_id: str):
    goal = fetch_one("SELECT * FROM goals WHERE id=?", (goal_id,))
    if not goal:
        return HTMLResponse("Goal not found.", status_code=404)
    if goal["replan_count"] >= MAX_REPLANS:
        return HTMLResponse("Maximum replans reached.", status_code=409)
    write(
        "UPDATE goals SET replan_count=replan_count+1,updated_at=? WHERE id=?",
        (now(), goal_id),
    )
    start_goal(goal_id)
    return RedirectResponse(f"/goals/{goal_id}", status_code=303)


@app.get("/api/goals/{goal_id}")
def goal_api(goal_id: str):
    goal = fetch_one("SELECT * FROM goals WHERE id=?", (goal_id,))
    if not goal:
        return JSONResponse({"error": "not_found"}, status_code=404)
    run = fetch_one("SELECT * FROM runs WHERE id=?", (goal["current_run_id"],)) if goal["current_run_id"] else None
    tasks = fetch(
        "SELECT * FROM tasks WHERE run_id=? ORDER BY created_at",
        (run["id"],),
    ) if run else []
    events = fetch(
        "SELECT * FROM events WHERE goal_id=? ORDER BY created_at DESC LIMIT 100",
        (goal_id,),
    )
    evidence = fetch(
        "SELECT * FROM evidence WHERE run_id=? ORDER BY created_at",
        (run["id"],),
    ) if run else []
    evaluation = fetch_one(
        "SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
        (run["id"],),
    ) if run else None

    return {
        "version": APP_VERSION,
        "goal": dict(goal),
        "run": dict(run) if run else None,
        "tasks": [dict(x) for x in tasks],
        "events": [dict(x) for x in events],
        "evidence": [dict(x) for x in evidence],
        "evaluation": dict(evaluation) if evaluation else None,
    }


@app.get("/goals/{goal_id}", response_class=HTMLResponse)
def goal_page(goal_id: str):
    goal = fetch_one("SELECT * FROM goals WHERE id=?", (goal_id,))
    if not goal:
        return HTMLResponse("Goal not found.", status_code=404)

    run = fetch_one("SELECT * FROM runs WHERE id=?", (goal["current_run_id"],)) if goal["current_run_id"] else None
    tasks = fetch("SELECT * FROM tasks WHERE run_id=? ORDER BY created_at", (run["id"],)) if run else []
    events = fetch(
        "SELECT * FROM events WHERE goal_id=? ORDER BY created_at DESC LIMIT 80",
        (goal_id,),
    )
    evidence = fetch("SELECT * FROM evidence WHERE run_id=? ORDER BY created_at", (run["id"],)) if run else []
    evaluation = fetch_one(
        "SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
        (run["id"],),
    ) if run else None

    task_rows = []
    for t in tasks:
        confidence = f"{t['confidence']:.2f}" if t["confidence"] is not None else "-"
        task_rows.append(
            f"""<div class="row">
            <strong>{esc(t['title'])}</strong>
            <div class="small muted">{esc(t['phase'])} | {esc(t['status'])} | confidence {confidence}</div>
            {f"<pre>{esc(t['output'])}</pre>" if t['output'] else ""}
            {f"<div class='small bad'>{esc(t['error_type'])}: {esc(t['error_message'])}</div>" if t['error_type'] else ""}
            </div>"""
        )

    event_rows = [
        f"<div class='row small'><strong>{esc(e['kind'])}</strong> {esc(e['message'])}<div class='muted'>{esc(e['created_at'])}</div></div>"
        for e in events
    ]

    evidence_rows = [
        f"<div class='row'><strong>{esc(e['source_title'] or e['source_url'])}</strong><br>"
        f"<a href='{esc(e['source_url'])}' target='_blank' rel='noopener'>{esc(e['source_url'])}</a></div>"
        for e in evidence
    ]

    retry = ""
    if goal["status"] in ("incomplete", "failed", "interrupted"):
        retry = '<form method="post" action="/goals/%s/retry"><button>Retry goal</button></form>' % goal_id

    body = f"""
    <p><a href="/">Back to dashboard</a></p>
    <div class="card">
      <h1>{esc(goal['title'])}</h1>
      <div><strong>Version:</strong> {APP_VERSION} &nbsp; <strong>Status:</strong> {esc(goal['status'])}</div>
      <div><strong>Budget:</strong> ${goal['budget']:.2f} &nbsp; <strong>Run:</strong> {esc(run['run_number'] if run else '-')}</div>
      <br>{retry}
    </div>

    <div class="card"><h2>CEO plan</h2>
      <pre>{esc(goal['plan_json'] or 'Planning...')}</pre>
    </div>

    <div class="card"><h2>Task execution</h2>
      {''.join(task_rows) or '<p class="muted">No tasks yet.</p>'}
    </div>

    <div class="card"><h2>Evaluator</h2>
      {('<pre>'+esc(json.dumps(dict(evaluation), indent=2))+'</pre>') if evaluation else '<p class="muted">No evaluation yet.</p>'}
    </div>

    <div class="card"><h2>Evidence / sources</h2>
      {''.join(evidence_rows) or '<p class="muted">No captured sources.</p>'}
    </div>

    <div class="card"><h2>Final output</h2>
      <pre>{esc(goal['final_output'] or 'Waiting for workforce execution...')}</pre>
    </div>

    <div class="card"><h2>Activity</h2>
      {''.join(event_rows) or '<p class="muted">No activity yet.</p>'}
    </div>
    """
    return page_shell(goal["title"], body)


init_db()
