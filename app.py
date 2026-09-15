import os, sqlite3, uuid, json, html, time, hmac, hashlib, threading
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

APP_VERSION = "0.3.5"
SCHEMA_VERSION = "035-6"
DB = Path(__file__).with_name("workforce_v035.db")
MAX_TASKS = int(os.getenv("MAX_TASKS_PER_GOAL", "10"))
MAX_AGENTS = int(os.getenv("MAX_AGENT_INSTANCES_PER_GOAL", "8"))
MAX_TASK_WORKERS = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))
MAX_GOAL_WORKERS = int(os.getenv("MAX_CONCURRENT_GOALS", "1"))
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "2"))
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "75"))
ENABLE_WEB_RESEARCH = os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true"
APP_ACCESS_TOKEN = os.getenv("APP_ACCESS_TOKEN", "").strip()
MODEL_PRICES = {"gpt-5-mini": (0.25, 2.00), "gpt-5.6-luna": (0.20, 1.20)}
DEFAULT_INPUT_PRICE = float(os.getenv("OPENAI_INPUT_PRICE_PER_MTOK", "0.25"))
DEFAULT_OUTPUT_PRICE = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_MTOK", "2.0"))
DEFAULT_GOAL_BUDGET = float(os.getenv("DEFAULT_GOAL_BUDGET_USD", "5.0"))

app = FastAPI(title="AI Workforce OS", version=APP_VERSION)
goal_executor = ThreadPoolExecutor(max_workers=MAX_GOAL_WORKERS)
task_executor = ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS)
future_lock = threading.Lock()
goal_futures = {}

# Minimal tool registry. v0.3.5 exposes capability checks without yet building a plugin marketplace.
TOOL_REGISTRY = {
    "web_search": {"requires": "web_research", "description": "External web research through the model Responses API."},
}

def tool_allowed(agent, tool_name):
    spec = TOOL_REGISTRY.get(tool_name)
    if not spec:
        return False
    try:
        caps = json.loads(agent["capabilities_json"] or "[]")
    except Exception:
        caps = []
    return spec["requires"] in caps


def now(): return datetime.now(timezone.utc).isoformat()
def uid(): return str(uuid.uuid4())
def esc(x): return html.escape(str(x or ""))

def db():
    c = sqlite3.connect(DB, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("PRAGMA journal_mode=WAL")
    return c

def fetch(sql, args=()):
    c=db()
    try: return c.execute(sql,args).fetchall()
    finally: c.close()

def fetch_one(sql,args=()):
    rows=fetch(sql,args); return rows[0] if rows else None

def write(sql,args=()):
    c=db()
    try: c.execute(sql,args); c.commit()
    finally: c.close()

def log_event(company_id=None,run_id=None,goal_id=None,task_id=None,kind="info",message="",payload=None):
    write("INSERT INTO events(id,company_id,run_id,goal_id,task_id,kind,message,payload_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
          (uid(),company_id,run_id,goal_id,task_id,kind,message,json.dumps(payload or {},ensure_ascii=False),now()))


def init():
    c=db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS companies(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS agent_profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL,role TEXT NOT NULL,instructions TEXT NOT NULL,capabilities_json TEXT NOT NULL,skills_json TEXT NOT NULL,model_policy_json TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY,company_id TEXT NOT NULL,title TEXT NOT NULL,description TEXT NOT NULL,criteria TEXT NOT NULL,budget REAL NOT NULL,spent REAL NOT NULL DEFAULT 0,status TEXT NOT NULL,current_run_id TEXT,plan_json TEXT,verification_plan_json TEXT,final_output TEXT,replan_count INTEGER NOT NULL DEFAULT 0,max_replans INTEGER NOT NULL DEFAULT 2,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,goal_id TEXT NOT NULL,parent_run_id TEXT,run_number INTEGER NOT NULL,status TEXT NOT NULL,reason TEXT,started_at TEXT,ended_at TEXT,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS agent_instances(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,profile_id TEXT NOT NULL,name TEXT NOT NULL,instructions_override TEXT,status TEXT NOT NULL,budget REAL NOT NULL DEFAULT 0,spend REAL NOT NULL DEFAULT 0,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,plan_task_id TEXT NOT NULL,agent_instance_id TEXT NOT NULL,title TEXT NOT NULL,instructions TEXT NOT NULL,contract_json TEXT NOT NULL,status TEXT NOT NULL,required INTEGER NOT NULL DEFAULT 1,output TEXT,structured_output_json TEXT,confidence REAL,budget_limit REAL NOT NULL DEFAULT 0,spent REAL NOT NULL DEFAULT 0,attempt_count INTEGER NOT NULL DEFAULT 0,max_attempts INTEGER NOT NULL DEFAULT 2,error_type TEXT,error_message TEXT,phase TEXT NOT NULL DEFAULT 'work',created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS task_dependencies(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,upstream_task_id TEXT NOT NULL,downstream_task_id TEXT NOT NULL,condition TEXT NOT NULL,required INTEGER NOT NULL DEFAULT 1,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS task_attempts(id TEXT PRIMARY KEY,task_id TEXT NOT NULL,attempt_number INTEGER NOT NULL,status TEXT NOT NULL,model TEXT,started_at TEXT,ended_at TEXT,latency_ms INTEGER,input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,cost REAL NOT NULL DEFAULT 0,error_type TEXT,error_message TEXT,strategy_json TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS model_calls(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,agent_instance_id TEXT,purpose TEXT NOT NULL,model TEXT NOT NULL,tool_mode TEXT,started_at TEXT,ended_at TEXT,latency_ms INTEGER,input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,cost REAL NOT NULL DEFAULT 0,request_id TEXT,status TEXT NOT NULL,error_type TEXT,error_message TEXT,metadata_json TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS budget_reservations(id TEXT PRIMARY KEY,goal_id TEXT NOT NULL,task_id TEXT,amount REAL NOT NULL,status TEXT NOT NULL,model_call_id TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS budget_ledger(id TEXT PRIMARY KEY,company_id TEXT NOT NULL,goal_id TEXT,task_id TEXT,agent_instance_id TEXT,model_call_id TEXT,amount REAL NOT NULL,currency TEXT NOT NULL DEFAULT 'USD',kind TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS handoffs(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,from_task_id TEXT NOT NULL,to_task_id TEXT NOT NULL,summary TEXT NOT NULL,evidence_ids_json TEXT NOT NULL,artifact_ids_json TEXT NOT NULL,assumptions_json TEXT NOT NULL,unknowns_json TEXT NOT NULL,confidence REAL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,task_id TEXT,claim TEXT NOT NULL,source_type TEXT NOT NULL,source_title TEXT,source_url TEXT,source_publisher TEXT,published_at TEXT,retrieved_at TEXT NOT NULL,snippet TEXT,evidence_state TEXT NOT NULL,confidence REAL,freshness_days INTEGER,metadata_json TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,task_id TEXT NOT NULL,name TEXT NOT NULL,artifact_type TEXT NOT NULL,version INTEGER NOT NULL,content TEXT,path TEXT,mime_type TEXT,content_hash TEXT,status TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS artifact_dependencies(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,upstream_artifact_id TEXT NOT NULL,downstream_task_id TEXT NOT NULL,relation TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,stage TEXT NOT NULL,score REAL,passed INTEGER NOT NULL,checks_json TEXT NOT NULL,failures_json TEXT NOT NULL,recommendations_json TEXT NOT NULL,model TEXT,input_tokens INTEGER NOT NULL DEFAULT 0,output_tokens INTEGER NOT NULL DEFAULT 0,cost REAL NOT NULL DEFAULT 0,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS memory_items(id TEXT PRIMARY KEY,company_id TEXT NOT NULL,goal_id TEXT,task_id TEXT,agent_instance_id TEXT,memory_type TEXT NOT NULL,key TEXT NOT NULL,value TEXT NOT NULL,source_evidence_ids_json TEXT NOT NULL,confidence REAL,freshness_days INTEGER,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS contradictions(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,claim_a TEXT NOT NULL,claim_b TEXT NOT NULL,evidence_a_json TEXT NOT NULL,evidence_b_json TEXT NOT NULL,severity TEXT NOT NULL,status TEXT NOT NULL,resolution TEXT,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,company_id TEXT,run_id TEXT,goal_id TEXT,task_id TEXT,kind TEXT NOT NULL,message TEXT NOT NULL,payload_json TEXT NOT NULL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS benchmark_runs(id TEXT PRIMARY KEY,app_version TEXT NOT NULL,goal_id TEXT,run_id TEXT,case_name TEXT NOT NULL,expected_version TEXT NOT NULL,version_match INTEGER NOT NULL,started_at TEXT NOT NULL,ended_at TEXT,outcome TEXT,spend_delta REAL NOT NULL DEFAULT 0,notes TEXT);
    CREATE INDEX IF NOT EXISTS idx_tasks_run ON tasks(run_id); CREATE INDEX IF NOT EXISTS idx_events_goal ON events(goal_id,created_at); CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id); CREATE INDEX IF NOT EXISTS idx_model_calls_run ON model_calls(run_id);
    """)
    c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('app_version',?)",(APP_VERSION,))
    c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",(SCHEMA_VERSION,))
    if not c.execute("SELECT id FROM companies LIMIT 1").fetchone():
        cid=uid(); t=now(); c.execute("INSERT INTO companies VALUES(?,?,?,?)",(cid,"My AI Company",t,t))
        seed=[
          ("CEO","ceo","Plan objectives, detect capability gaps, delegate work, define verification.",["planning","delegation","orchestration"]),
          ("Research Specialist","research","Research facts, evidence, market context, assumptions and unknowns.",["web_research","source_verification","market_research"]),
          ("Engineering Analyst","engineering","Analyze technical requirements, feasibility, architecture and risks.",["technical_analysis","architecture","risk"]),
          ("Procurement Analyst","procurement","Analyze vendors, tooling, sourcing, costs and operational constraints.",["procurement","vendors","commercial"]),
          ("Product & UX Specialist","product","Analyze customers, workflows, adoption, product scope and UX risks.",["customer","product","ux"]),
          ("Data Analyst","data","Perform quantitative reasoning, metrics, calculations and sanity checks.",["data","metrics","quantitative"]),
          ("Quality & Risk Reviewer","qa","Challenge claims, contradictions, unsupported assertions and omissions.",["qa","risk","verification"]),
          ("Executive Report Writer","report","Synthesize verified findings into a decision-ready deliverable.",["synthesis","report","decision_support"])]
        for name,role,instr,caps in seed: c.execute("INSERT INTO agent_profiles VALUES(?,?,?,?,?,?,?,?,?)",(uid(),name,role,instr,json.dumps(caps),"[]","{}",t,t))
    c.commit(); c.close()


def company(): return fetch_one("SELECT * FROM companies ORDER BY created_at LIMIT 1")
def get_goal(gid): return fetch_one("SELECT * FROM goals WHERE id=?",(gid,))
def get_goal_from_run(rid): return fetch_one("SELECT g.* FROM goals g JOIN runs r ON r.goal_id=g.id WHERE r.id=?",(rid,))
def profile_by_role(role): return fetch_one("SELECT * FROM agent_profiles WHERE role=? ORDER BY created_at LIMIT 1",(role,))


def require_auth(request):
    if not APP_ACCESS_TOKEN: return None
    supplied=request.headers.get("Authorization","")
    cookie=request.cookies.get("wf_auth","")
    expected=hashlib.sha256(APP_ACCESS_TOKEN.encode()).hexdigest()
    if (supplied.startswith("Bearer ") and hmac.compare_digest(supplied[7:],APP_ACCESS_TOKEN)) or hmac.compare_digest(cookie,expected): return None
    return HTMLResponse("Unauthorized. Use /login.",status_code=401)

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request,call_next):
        if APP_ACCESS_TOKEN and request.url.path not in {"/health","/login"}: 
            denied=require_auth(request)
            if denied:return denied
        return await call_next(request)
app.add_middleware(AuthMiddleware)


def model_name_for(purpose, agent=None):
    specific = {"planner": os.getenv("OPENAI_PLANNER_MODEL", ""), "evaluator": os.getenv("OPENAI_EVALUATOR_MODEL", ""), "report": os.getenv("OPENAI_REPORT_MODEL", "")}.get(purpose, "")
    if specific: return specific
    if purpose == "task" and agent is not None:
        try:
            policy = json.loads(agent["model_policy_json"] or "{}") if "model_policy_json" in agent.keys() else {}
            if policy.get("model"): return policy["model"]
        except Exception: pass
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")
def price_for(model): return MODEL_PRICES.get(model,(DEFAULT_INPUT_PRICE,DEFAULT_OUTPUT_PRICE))
def calc_cost(model,inp,out):
    ip,op=price_for(model); return inp/1_000_000*ip+out/1_000_000*op

def classify_error(exc):
    n=type(exc).__name__.lower(); t=str(exc).lower()
    if "timeout" in n or "timed out" in t:return "transient_timeout"
    if "ratelimit" in n or "429" in t or "rate limit" in t:return "transient_rate_limit"
    if any(x in n for x in ("connection","apierror","internalserver")) or any(x in t for x in ("502","503","504","connection reset","temporarily unavailable")):return "transient_provider"
    if "auth" in n or "401" in t or "403" in t:return "permanent_auth"
    if "400" in t or "bad request" in t:return "permanent_request"
    return "unknown"

def usage_counts(resp):
    u=getattr(resp,"usage",None)
    return (int(getattr(u,"input_tokens",0) or 0),int(getattr(u,"output_tokens",0) or 0)) if u else (0,0)
def response_id(resp): return str(getattr(resp,"id","") or "")


def parse_structured_text(text):
    """Parse structured JSON returned by the Responses API.

    Accepts plain JSON or JSON wrapped in a markdown fence. Raises a
    ValueError with a useful message when the payload is not valid JSON.
    """
    raw=(text or "").strip()
    if not raw:
        raise ValueError("Structured response was empty")
    if raw.startswith("```"):
        lines=raw.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines=lines[1:]
        if lines and lines[-1].strip()=="```":
            lines=lines[:-1]
        raw="\n".join(lines).strip()
        if raw.lower().startswith("json\n"):
            raw=raw[5:].lstrip()
    try:
        value=json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Structured JSON parse failed: {exc}") from exc
    if not isinstance(value,dict):
        raise ValueError(f"Structured response must be a JSON object, got {type(value).__name__}")
    return value

def citations(resp):
    out=[]
    try:
        for item in getattr(resp,"output",[]) or []:
            for part in getattr(item,"content",[]) or []:
                for a in getattr(part,"annotations",[]) or []:
                    if getattr(a,"type","")=="url_citation" and getattr(a,"url",None): out.append({"title":str(getattr(a,"title","") or ""),"url":str(a.url),"publisher":"","published_at":""})
    except Exception: pass
    seen=set(); result=[]
    for x in out:
        if x["url"] not in seen: seen.add(x["url"]); result.append(x)
    return result


def reserve_budget(goal_id,task_id,estimated):
    c=db()
    try:
        c.execute("BEGIN IMMEDIATE")
        g=c.execute("SELECT budget,spent FROM goals WHERE id=?",(goal_id,)).fetchone()
        t=c.execute("SELECT budget_limit,spent FROM tasks WHERE id=?",(task_id,)).fetchone() if task_id else None
        if not g: raise RuntimeError("Goal not found")
        goal_reserved=float(c.execute("SELECT COALESCE(SUM(amount),0) FROM budget_reservations WHERE goal_id=? AND status='reserved'",(goal_id,)).fetchone()[0] or 0)
        task_reserved=float(c.execute("SELECT COALESCE(SUM(amount),0) FROM budget_reservations WHERE task_id=? AND status='reserved'",(task_id,)).fetchone()[0] or 0) if task_id else 0
        goal_remaining=max(0,float(g[0])-float(g[1])-goal_reserved)
        task_remaining=max(0,float(t[0])-float(t[1])-task_reserved) if t and float(t[0])>0 else goal_remaining
        limit=min(goal_remaining,task_remaining)
        if estimated>limit+1e-9: raise RuntimeError(f"Budget guard blocked call: estimated ${estimated:.6f}, remaining ${limit:.6f}")
        rid=uid(); c.execute("INSERT INTO budget_reservations VALUES(?,?,?,?,?,?,?,?)",(rid,goal_id,task_id,estimated,"reserved",None,now(),now())); c.commit(); return rid
    except Exception:
        c.rollback(); raise
    finally:c.close()

def settle_budget(reservation_id,actual,goal_id,task_id,agent_id,call_id):
    c=db()
    try:
        c.execute("BEGIN IMMEDIATE")
        r=c.execute("SELECT amount,status FROM budget_reservations WHERE id=?",(reservation_id,)).fetchone()
        if not r or r[1]!="reserved": c.rollback(); return
        g=c.execute("SELECT budget,spent,company_id FROM goals WHERE id=?",(goal_id,)).fetchone()
        # Reservation is a guard, not a measurement. Record actual usage; if it exceeds the
        # reservation, the ledger captures the true spend and future reservations see the higher spend.
        actual=float(actual)
        c.execute("UPDATE budget_reservations SET status='settled',model_call_id=?,updated_at=? WHERE id=?",(call_id,now(),reservation_id))
        c.execute("INSERT INTO budget_ledger VALUES(?,?,?,?,?,?,?,?,?,?)",(uid(),g[2],goal_id,task_id,agent_id,call_id,actual,"USD","model_call",now()))
        c.execute("UPDATE goals SET spent=spent+?,updated_at=? WHERE id=?",(actual,now(),goal_id))
        if task_id:c.execute("UPDATE tasks SET spent=spent+?,updated_at=? WHERE id=?",(actual,now(),task_id))
        if agent_id:c.execute("UPDATE agent_instances SET spend=spend+?,updated_at=? WHERE id=?",(actual,now(),agent_id))
        c.commit()
    except Exception:c.rollback();raise
    finally:c.close()


def call_model(run_id,task_id,agent_id,system,prompt,*,purpose,use_web=False,structured_schema=None,max_output_tokens=1600,max_attempts=2):
    key=os.getenv("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY is not configured")
    from openai import OpenAI
    routing_agent = fetch_one("SELECT p.model_policy_json FROM agent_instances ai JOIN agent_profiles p ON p.id=ai.profile_id WHERE ai.id=?", (agent_id,)) if agent_id else None
    model=model_name_for(purpose, routing_agent)
    web=bool(use_web and ENABLE_WEB_RESEARCH)
    goal=get_goal_from_run(run_id) if run_id else None
    estimated_input=max(1,(len(system)+len(prompt))//4)
    estimated=calc_cost(model,estimated_input,max_output_tokens)
    last=None
    for attempt in range(1,max_attempts+1):
        reservation=reserve_budget(goal["id"],task_id,estimated) if goal else None
        started=time.time(); started_at_iso=now(); call_id=uid(); tool_mode="web" if web else "none"
        meta={"attempt":attempt,"structured":bool(structured_schema),"input_chars":len(system)+len(prompt),"estimated_input_tokens":estimated_input}
        write("INSERT INTO model_calls(id,run_id,task_id,agent_instance_id,purpose,model,tool_mode,started_at,ended_at,latency_ms,input_tokens,output_tokens,cost,request_id,status,error_type,error_message,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (call_id,run_id,task_id,agent_id,purpose,model,tool_mode,now(),None,None,0,0,0,None,"running",None,None,json.dumps(meta),now()))
        try:
            client=OpenAI(api_key=key,timeout=OPENAI_TIMEOUT,max_retries=0)
            kwargs={"model":model,"input":f"SYSTEM:\n{system}\n\nUSER:\n{prompt}","max_output_tokens":max_output_tokens}
            if web: kwargs["tools"]=[{"type":"web_search"}]
            if structured_schema: kwargs["text"]={"format":{"type":"json_schema","name":structured_schema["name"],"strict":True,"schema":structured_schema["schema"]}}
            resp=client.responses.create(**kwargs)
            text=(getattr(resp,"output_text","") or "").strip()
            if not text: raise RuntimeError("Model returned empty output")
            inp,out=usage_counts(resp); cost=calc_cost(model,inp,out); lat=int((time.time()-started)*1000); req=response_id(resp); cites=citations(resp)
            meta.update({"citations":cites,"request_id":req})
            # Parse structured output before declaring the call successful. The model may have
            # completed while the application-level contract validation failed.
            try:
                structured = parse_structured_text(text) if structured_schema else None
            except Exception as parse_exc:
                meta.update({"validation_error":str(parse_exc)})
                write("UPDATE model_calls SET ended_at=?,latency_ms=?,input_tokens=?,output_tokens=?,cost=?,request_id=?,status='failed',error_type='structured_output_validation',error_message=?,metadata_json=? WHERE id=?",(now(),lat,inp,out,cost,req,str(parse_exc),json.dumps(meta),call_id))
                if reservation:
                    settle_budget(reservation,cost,goal["id"],task_id,agent_id,call_id); reservation=None
                log_event(run_id=run_id,goal_id=goal["id"] if goal else None,task_id=task_id,kind="model_call_validation_error",message=f"{purpose} attempt {attempt}: structured output validation failed",payload={"model":model,"latency_ms":lat,"cost":cost,"request_id":req,"error":str(parse_exc)})
                last=parse_exc
                if attempt < max_attempts:
                    time.sleep(min(2**(attempt-1),4)); continue
                break
            write("UPDATE model_calls SET ended_at=?,latency_ms=?,input_tokens=?,output_tokens=?,cost=?,request_id=?,status='completed',metadata_json=? WHERE id=?",(now(),lat,inp,out,cost,req,json.dumps(meta),call_id))
            if reservation: settle_budget(reservation,cost,goal["id"],task_id,agent_id,call_id); reservation=None
            log_event(run_id=run_id,goal_id=goal["id"] if goal else None,task_id=task_id,kind="model_call_ok",message=f"{purpose} attempt {attempt} completed",payload={"model":model,"latency_ms":lat,"input_tokens":inp,"output_tokens":out,"cost":cost,"request_id":req,"web":web,"structured_validated":bool(structured_schema)})
            return {"text":text,"structured":structured,"citations":cites,"input_tokens":inp,"output_tokens":out,"cost":cost,"latency_ms":lat,"request_id":req,"model":model,"started_at":started_at_iso,"ended_at":now()}
        except Exception as exc:
            cls=classify_error(exc); lat=int((time.time()-started)*1000); last=exc
            write("UPDATE model_calls SET ended_at=?,latency_ms=?,status='failed',error_type=?,error_message=?,metadata_json=? WHERE id=?",(now(),lat,cls,str(exc),json.dumps(meta),call_id))
            log_event(run_id=run_id,goal_id=goal["id"] if goal else None,task_id=task_id,kind="model_call_error",message=f"{purpose} attempt {attempt}: {cls}",payload={"model":model,"latency_ms":lat,"error":str(exc),"web":web})
            if reservation:
                write("UPDATE budget_reservations SET status='released',updated_at=? WHERE id=?",(now(),reservation))
                reservation=None
            if web and attempt==1:
                web=False; log_event(run_id=run_id,goal_id=goal["id"] if goal else None,task_id=task_id,kind="tool_fallback",message="Retrying without web search.")
                continue
            if cls.startswith("transient") and attempt<max_attempts:
                time.sleep(min(2**(attempt-1),4)); continue
            break
    raise last or RuntimeError("Model call failed")


def plan_schema():
    contract = {
        "type": "object", "additionalProperties": False,
        "required": ["inputs", "outputs", "success_conditions", "failure_conditions", "evidence_required", "requires_web", "notes"],
        "properties": {
            "inputs": {"type": "array", "items": {"type": "string"}},
            "outputs": {"type": "array", "items": {"type": "string"}},
            "success_conditions": {"type": "array", "items": {"type": "string"}},
            "failure_conditions": {"type": "array", "items": {"type": "string"}},
            "evidence_required": {"type": "boolean"},
            "requires_web": {"type": "boolean"},
            "notes": {"type": "string"}
        }
    }
    task = {
        "type": "object", "additionalProperties": False,
        "required": ["id", "title", "instructions", "agent_role", "depends_on", "dependency_conditions", "required", "requires_web", "budget_limit", "max_attempts", "phase", "contract"],
        "properties": {
            "id": {"type": "string"}, "title": {"type": "string"}, "instructions": {"type": "string"},
            "agent_role": {"type": "string"}, "depends_on": {"type": "array", "items": {"type": "string"}},
            "dependency_conditions": {"type": "array", "items": {"type": "string"}},
            "required": {"type": "boolean"}, "requires_web": {"type": "boolean"},
            "budget_limit": {"type": "number"}, "max_attempts": {"type": "integer"}, "phase": {"type": "string"},
            "contract": contract
        }
    }
    agent = {
        "type": "object", "additionalProperties": False,
        "required": ["name", "role", "instructions", "capabilities"],
        "properties": {
            "name": {"type": "string"}, "role": {"type": "string"}, "instructions": {"type": "string"},
            "capabilities": {"type": "array", "items": {"type": "string"}}
        }
    }
    verification = {
        "type": "object", "additionalProperties": False, "required": ["required_checks", "pass_threshold"],
        "properties": {"required_checks": {"type": "array", "items": {"type": "string"}}, "pass_threshold": {"type": "number"}}
    }
    return {"name": "workforce_plan", "schema": {
        "type": "object", "additionalProperties": False, "required": ["agents", "tasks", "verification"],
        "properties": {"agents": {"type": "array", "items": agent}, "tasks": {"type": "array", "items": task}, "verification": verification}
    }}


def task_output_schema():
    return {"name":"workforce_task_output","schema":{"type":"object","additionalProperties":False,"required":["summary","facts","assumptions","unknowns","claims","recommendations"],"properties":{
      "summary":{"type":"string"},"facts":{"type":"array","items":{"type":"string"}},"assumptions":{"type":"array","items":{"type":"string"}},"unknowns":{"type":"array","items":{"type":"string"}},"claims":{"type":"array","items":{"type":"string"}},"recommendations":{"type":"array","items":{"type":"string"}}}}}

def evaluator_schema():
    return {"name":"workforce_evaluation","schema":{"type":"object","additionalProperties":False,"required":["passed","score","checks","failures","replan_tasks","contradictions"],"properties":{
      "passed":{"type":"boolean"},"score":{"type":"number"},"checks":{"type":"array","items":{"type":"object","additionalProperties":False,"required":["name","score","passed","reason"],"properties":{"name":{"type":"string"},"score":{"type":"number"},"passed":{"type":"boolean"},"reason":{"type":"string"}}}},
      "failures":{"type":"array","items":{"type":"string"}},"replan_tasks":{"type":"array","items":{"type":"object","additionalProperties":False,"required":["title","instructions","agent_role","depends_on","requires_web","budget_limit","success_conditions"],"properties":{"title":{"type":"string"},"instructions":{"type":"string"},"agent_role":{"type":"string"},"depends_on":{"type":"array","items":{"type":"string"}},"requires_web":{"type":"boolean"},"budget_limit":{"type":"number"},"success_conditions":{"type":"array","items":{"type":"string"}}}}},
      "contradictions":{"type":"array","items":{"type":"object","additionalProperties":False,"required":["claim_a","claim_b","severity","resolution_needed"],"properties":{"claim_a":{"type":"string"},"claim_b":{"type":"string"},"severity":{"type":"string"},"resolution_needed":{"type":"boolean"}}}}}}}


def _gval(g, key, default=""):
    if isinstance(g, dict):
        return g.get(key, default)
    try:
        return g[key]
    except Exception:
        return default


def classify_objective(g):
    text = f"{_gval(g,'title')} {_gval(g,'description')} {_gval(g,'criteria')}".lower()
    compute_terms = ("calculate", "calculation", "break-even", "break even", "revenue", "profit", "margin", "arithmetic", "math", "numerical")
    research_terms = ("research", "market", "competitor", "industry", "sources", "evidence", "pricing", "customer discovery")
    action_terms = ("send", "email", "deploy", "purchase", "publish", "contact", "execute")
    create_terms = ("write", "draft", "design", "build", "create", "develop", "prototype")
    has_compute = any(t in text for t in compute_terms)
    has_research = any(t in text for t in research_terms)
    has_action = any(t in text for t in action_terms)
    has_create = any(t in text for t in create_terms)
    kinds = [name for name, flag in (("compute",has_compute),("research",has_research),("action",has_action),("create",has_create)) if flag]
    return "mixed" if len(kinds) > 1 else (kinds[0] if kinds else "reason")


def minimal_safe_plan(g):
    kind = classify_objective(g)
    if kind == "compute":
        agents = [
            {"name":"Quantitative Analyst","role":"data","instructions":"Perform deterministic calculations and sanity checks. Do not browse unless explicitly required.","capabilities":["data","metrics","calculation"]},
            {"name":"Quality Reviewer","role":"qa","instructions":"Verify arithmetic, assumptions and whether the success criteria are satisfied.","capabilities":["qa","risk","verification"]},
            {"name":"Executive Report Writer","role":"report","instructions":"Present verified results clearly and concisely.","capabilities":["synthesis","report"]},
        ]
        tasks = [
            {"id":"T1","title":"Quantitative analysis","instructions":"Solve the stated quantitative problem using only the provided information. Show arithmetic and identify assumptions.","agent_role":"data","depends_on":[],"dependency_conditions":[],"required":True,"requires_web":False,"budget_limit":0.45,"max_attempts":2,"phase":"work","contract":{"inputs":[],"outputs":["calculations","assumptions"],"success_conditions":["calculations are internally consistent","all requested metrics are answered"],"failure_conditions":["insufficient information"],"evidence_required":False,"requires_web":False,"notes":"Deterministic calculation; no web needed."}},
            {"id":"T2","title":"Quality verification","instructions":"Check the quantitative analysis for arithmetic errors, missing requested outputs and unsupported assumptions.","agent_role":"qa","depends_on":["T1"],"dependency_conditions":["completed"],"required":True,"requires_web":False,"budget_limit":0.25,"max_attempts":2,"phase":"work","contract":{"inputs":["T1"],"outputs":["verification"],"success_conditions":["critical errors identified or none found"],"failure_conditions":["missing analysis"],"evidence_required":False,"requires_web":False,"notes":"Independent verification."}},
            {"id":"T3","title":"Executive answer","instructions":"Present the verified calculation, assumptions and conclusion in a concise answer.","agent_role":"report","depends_on":["T1","T2"],"dependency_conditions":["completed","completed"],"required":True,"requires_web":False,"budget_limit":0.30,"max_attempts":2,"phase":"report","contract":{"inputs":["T1","T2"],"outputs":["decision_ready_answer"],"success_conditions":["all requested metrics addressed"],"failure_conditions":["verification failed"],"evidence_required":False,"requires_web":False,"notes":"No external research required."}},
        ]
    elif kind == "action":
        agents=[{"name":"Action Planner","role":"operator","instructions":"Translate the request into safe executable steps and identify approvals or missing permissions.","capabilities":["planning","operations"]},{"name":"Quality Reviewer","role":"qa","instructions":"Check safety, completeness and permissions before action.","capabilities":["qa","risk","verification"]},{"name":"Executive Report Writer","role":"report","instructions":"Summarize the action plan and results.","capabilities":["synthesis","report"]}]
        tasks=[
            {"id":"T1","title":"Action planning","instructions":"Define the minimal safe action sequence and identify required permissions or human approvals.","agent_role":"operator","depends_on":[],"dependency_conditions":[],"required":True,"requires_web":False,"budget_limit":0.45,"max_attempts":2,"phase":"work","contract":{"inputs":[],"outputs":["action_plan"],"success_conditions":["action plan is complete"],"failure_conditions":["required permissions unknown"],"evidence_required":False,"requires_web":False,"notes":"No external action is performed in v0.3.5."}},
            {"id":"T2","title":"Quality verification","instructions":"Review the action plan for missing constraints, unsafe steps or approval requirements.","agent_role":"qa","depends_on":["T1"],"dependency_conditions":["completed"],"required":True,"requires_web":False,"budget_limit":0.25,"max_attempts":2,"phase":"work","contract":{"inputs":["T1"],"outputs":["verification"],"success_conditions":["critical issues identified or none found"],"failure_conditions":["missing plan"],"evidence_required":False,"requires_web":False,"notes":"No action without explicit tool permissions."}},
            {"id":"T3","title":"Executive answer","instructions":"Summarize the verified action plan and required approvals.","agent_role":"report","depends_on":["T1","T2"],"dependency_conditions":["completed","completed"],"required":True,"requires_web":False,"budget_limit":0.30,"max_attempts":2,"phase":"report","contract":{"inputs":["T1","T2"],"outputs":["decision_ready_answer"],"success_conditions":["all requested items addressed"],"failure_conditions":["verification failed"],"evidence_required":False,"requires_web":False,"notes":"Planning only; no external action."}},
        ]
    else:
        agents=[{"name":"Generalist Analyst","role":"generalist","instructions":"Analyze the objective, make explicit assumptions and identify unknowns. Do not browse unless the task truly needs external evidence.","capabilities":["reasoning","analysis"]},{"name":"Quality Reviewer","role":"qa","instructions":"Challenge omissions, contradictions and unsupported claims.","capabilities":["qa","risk","verification"]},{"name":"Executive Report Writer","role":"report","instructions":"Synthesize verified work into a concise answer.","capabilities":["synthesis","report"]}]
        tasks=[
            {"id":"T1","title":"Objective analysis","instructions":"Analyze the objective, produce the most useful initial answer, and clearly distinguish facts, assumptions and unknowns.","agent_role":"generalist","depends_on":[],"dependency_conditions":[],"required":True,"requires_web":False,"budget_limit":0.45,"max_attempts":2,"phase":"work","contract":{"inputs":[],"outputs":["analysis","assumptions","unknowns"],"success_conditions":["objective addressed"],"failure_conditions":["insufficient information"],"evidence_required":False,"requires_web":False,"notes":"Only use web if later planner logic determines it is genuinely necessary."}},
            {"id":"T2","title":"Quality verification","instructions":"Check the initial analysis for omissions, contradictions, unsupported claims and failure to follow the success criteria.","agent_role":"qa","depends_on":["T1"],"dependency_conditions":["completed"],"required":True,"requires_web":False,"budget_limit":0.25,"max_attempts":2,"phase":"work","contract":{"inputs":["T1"],"outputs":["verification"],"success_conditions":["critical issues identified or none found"],"failure_conditions":["missing context"],"evidence_required":False,"requires_web":False,"notes":"Independent verification."}},
            {"id":"T3","title":"Executive answer","instructions":"Present the verified answer and clearly state assumptions, unknowns and next actions.","agent_role":"report","depends_on":["T1","T2"],"dependency_conditions":["completed","completed"],"required":True,"requires_web":False,"budget_limit":0.30,"max_attempts":2,"phase":"report","contract":{"inputs":["T1","T2"],"outputs":["decision_ready_answer"],"success_conditions":["all requested items addressed"],"failure_conditions":["verification failed"],"evidence_required":False,"requires_web":False,"notes":"No external research unless explicitly required."}},
        ]
    return {"agents":agents,"tasks":tasks,"verification":{"required_checks":["all_success_criteria_addressed","contradictions_checked","assumptions_labeled","budget_respected","actionable"],"pass_threshold":0.80}}


def default_plan(g):
    return minimal_safe_plan(g)


def normalize_plan(plan,g):
    if not isinstance(plan,dict) or not isinstance(plan.get("tasks"),list) or not plan["tasks"]:
        plan=minimal_safe_plan(g)
    tasks=plan["tasks"][:MAX_TASKS]
    raw_ids=[str(t.get("id")) for t in tasks]
    out=[]; seen=set(); id_alias={}
    for i,t in enumerate(tasks,1):
        t=dict(t) if isinstance(t,dict) else {}
        original=str(t.get("id") or f"T{i}")
        tid=original if original not in seen else f"T{i}"
        while tid in seen: tid=f"T{i}_{len(seen)}"
        seen.add(tid); id_alias[original]=tid
        t["id"]=tid
        t["agent_role"]=str(t.get("agent_role") or "generalist").strip().lower()
        deps=[id_alias.get(str(x), str(x)) for x in (t.get("depends_on") or [])]
        # Only keep dependencies that refer to a real task in the input plan; normalize after IDs exist.
        t["_raw_deps"]=deps
        cond=list(t.get("dependency_conditions") or [])
        t["dependency_conditions"]=cond
        t["required"]=bool(t.get("required",False))
        t["requires_web"]=bool(t.get("requires_web",False))
        t["budget_limit"]=max(0,float(t.get("budget_limit",0) or 0))
        t["max_attempts"]=max(1,min(3,int(t.get("max_attempts",2) or 2)))
        t["phase"]="report" if str(t.get("phase","work")).lower()=="report" else "work"
        c=t.get("contract") if isinstance(t.get("contract"),dict) else {}
        t["contract"]={
            "inputs":[str(x) for x in c.get("inputs",[])],
            "outputs":[str(x) for x in c.get("outputs",[])],
            "success_conditions":[str(x) for x in c.get("success_conditions",[])],
            "failure_conditions":[str(x) for x in c.get("failure_conditions",[])],
            "evidence_required":bool(c.get("evidence_required",False)),
            "requires_web":t["requires_web"],
            "notes":str(c.get("notes",""))
        }
        out.append(t)
    valid_ids={t["id"] for t in out}
    for t in out:
        deps=[d for d in t.pop("_raw_deps",[]) if d in valid_ids and d!=t["id"]]
        cond=list(t.get("dependency_conditions") or [])
        while len(cond)<len(deps): cond.append("completed")
        t["depends_on"]=deps
        t["dependency_conditions"]=cond[:len(deps)]
    # Inject a report task only when the plan has none, and use a simple report role rather than replacing an arbitrary task.
    if not any(t.get("phase")=="report" or t.get("agent_role")=="report" for t in out):
        if len(out)<MAX_TASKS:
            rpt=dict(default_plan(g)["tasks"][-1]); rpt["id"]=f"T{len(out)+1}"; rpt["depends_on"]=[t["id"] for t in out if t.get("required",False)]; rpt["dependency_conditions"]=["completed"]*len(rpt["depends_on"]); out.append(rpt)
        else:
            last=out[-1]; last["phase"]="report"; last["agent_role"]="report"
    # Deduplicate agents by canonical role. Multiple task references to the same role share one instance.
    agents=[]; roles=set()
    for a in plan.get("agents",[]):
        if not isinstance(a,dict): continue
        role=" ".join(str(a.get("role") or "").strip().lower().split())
        if not role or role in roles: continue
        roles.add(role)
        agents.append({
            "name":str(a.get("name") or role.title()),
            "role":role,
            "instructions":str(a.get("instructions") or f"Act as {role} specialist."),
            "capabilities":[str(x) for x in (a.get("capabilities") or [])]
        })
        if len(agents)>=MAX_AGENTS: break
    for t in out:
        role=t["agent_role"]
        if role not in roles:
            agents.append({"name":role.title(),"role":role,"instructions":f"Act as {role} specialist.","capabilities":[]}); roles.add(role)
            if len(agents)>=MAX_AGENTS: break
    plan["agents"]=agents[:MAX_AGENTS]
    plan["tasks"]=out
    plan["verification"]=plan.get("verification") if isinstance(plan.get("verification"),dict) else default_plan(g)["verification"]
    return plan


def create_run(gid,reason="initial"):
    g=get_goal(gid); n=int(fetch_one("SELECT COALESCE(MAX(run_number),0)+1 n FROM runs WHERE goal_id=?",(gid,))["n"]);rid=uid();write("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?)",(rid,gid,None,n,"planning",reason,None,None,now()));write("UPDATE goals SET current_run_id=?,status='planning',updated_at=? WHERE id=?",(rid,now(),gid));return fetch_one("SELECT * FROM runs WHERE id=?",(rid,))

def ensure_agent(run_id,spec):
    role=str(spec.get("role") or "generalist");p=profile_by_role(role)
    if not p:
        pid=uid();write("INSERT INTO agent_profiles VALUES(?,?,?,?,?,?,?,?,?)",(pid,spec.get("name") or role,role,spec.get("instructions") or f"Act as {role}.",json.dumps(spec.get("capabilities",[])),"[]","{}",now(),now()));p=fetch_one("SELECT * FROM agent_profiles WHERE id=?",(pid,))
    existing=fetch_one("SELECT * FROM agent_instances WHERE run_id=? AND profile_id=?",(run_id,p["id"]))
    if existing:return existing
    iid=uid();write("INSERT INTO agent_instances VALUES(?,?,?,?,?,?,?,?,?,?)",(iid,run_id,p["id"],spec.get("name") or p["name"],spec.get("instructions"),"idle",0,0,now(),now()));return fetch_one("SELECT * FROM agent_instances WHERE id=?",(iid,))

def make_plan(run_id,g):
    system="""You are the CEO of an AI workforce. Build the smallest sufficient workforce and task graph for the user's objective. First classify the objective mentally as compute, research, reason, create, action, or mixed. Assign only the capabilities that are actually needed. Do not require web research for deterministic calculations, reasoning, or writing. Detect blocking ambiguity and capability gaps. Dependencies must express real information requirements. Include task contracts, budgets and a verification plan. Never invent evidence. If requirements conflict, expose the conflict rather than silently assuming it away. Return ONLY the schema-defined object."""
    prompt=f"Objective: {g['title']}\nDescription: {g['description']}\nSuccess criteria: {g['criteria']}\nGoal budget: ${g['budget']:.2f}"
    try:
        r=call_model(run_id,None,None,system,prompt,purpose="planner",structured_schema=plan_schema(),max_output_tokens=2000,max_attempts=2)
        plan=r["structured"]
        if not isinstance(plan,dict) or not plan.get("tasks"):
            raise ValueError("Planner returned an unusable structured plan")
        log_event(run_id=run_id,goal_id=g["id"],kind="plan_created",message="CEO created structured plan.",payload={"fallback":False,"objective_type":classify_objective(g)})
    except Exception as exc:
        # Safe recovery: never substitute a generic AI-business workflow for the user's actual objective.
        plan=minimal_safe_plan(g)
        log_event(run_id=run_id,goal_id=g["id"],kind="planner_recovery",message=f"Planner recovery activated: {type(exc).__name__}: {exc}",payload={"degraded":True,"objective_type":classify_objective(g)})
        write("UPDATE runs SET reason=? WHERE id=?",(f"SAFE_PLANNER_RECOVERY: {type(exc).__name__}",run_id))
    return normalize_plan(plan,g)


def create_tasks(run_id,plan):
    rolemap={}
    for a in plan["agents"]:
        inst=ensure_agent(run_id,a);rolemap[str(a.get("role"))]=inst["id"]
    mapping={}
    for t in plan["tasks"]:
        aid=rolemap.get(t["agent_role"])
        if not aid: raise RuntimeError(f"No agent for role {t['agent_role']}")
        tid=uid();write("INSERT INTO tasks(id,run_id,plan_task_id,agent_instance_id,title,instructions,contract_json,status,required,output,structured_output_json,confidence,budget_limit,spent,attempt_count,max_attempts,error_type,error_message,phase,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(tid,run_id,t["id"],aid,t["title"],t["instructions"],json.dumps(t["contract"]),"pending",1 if t.get("required",False) else 0,None,None,None,t["budget_limit"],0,0,t["max_attempts"],None,None,t["phase"],now(),now()));mapping[t["id"]]=tid
    for t in plan["tasks"]:
        for i,dep in enumerate(t.get("depends_on",[])):
            if dep not in mapping:continue
            cond=(t.get("dependency_conditions") or ["completed"]*len(t.get("depends_on",[])))[i]
            required=0 if cond in ("optional","completed_or_failed_with_fallback") else 1
            write("INSERT INTO task_dependencies VALUES(?,?,?,?,?,?,?)",(uid(),run_id,mapping[dep],mapping[t["id"]],cond,required,now()))
    return mapping


def deps(run_id,tid):return fetch("SELECT * FROM task_dependencies WHERE run_id=? AND downstream_task_id=?",(run_id,tid))
def dep_state(d):
    t=fetch_one("SELECT status FROM tasks WHERE id=?",(d["upstream_task_id"],))
    if not t:return "missing"
    return t["status"]
def dep_satisfied(d):
    s=dep_state(d);c=d["condition"]
    if c=="completed":return s=="completed"
    if c in ("completed_or_failed_with_fallback","optional"):return s in ("completed","failed","blocked","cancelled")
    if c=="artifact_available":return bool(fetch_one("SELECT id FROM artifacts WHERE task_id=? AND status='available' LIMIT 1",(d["upstream_task_id"],)))
    return s=="completed"
def dep_impossible(d):
    s=dep_state(d);c=d["condition"]
    if c=="completed":return s in ("failed","blocked","cancelled","interrupted")
    if c=="artifact_available":return s in ("failed","blocked","cancelled","interrupted") and not fetch_one("SELECT id FROM artifacts WHERE task_id=? AND status='available' LIMIT 1",(d["upstream_task_id"],))
    return False


def relevant_context(run_id,tid):
    pieces=[]
    for d in deps(run_id,tid):
        up=fetch_one("SELECT * FROM tasks WHERE id=?",(d["upstream_task_id"],))
        if not up:continue
        pieces.append(f"UPSTREAM {up['title']} [{up['status']}]\n{up['output'] or up['error_message'] or ''}")
        for e in fetch("SELECT * FROM evidence WHERE task_id=? ORDER BY created_at DESC LIMIT 8",(up["id"],)):pieces.append(f"EVIDENCE: {e['claim']} | {e['source_title'] or ''} | {e['source_url'] or ''}")
        for a in fetch("SELECT * FROM artifacts WHERE task_id=? AND status='available' ORDER BY created_at DESC LIMIT 3",(up["id"],)):pieces.append(f"ARTIFACT: {a['name']}\n{a['content'] or ''}")
    g=get_goal_from_run(run_id)
    for m in fetch("SELECT * FROM memory_items WHERE goal_id=? ORDER BY updated_at DESC LIMIT 8",(g["id"],)):pieces.append(f"MEMORY [{m['memory_type']}] {m['key']}: {m['value']}")
    return "\n\n".join(pieces[-24:])


def add_evidence(run_id,task_id,cites):
    ids=[]
    for c in cites:
        eid=uid();write("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(eid,run_id,task_id,"Source referenced by model output","web",c.get("title"),c.get("url"),c.get("publisher"),c.get("published_at"),now(),None,"captured",None,None,json.dumps(c),now()));ids.append(eid)
    return ids

def create_artifact(run_id,task_id,name,content):
    aid=uid();digest=hashlib.sha256((content or "").encode()).hexdigest();v=int(fetch_one("SELECT COALESCE(MAX(version),0)+1 v FROM artifacts WHERE task_id=? AND name=?",(task_id,name))["v"])
    write("INSERT INTO artifacts(id,run_id,task_id,name,artifact_type,version,content,path,mime_type,content_hash,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(aid,run_id,task_id,name,"text",v,content,None,"text/plain",digest,"available",now()));return aid

def confidence_from(data,evidence_count,contradictions=0):
    score=.45+min(.25,evidence_count*.06)+(.10 if data.get("facts") else 0)+(.08 if data.get("assumptions") or data.get("unknowns") else 0)-min(.20,contradictions*.07)
    return round(max(.05,min(.95,score)),3)

def save_memory(run_id,task,agent_id,data,eids,conf):
    g=get_goal_from_run(run_id);t=now()
    rows=[("goal_memory","task_summary",data.get("summary","")[:1200]),("agent_learning",task["title"],data.get("summary","")[:1200])]
    for memory_type,key,val in rows:
        if val:write("INSERT INTO memory_items VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),company()["id"],g["id"],task["id"],agent_id,memory_type,key,val,json.dumps(eids),conf,30,t,t))
    for u in data.get("unknowns",[])[:10]:write("INSERT INTO memory_items VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),company()["id"],g["id"],task["id"],agent_id,"goal_memory","unknown",str(u)[:500],json.dumps(eids),.4,14,t,t))
    for fact in data.get("facts",[])[:10]:
        write("INSERT INTO memory_items VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),company()["id"],g["id"],task["id"],agent_id,"company_memory","fact",str(fact)[:700],json.dumps(eids),conf,30,t,t))

def run_task(run_id,task_id):
    task=fetch_one("SELECT * FROM tasks WHERE id=?",(task_id,));
    if not task:return False
    goal=get_goal_from_run(run_id);agent=fetch_one("SELECT ai.*,p.role,p.instructions base_instructions,p.capabilities_json FROM agent_instances ai JOIN agent_profiles p ON p.id=ai.profile_id WHERE ai.id=?",(task["agent_instance_id"],))
    write("UPDATE tasks SET status='running',updated_at=?,error_type=NULL,error_message=NULL WHERE id=?",(now(),task_id));write("UPDATE agent_instances SET status='working',updated_at=? WHERE id=?",(now(),task["agent_instance_id"]))
    log_event(run_id=run_id,goal_id=goal["id"],task_id=task_id,kind="task_started",message=f"{agent['name']} started {task['title']}")
    system=f"""You are the {agent['role']} in a coordinated AI workforce.\n{agent['base_instructions']}\n\nTask contract:\n{task['contract_json']}\n\nReturn structured work. Facts must be evidence-backed when external. Label assumptions and unknowns. Never fabricate sources. If evidence is insufficient, say so."""
    prompt=f"""Objective: {goal['title']}\nDescription: {goal['description']}\nSuccess criteria: {goal['criteria']}\nTask: {task['title']}\nInstructions: {task['instructions']}\nDirect dependency context:\n{relevant_context(run_id,task_id) or '(none)'}"""
    max_attempts=int(task["max_attempts"])
    for attempt in range(1,max_attempts+1):
        attempt_started=time.time(); attempt_started_iso=now()
        try:
            wants_web = bool(task["phase"] == "work" and json.loads(task["contract_json"] or "{}").get("requires_web", False) and task["instructions"] is not None)
            if wants_web and not tool_allowed(agent, "web_search"):
                log_event(run_id=run_id, goal_id=goal["id"], task_id=task_id, kind="capability_gap", message="Task requested web search but assigned agent lacks web_research capability.")
                wants_web = False
            result=call_model(run_id,task_id,task["agent_instance_id"],system,prompt,purpose="task",use_web=wants_web,structured_schema=task_output_schema(),max_output_tokens=1500,max_attempts=1)
            data=result["structured"]
            if not isinstance(data,dict): raise ValueError("Task structured output is not an object")
            eids=add_evidence(run_id,task_id,result["citations"]);conf=confidence_from(data,len(eids));aid=create_artifact(run_id,task_id,f"{task['title']} â work output",result["text"])
            aended=result.get("ended_at") or now()
            write("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),task_id,attempt,"completed",result["model"],result.get("started_at",attempt_started_iso),aended,result["latency_ms"],result["input_tokens"],result["output_tokens"],result["cost"],result.get("request_id"),None,json.dumps({"strategy":"primary","request_id":result.get("request_id"),"web":wants_web}),now()))
            write("UPDATE tasks SET status='completed',attempt_count=?,output=?,structured_output_json=?,confidence=?,updated_at=?,error_type=NULL,error_message=NULL WHERE id=?",(attempt,result["text"],json.dumps(data),conf,now(),task_id));write("UPDATE agent_instances SET status='idle',updated_at=? WHERE id=?",(now(),task["agent_instance_id"]))
            save_memory(run_id,task,task["agent_instance_id"],data,eids,conf)
            downstream=fetch("SELECT downstream_task_id FROM task_dependencies WHERE run_id=? AND upstream_task_id=?",(run_id,task_id))
            for d in downstream:
                hid=uid();write("INSERT INTO handoffs VALUES(?,?,?,?,?,?,?,?,?,?,?)",(hid,run_id,task_id,d["downstream_task_id"],data.get("summary",result["text"][:1200]),json.dumps(eids),json.dumps([aid]),json.dumps(data.get("assumptions",[])),json.dumps(data.get("unknowns",[])),conf,now()))
                write("INSERT INTO artifact_dependencies VALUES(?,?,?,?,?,?)",(uid(),run_id,aid,d["downstream_task_id"],"handoff",now()))
            log_event(run_id=run_id,goal_id=goal["id"],task_id=task_id,kind="task_completed",message=f"{agent['name']} completed {task['title']}",payload={"confidence":conf,"evidence":len(eids),"artifact_id":aid,"attempt":attempt})
            return True
        except Exception as exc:
            cls=classify_error(exc);msg=str(exc);lat=int((time.time()-attempt_started)*1000)
            write("INSERT INTO task_attempts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),task_id,attempt,"failed",model_name_for("task", agent),attempt_started_iso,now(),lat,0,0,0,None,cls,msg,json.dumps({"strategy":"retry","attempt":attempt}),now()))
            write("UPDATE tasks SET attempt_count=?,error_type=?,error_message=?,output=?,updated_at=? WHERE id=?",(attempt,cls,msg,f"ATTEMPT {attempt} FAILED\n{type(exc).__name__}: {msg}",now(),task_id))
            log_event(run_id=run_id,goal_id=goal["id"],task_id=task_id,kind="task_failed",message=f"{agent['name']} failed {task['title']}: {cls}",payload={"attempt":attempt,"latency_ms":lat,"error":msg})
            if cls.startswith("transient") and attempt<max_attempts:
                write("UPDATE tasks SET status='retrying',updated_at=? WHERE id=?",(now(),task_id));time.sleep(min(2**(attempt-1),4));continue
            write("UPDATE tasks SET status='failed',updated_at=? WHERE id=?",(now(),task_id));write("UPDATE agent_instances SET status='idle',updated_at=? WHERE id=?",(now(),task["agent_instance_id"]));return False
    return False


def schedule_work(run_id):
    while True:
        rows=fetch("SELECT * FROM tasks WHERE run_id=? AND phase='work'",(run_id,));pending=[r for r in rows if r["status"] in ("pending","retrying","waiting_dependency","ready")]
        if not pending:
            required_rows=fetch("SELECT status,required FROM tasks WHERE run_id=? AND phase='work'",(run_id,))
            return all(r["status"]=="completed" or not int(r["required"]) for r in required_rows)
        ready=[];changed=False
        for t in pending:
            ds=deps(run_id,t["id"])
            if any(d["required"] and dep_impossible(d) for d in ds):write("UPDATE tasks SET status='blocked',error_type='dependency',error_message='Required dependency failed.',updated_at=? WHERE id=?",(now(),t["id"]));changed=True;continue
            if all(dep_satisfied(d) or not d["required"] for d in ds):ready.append(t["id"])
            else:write("UPDATE tasks SET status='waiting_dependency',updated_at=? WHERE id=?",(now(),t["id"]))
        if not ready:
            if changed:continue
            # no resolvable work: mark cycle/deadlock as blocked
            for t in pending:write("UPDATE tasks SET status='blocked',error_type='dependency_cycle',error_message='Task graph could not be resolved.',updated_at=? WHERE id=?",(now(),t["id"]))
            return False
        futs={task_executor.submit(run_task,run_id,tid):tid for tid in ready}
        for f in as_completed(futs):
            try:f.result()
            except Exception as exc:log_event(run_id=run_id,task_id=futs[f],kind="executor_error",message=str(exc))


def evaluation_context(run_id):
    parts=[]
    for t in fetch("SELECT * FROM tasks WHERE run_id=? AND phase='work' ORDER BY created_at",(run_id,)):
        parts.append(f"TASK {t['plan_task_id']} {t['title']} status={t['status']}\n{t['output'] or t['error_message'] or ''}")
        for e in fetch("SELECT * FROM evidence WHERE task_id=? ORDER BY created_at DESC LIMIT 6",(t["id"],)):parts.append(f"EVIDENCE {e['id']}: {e['claim']} | {e['source_title']} | {e['source_url']}")
    return "\n\n".join(parts)

def required_work_ready(run_id):
    rows=fetch("SELECT id,title,status,required FROM tasks WHERE run_id=? AND phase='work' ORDER BY created_at",(run_id,))
    blockers=[]
    for r in rows:
        if int(r["required"]) and r["status"]!="completed": blockers.append({"task_id":r["id"],"title":r["title"],"status":r["status"]})
    return (not blockers), blockers

def mark_unresolvable_report_tasks(run_id):
    reports=fetch("SELECT * FROM tasks WHERE run_id=? AND phase='report' AND status IN ('pending','waiting_dependency','ready','retrying')",(run_id,))
    for t in reports:
        ds=deps(run_id,t["id"])
        impossible=[d for d in ds if d["required"] and dep_impossible(d)]
        if impossible:
            labels=[]
            for d in impossible:
                up=fetch_one("SELECT title,status FROM tasks WHERE id=?",(d["upstream_task_id"],))
                labels.append(f"{up['title'] if up else d['upstream_task_id']} [{up['status'] if up else 'missing'}]")
            write("UPDATE tasks SET status='blocked',error_type='dependency',error_message=?,updated_at=? WHERE id=?",("Required dependency unavailable: "+"; ".join(labels),now(),t["id"]))
            log_event(run_id=run_id,task_id=t["id"],kind="task_blocked",message="Report task blocked by required dependency.",payload={"dependencies":labels})

def evaluate(run_id):
    g=get_goal_from_run(run_id);vp=json.loads(g["verification_plan_json"] or "{}");threshold=float(vp.get("pass_threshold",.8))
    system="""You are an independent evaluator. Judge the actual workforce record against the user's success criteria. Check completeness, evidence, assumptions/unknowns, contradictions, numerical integrity, risk and actionability. A fluent answer is not enough. Return only the required JSON."""
    prompt=f"Objective: {g['title']}\nDescription: {g['description']}\nSuccess criteria: {g['criteria']}\nThreshold: {threshold}\nWorkforce record:\n{evaluation_context(run_id)}"
    try:r=call_model(run_id,None,None,system,prompt,purpose="evaluator",structured_schema=evaluator_schema(),max_output_tokens=1500,max_attempts=3)
    except Exception as exc:
        log_event(run_id=run_id,goal_id=g["id"],kind="evaluator_failure",message=f"Evaluator service failed: {type(exc).__name__}: {exc}",payload={"error_type":classify_error(exc)})
        return {"service_failed":True,"passed":False,"score":None,"failures":["Evaluator service unavailable"],"replan_tasks":[],"contradictions":[]}
    d=r["structured"];write("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",(uid(),run_id,"pre_report",float(d.get("score",0)),1 if d.get("passed") else 0,json.dumps(d.get("checks",[])),json.dumps(d.get("failures",[])),json.dumps(d.get("replan_tasks",[])),r["model"],r["input_tokens"],r["output_tokens"],r["cost"],now()))
    all_e=fetch("SELECT id,claim FROM evidence WHERE run_id=?",(run_id,))
    for c in d.get("contradictions",[]):
        if not c.get("resolution_needed"):continue
        ea=[x["id"] for x in all_e if any(w.lower() in x["claim"].lower() for w in str(c.get("claim_a","")).split()[:3])][:5] or [x["id"] for x in all_e[:5]]
        eb=[x["id"] for x in all_e if any(w.lower() in x["claim"].lower() for w in str(c.get("claim_b","")).split()[:3])][:5] or [x["id"] for x in all_e[:5]]
        write("INSERT INTO contradictions VALUES(?,?,?,?,?,?,?,?,?,?,?)",(uid(),run_id,c.get("claim_a",""),c.get("claim_b",""),json.dumps(ea),json.dumps(eb),c.get("severity","medium"),"open",None,now(),now()))
    return d


def replan(run_id,evaluation):
    g=get_goal_from_run(run_id);count=int(g["replan_count"])
    if count>=int(g["max_replans"]):return 0
    plan=json.loads(g["plan_json"] or "{}");existing={t.get("id") for t in plan.get("tasks",[])};rolemap={}
    for a in fetch("SELECT ai.*,p.role FROM agent_instances ai JOIN agent_profiles p ON p.id=ai.profile_id WHERE ai.run_id=?",(run_id,)):rolemap[a["role"]]=a["id"]
    added=0
    for i,s in enumerate(evaluation.get("replan_tasks",[])[:3],1):
        role=str(s.get("agent_role") or "qa")
        if role not in rolemap:rolemap[role]=ensure_agent(run_id,{"role":role,"name":role,"instructions":f"Act as {role}.","capabilities":[]})["id"]
        pid=f"R{count+1}_{i}";j=i
        while pid in existing:j+=1;pid=f"R{count+1}_{j}"
        t={"id":pid,"title":str(s.get("title") or "Close evaluator gap"),"instructions":str(s.get("instructions") or "Resolve evaluator finding."),"agent_role":role,"depends_on":[str(x) for x in s.get("depends_on",[]) if str(x) in existing],"dependency_conditions":["completed"]*len(s.get("depends_on",[])),"required":True,"requires_web":bool(s.get("requires_web",False)),"budget_limit":max(.05,float(s.get("budget_limit",.3) or .3)),"max_attempts":2,"phase":"work","contract":{"inputs":[],"outputs":["gap_resolution"],"success_conditions":[str(x) for x in s.get("success_conditions",[])],"failure_conditions":["insufficient evidence"],"evidence_required":bool(s.get("requires_web",False)),"requires_web":bool(s.get("requires_web",False)),"notes":"Generated from evaluator finding."}}
        tid=uid();write("INSERT INTO tasks(id,run_id,plan_task_id,agent_instance_id,title,instructions,contract_json,status,required,output,structured_output_json,confidence,budget_limit,spent,attempt_count,max_attempts,error_type,error_message,phase,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(tid,run_id,pid,rolemap[role],t["title"],t["instructions"],json.dumps(t["contract"]),"pending",1 if t.get("required",False) else 0,None,None,None,t["budget_limit"],0,0,2,None,None,"work",now(),now()))
        for dep in t["depends_on"]:
            up=fetch_one("SELECT id FROM tasks WHERE run_id=? AND plan_task_id=?",(run_id,dep))
            if up:write("INSERT INTO task_dependencies VALUES(?,?,?,?,?,?,?)",(uid(),run_id,up["id"],tid,"completed",1,now()))
        plan.setdefault("tasks",[]).append(t);existing.add(pid);added+=1
    if added:write("UPDATE goals SET replan_count=replan_count+1,plan_json=?,status='executing',updated_at=? WHERE id=?",(json.dumps(plan,indent=2),now(),g["id"]));log_event(run_id=run_id,goal_id=g["id"],kind="replan",message=f"Added {added} evaluator-driven tasks.")
    return added


def execute_report(run_id):
    g=get_goal_from_run(run_id);ev=fetch_one("SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 1",(run_id,));report=fetch_one("SELECT * FROM tasks WHERE run_id=? AND phase='report' ORDER BY created_at DESC LIMIT 1",(run_id,))
    if not ev or not ev["passed"]:return False,"Report blocked: evaluator has not passed."
    if not report:return False,"No report task."
    mark_unresolvable_report_tasks(run_id)
    report=fetch_one("SELECT * FROM tasks WHERE id=?",(report["id"],))
    if report["status"]=="blocked":return False,report["error_message"] or "Report blocked by dependency."
    ready,blockers=required_work_ready(run_id)
    if not ready:return False,"Report blocked: required work incomplete: "+"; ".join(f"{b['title']} [{b['status']}]" for b in blockers)
    if any(d["required"] and not dep_satisfied(d) for d in deps(run_id,report["id"])):return False,"Report blocked by dependency."
    system="You are the executive report writer. Use only the verified workforce record. Produce a decision-ready report with facts/evidence, assumptions, unknowns, contradictions, risks, recommendations and next actions. Do not invent evidence."
    prompt=f"Objective: {g['title']}\nDescription: {g['description']}\nSuccess criteria: {g['criteria']}\nVerified record:\n{evaluation_context(run_id)}"
    try:r=call_model(run_id,report["id"],report["agent_instance_id"],system,prompt,purpose="report",structured_schema=None,max_output_tokens=2200,max_attempts=2)
    except Exception as exc:
        write("UPDATE tasks SET status='failed',error_type=?,error_message=?,updated_at=? WHERE id=?",(classify_error(exc),str(exc),now(),report["id"]));return False,str(exc)
    eids=add_evidence(run_id,report["id"],r["citations"]);aid=create_artifact(run_id,report["id"],"Final executive report",r["text"])
    write("UPDATE tasks SET status='completed',output=?,confidence=?,updated_at=? WHERE id=?",(r["text"],.85,now(),report["id"]));write("UPDATE runs SET status='completed',ended_at=? WHERE id=?",(now(),run_id));write("UPDATE goals SET status='completed',final_output=?,updated_at=? WHERE id=?",(r["text"],now(),g["id"]));log_event(run_id=run_id,goal_id=g["id"],task_id=report["id"],kind="goal_completed",message="Verified report delivered.",payload={"artifact_id":aid,"evidence":len(eids)});return True,r["text"]


def execute_goal(gid):
    try:
        g=get_goal(gid);run=create_run(gid,"initial");rid=run["id"];log_event(run_id=rid,goal_id=gid,kind="planning",message="CEO designing workforce.");plan=make_plan(rid,g);write("UPDATE goals SET plan_json=?,verification_plan_json=?,max_replans=?,status='executing',updated_at=? WHERE id=?",(json.dumps(plan,indent=2),json.dumps(plan.get("verification",{}),indent=2),MAX_REPLANS,now(),gid));create_tasks(rid,plan);write("UPDATE runs SET status='executing',started_at=? WHERE id=?",(now(),rid))
        for cycle in range(MAX_REPLANS+1):
            schedule_ok=schedule_work(rid)
            ready,blockers=required_work_ready(rid)
            mark_unresolvable_report_tasks(rid)
            log_event(run_id=rid,goal_id=gid,kind="work_readiness",message=f"Required work ready={ready}",payload={"schedule_ok":schedule_ok,"blockers":blockers})
            ev=evaluate(rid) if ready else {"service_failed":False,"not_evaluated":True,"passed":False,"score":None,"failures":["Required work incomplete"],"replan_tasks":[],"contradictions":[]}
            log_event(run_id=rid,goal_id=gid,kind="evaluation_skipped" if ev.get("not_evaluated") else "evaluation",message=("Evaluator not run: required work incomplete" if ev.get("not_evaluated") else f"Evaluator score={ev.get('score')} passed={ev.get('passed')}"))
            if ev.get("service_failed"):
                write("UPDATE runs SET status='verification_failed',ended_at=?,reason=? WHERE id=?",(now(),"Evaluator service unavailable",rid));write("UPDATE goals SET status='verification_failed',final_output=?,updated_at=? WHERE id=?",(f"VERIFICATION SERVICE FAILED: {ev.get('failures',["Evaluator service unavailable"])[0]}",now(),gid));break
            if ev.get("passed"):
                ready,blockers=required_work_ready(rid)
                if not ready:
                    write("UPDATE runs SET status='incomplete',ended_at=?,reason=? WHERE id=?",(now(),"Evaluator pass rejected because required work is incomplete.",rid))
                    write("UPDATE goals SET status='incomplete',final_output=?,updated_at=? WHERE id=?",("INCOMPLETE: evaluator passed but required work remained incomplete: "+"; ".join(f"{b['title']} [{b['status']}]" for b in blockers),now(),gid))
                    break
                ok,msg=execute_report(rid)
                if not ok:write("UPDATE runs SET status='failed',ended_at=?,reason=? WHERE id=?",(now(),msg,rid));write("UPDATE goals SET status='failed',final_output=?,updated_at=? WHERE id=?",(f"REPORT FAILED: {msg}",now(),gid))
                break
            if not replan(rid,ev):
                write("UPDATE runs SET status='incomplete',ended_at=?,reason=? WHERE id=?",(now(),"Evaluation failed and no further replan was available.",rid));write("UPDATE goals SET status='incomplete',final_output=?,updated_at=? WHERE id=?",("INCOMPLETE: required work was not ready; evaluator was not run.",now(),gid));break
    except Exception as exc:
        g=get_goal(gid)
        if g:write("UPDATE goals SET status='failed',final_output=?,updated_at=? WHERE id=?",(f"EXECUTION FAILED\n{type(exc).__name__}: {exc}",now(),gid));log_event(goal_id=gid,kind="goal_failed",message=f"{type(exc).__name__}: {exc}")
    finally:
        write("UPDATE agent_instances SET status='idle',updated_at=? WHERE status='working'",(now(),))
        with future_lock:goal_futures.pop(gid,None)


@app.get("/login",response_class=HTMLResponse)
def login():return "<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><h1>AI Workforce OS</h1><form method='post'><input name=token type=password placeholder='Access token' required><button>Sign in</button></form>"
@app.post("/login")
def login_post(token:str=Form(...)):
    if not APP_ACCESS_TOKEN or not hmac.compare_digest(token,APP_ACCESS_TOKEN):return HTMLResponse("Invalid token",status_code=401)
    r=RedirectResponse("/",303);r.set_cookie("wf_auth",hashlib.sha256(APP_ACCESS_TOKEN.encode()).hexdigest(),httponly=True,samesite="lax",secure=False);return r

@app.get("/",response_class=HTMLResponse)
def home(request:Request):
    denied=require_auth(request)
    if denied:return denied
    goals=fetch("SELECT * FROM goals ORDER BY created_at DESC");agents=fetch("SELECT * FROM agent_profiles ORDER BY role")
    gh="".join(f"<div class=row><a href='/goals/{g['id']}'><b>{esc(g['title'])}</b></a> Â· {esc(g['status'])} Â· ${g['spent']:.4f}/${g['budget']:.2f} Â· replans {g['replan_count']}</div>" for g in goals) or "<p>No objectives yet.</p>"
    ah="".join(f"<div class=card><b>{esc(a['name'])}</b><div class=muted>{esc(a['role'])}</div><p>{esc(a['instructions'])}</p></div>" for a in agents)
    return f"""<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><title>AI Workforce OS</title><style>body{{margin:0;background:#f3f5f7;font-family:-apple-system,system-ui}}main{{max-width:1100px;margin:auto;padding:18px}}section,.card{{background:#fff;border:1px solid #dfe3e7;border-radius:14px;padding:16px;margin:12px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}input,textarea,button{{width:100%;box-sizing:border-box;padding:11px;margin:6px 0;border-radius:9px;font:inherit}}textarea{{min-height:100px}}button{{background:#111;color:#fff;border:0;font-weight:700}}.row{{padding:10px 0;border-bottom:1px solid #eee}}.muted{{color:#69717c;font-size:.9em}}a{{color:#145ac6;text-decoration:none}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style><main><h1>AI WORKFORCE OS <small>{APP_VERSION}</small></h1><div class=grid><section><h2>Give the company an objective</h2><form method=post action=/goals><input name=title placeholder='Objective title' required><textarea name=description placeholder='Describe what you want the company to accomplish.' required></textarea><textarea name=criteria placeholder='What does success look like?' required></textarea><label class=muted for=budget>Budget (USD) â optional; leave blank to use the system safety budget</label><input id=budget name=budget type=number min=0 step=.01 placeholder='e.g. 5.00'><button>Create objective</button></form></section><section><h2>Objectives</h2>{gh}</section></div><section><h2>Workforce profiles</h2><div class=cards>{ah}</div></section></main>"""

@app.post("/goals")
def create_goal(request:Request,title:str=Form(...),description:str=Form(...),criteria:str=Form(...),budget:Optional[float]=Form(None)):
    denied=require_auth(request)
    if denied:return denied
    if len(title)>300 or len(description)>12000 or len(criteria)>12000:return JSONResponse({"error":"Input too long"},status_code=413)
    if budget is None: budget=DEFAULT_GOAL_BUDGET
    if budget<0 or budget>1000:return JSONResponse({"error":"Budget out of allowed prototype range"},status_code=400)
    c=company();gid=uid();write("INSERT INTO goals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(gid,c["id"],title.strip(),description.strip(),criteria.strip(),budget,0,"queued",None,None,None,None,0,MAX_REPLANS,now(),now()));log_event(company_id=c["id"],goal_id=gid,kind="goal_created",message=f"Human assigned objective: {title.strip()}")
    with future_lock:goal_futures[gid]=goal_executor.submit(execute_goal,gid)
    return RedirectResponse(f"/goals/{gid}",303)

@app.post("/goals/{gid}/retry")
def retry_goal(request:Request,gid:str):
    denied=require_auth(request)
    if denied:return denied
    g=get_goal(gid)
    if not g:return JSONResponse({"error":"not found"},404)
    if g["status"] not in ("failed","incomplete","interrupted","verification_failed"):return JSONResponse({"error":"not retryable"},400)
    write("UPDATE goals SET status='queued',spent=0,final_output=NULL,plan_json=NULL,verification_plan_json=NULL,replan_count=0,updated_at=? WHERE id=?",(now(),gid))
    with future_lock:goal_futures[gid]=goal_executor.submit(execute_goal,gid)
    return RedirectResponse(f"/goals/{gid}",303)

@app.get("/goals/{gid}",response_class=HTMLResponse)
def detail(request:Request,gid:str):
    denied=require_auth(request)
    if denied:return denied
    g=get_goal(gid)
    if not g:return HTMLResponse("Not found",404)
    ts=fetch("SELECT t.*,p.role agent_role,ai.name agent_name FROM tasks t JOIN agent_instances ai ON ai.id=t.agent_instance_id JOIN agent_profiles p ON p.id=ai.profile_id WHERE t.run_id=? ORDER BY t.phase,t.created_at",(g["current_run_id"],)) if g["current_run_id"] else []
    hs=fetch("SELECT h.*,a1.name from_name,a2.name to_name FROM handoffs h JOIN tasks t1 ON t1.id=h.from_task_id JOIN tasks t2 ON t2.id=h.to_task_id JOIN agent_instances a1 ON a1.id=t1.agent_instance_id JOIN agent_instances a2 ON a2.id=t2.agent_instance_id WHERE h.run_id=? ORDER BY h.created_at",(g["current_run_id"],)) if g["current_run_id"] else []
    evs=fetch("SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC",(g["current_run_id"],)) if g["current_run_id"] else []
    sources=fetch("SELECT * FROM evidence WHERE run_id=? ORDER BY created_at DESC LIMIT 40",(g["current_run_id"],)) if g["current_run_id"] else []
    events=fetch("SELECT * FROM events WHERE goal_id=? ORDER BY created_at DESC LIMIT 80",(gid,))
    th="".join(f"<details><summary><b>{esc(t['title'])}</b> â {esc(t['agent_name'])} Â· {esc(t['status'])} Â· confidence {esc(t['confidence'])}</summary><p class=muted>attempts={t['attempt_count']} Â· spend=${t['spent']:.4f} Â· budget=${t['budget_limit']:.4f}</p><pre>{esc(t['output'] or t['error_message'] or '')}</pre></details>" for t in ts) or "<p>Tasks will appear.</p>"
    hh="".join(f"<div class=row><b>{esc(h['from_name'])}</b> â <b>{esc(h['to_name'])}</b><pre>{esc(h['summary'])}</pre><div class=muted>assumptions: {esc(h['assumptions_json'])}<br>unknowns: {esc(h['unknowns_json'])}</div></div>" for h in hs) or "<p>No handoffs recorded.</p>"
    eh="".join(f"<div class=row><b>{('not evaluated' if e['score'] is None else f'{e["score"]:.2f}')}</b> Â· {'PASS' if e['passed'] else 'FAIL'}<pre>{esc(e['failures_json'])}</pre></div>" for e in evs) or "<p>No evaluation yet.</p>"
    sh="".join(f"<div class=row><a href='{esc(s['source_url'])}' target=_blank>{esc(s['source_title'] or s['source_url'])}</a></div>" for s in sources if s['source_url']) or "<p>No captured sources.</p>"
    ac="".join(f"<div class=row><small>{esc(e['created_at'][11:19])}</small> {esc(e['message'])}</div>" for e in events)
    retry=f"<form method=post action='/goals/{gid}/retry'><button>Retry goal</button></form>" if g['status'] in ('failed','incomplete','interrupted','verification_failed') else ""
    refresh="<script>setTimeout(()=>location.reload(),4000)</script>" if g['status'] in ('queued','planning','executing') else ""
    return f"""<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><title>{esc(g['title'])}</title><style>body{{margin:0;background:#f3f5f7;font-family:-apple-system,system-ui}}main{{max-width:1000px;margin:auto;padding:18px}}section{{background:#fff;border:1px solid #dfe3e7;border-radius:14px;padding:16px;margin:14px 0}}pre{{white-space:pre-wrap;background:#f7f8fa;padding:12px;border-radius:9px;overflow:auto}}.row{{padding:10px 0;border-bottom:1px solid #eee}}button{{width:100%;padding:11px;background:#111;color:#fff;border:0;border-radius:9px;font-weight:700}}a{{color:#145ac6;text-decoration:none}}.muted{{color:#69717c;font-size:.9em}}</style><main><a href='/'>â Workforce dashboard</a><h1>{esc(g['title'])}</h1><section><b>Version:</b> {APP_VERSION} Â· <b>Status:</b> {esc(g['status'])}<br><b>Budget:</b> ${g['spent']:.4f}/${g['budget']:.2f} Â· <b>Replans:</b> {g['replan_count']}/{g['max_replans']}</section>{retry}<section><h2>CEO plan</h2><pre>{esc(g['plan_json'] or 'Planning in progress...')}</pre></section><section><h2>Task execution</h2>{th}</section><section><h2>Evaluator</h2>{eh}</section><section><h2>Agent handoffs</h2>{hh}</section><section><h2>Evidence / sources</h2>{sh}</section><section><h2>Final output</h2><pre>{esc(g['final_output'] or 'Verification/report in progress...')}</pre></section><section><h2>Activity</h2>{ac}</section></main>{refresh}"""

@app.get("/api/goals/{gid}")
def api_goal(request:Request,gid:str):
    denied=require_auth(request)
    if denied:return denied
    g=get_goal(gid)
    if not g:return JSONResponse({"error":"not found"},404)
    ts=fetch("SELECT * FROM tasks WHERE run_id=? ORDER BY created_at",(g["current_run_id"],)) if g["current_run_id"] else []
    return {"version":APP_VERSION,"schema_version":SCHEMA_VERSION,"goal":dict(g),"tasks":[dict(x) for x in ts],"evaluations":[dict(x) for x in fetch("SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at",(g["current_run_id"],))] if g["current_run_id"] else [],"events":[dict(x) for x in fetch("SELECT * FROM events WHERE goal_id=? ORDER BY created_at",(gid,))]}

@app.get("/diagnostics/generation")
def diagnostics_generation(request:Request):
    denied=require_auth(request)
    if denied:return denied
    key=os.getenv("OPENAI_API_KEY");model=os.getenv("OPENAI_MODEL","gpt-5-mini")
    if not key:return JSONResponse({"status":"failed","error_type":"ConfigurationError","message":"OPENAI_API_KEY is not configured","model":model},500)
    try:
        r=call_model(None,None,None,"Reply exactly with OK.","Reply exactly with: OK",purpose="diagnostic",max_output_tokens=32,max_attempts=1)
        return {"status":"ok","message":"Real Responses API generation succeeded.","version":APP_VERSION,"model":r["model"],"output":r["text"],"latency_ms":r["latency_ms"]}
    except Exception as exc:return JSONResponse({"status":"failed","error_type":type(exc).__name__,"message":str(exc),"version":APP_VERSION,"model":model},502)

@app.post("/benchmarks/start")
def benchmark_start(request: Request, case_name: str = Form(...), expected_version: str = Form(...)):
    denied=require_auth(request)
    if denied:return denied
    c=company();bid=uid();write("INSERT INTO benchmark_runs(id,app_version,goal_id,run_id,case_name,expected_version,version_match,started_at,ended_at,outcome,spend_delta,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(bid,APP_VERSION,None,None,case_name,expected_version,1 if APP_VERSION==expected_version else 0,now(),None,None,0.0,"Benchmark registration."));return {"benchmark_id":bid,"version":APP_VERSION,"version_match":APP_VERSION==expected_version}

@app.get("/health")
def health(expected_version: str = ""):
    match = (not expected_version) or (expected_version == APP_VERSION)
    return {"status":"ok" if match else "version_mismatch","version":APP_VERSION,"schema_version":SCHEMA_VERSION,"database":DB.name,"auth_enabled":bool(APP_ACCESS_TOKEN),"version_match":match,"expected_version":expected_version or None,"pid":os.getpid(),"openai_configured":bool(os.getenv("OPENAI_API_KEY")),"web_research_enabled":ENABLE_WEB_RESEARCH,"build_fingerprint":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]}

init()
