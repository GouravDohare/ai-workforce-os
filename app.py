import os, json, sqlite3, uuid, time, math, re, hashlib, threading, traceback
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from openai import OpenAI

APP_VERSION = "0.4.6"
SCHEMA_VERSION = "046-2"
DB_ENV = os.getenv("WORKFORCE_DB", "")
DB = DB_ENV or "workforce_v0466.db"
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
WEB = os.getenv("ENABLE_WEB_RESEARCH", "true").lower() == "true"
TIMEOUT = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "150"))
DEFAULT_BUDGET = float(os.getenv("DEFAULT_GOAL_BUDGET_USD", "5"))
MAX_BUDGET = float(os.getenv("MAX_GOAL_BUDGET_USD", "25"))
DAILY_CAP = float(os.getenv("GLOBAL_DAILY_SPEND_CAP_USD", "20"))
MAX_REPLANS = int(os.getenv("MAX_REPLANS", "2"))
AUTH = os.getenv("APP_ACCESS_TOKEN", "")
INPUT_PRICE = float(os.getenv("OPENAI_INPUT_PRICE_PER_MTOK", ".25"))
OUTPUT_PRICE = float(os.getenv("OPENAI_OUTPUT_PRICE_PER_MTOK", "2"))
MAX_CONCURRENT_GOALS = int(os.getenv("MAX_CONCURRENT_GOALS", "1"))
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))
RUN_TIMEOUT = float(os.getenv("RUN_TIMEOUT_SECONDS", "900"))

app = FastAPI(title="AI Workforce OS", version=APP_VERSION)
lock = threading.RLock()
goal_pool = ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_GOALS))
task_pool = ThreadPoolExecutor(max_workers=max(1, MAX_CONCURRENT_TASKS))

uid = lambda p: f"{p}_{uuid.uuid4().hex[:14]}"
now = lambda: datetime.now(timezone.utc).isoformat()

def jd(x):
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))

def jl(x, default=None):
    if not x:
        return {} if default is None else default
    try:
        return json.loads(x)
    except Exception:
        return {} if default is None else default

def db():
    c = sqlite3.connect(DB, check_same_thread=False, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c

def q(sql, params=(), one=False):
    with lock:
        c = db()
        try:
            rows = c.execute(sql, params).fetchall()
            return (rows[0] if rows else None) if one else rows
        finally:
            c.close()

def x(sql, params=()):
    with lock:
        c = db()
        try:
            r = c.execute(sql, params)
            c.commit()
            return r.lastrowid
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))

def table_columns(c, table):
    return [r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()]

def schema_ok(c):
    expected = {
        "goals": ["id","company_id","title","description","criteria","budget","spent","status","plan","replan_count","max_replans","verification_status","final_output","run_id","planner_degraded","created_at","updated_at"],
        "tasks": ["id","run_id","plan_id","instance_id","title","instructions","contract","status","output","structured","confidence","budget_limit","spent","attempts","max_attempts","required","requires_web","error_type","error_message","checkpoint","created_at","updated_at"],
        "reservations": ["id","goal_id","task_id","amount","status","created_at","settled_at"],
    }
    return all(table_columns(c, t) == cols for t, cols in expected.items())

def init():
    global DB
    with lock:
        c = db()
        if table_columns(c, "goals") and not schema_ok(c):
            c.close()
            # Never try to retrofit an unknown old schema in-place. Use a fresh, versioned DB.
            base = os.path.splitext(DB)[0]
            DB = base + "_v0466_fresh.db"
            c = db()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS companies(id TEXT PRIMARY KEY,name TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS agents(id TEXT PRIMARY KEY,name TEXT,role TEXT,instructions TEXT,capabilities TEXT,skills TEXT,model_policy TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY,company_id TEXT,title TEXT,description TEXT,criteria TEXT,budget REAL,spent REAL,status TEXT,plan TEXT,replan_count INT,max_replans INT,verification_status TEXT,final_output TEXT,run_id TEXT,planner_degraded INT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,goal_id TEXT,run_no INT,status TEXT,reason TEXT,started_at TEXT,ended_at TEXT,version TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS instances(id TEXT PRIMARY KEY,goal_id TEXT,agent_id TEXT,name TEXT,instructions TEXT,status TEXT,spend REAL,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,run_id TEXT,plan_id TEXT,instance_id TEXT,title TEXT,instructions TEXT,contract TEXT,status TEXT,output TEXT,structured TEXT,confidence REAL,budget_limit REAL,spent REAL,attempts INT,max_attempts INT,required INT,requires_web INT,error_type TEXT,error_message TEXT,checkpoint TEXT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS deps(id TEXT PRIMARY KEY,run_id TEXT,upstream TEXT,downstream TEXT,condition TEXT,required INT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,task_id TEXT,n INT,status TEXT,started_at TEXT,ended_at TEXT,latency_ms INT,input_tokens INT,output_tokens INT,cost REAL,model TEXT,error_type TEXT,error_message TEXT,request_id TEXT,strategy TEXT,tool_mode TEXT);
        CREATE TABLE IF NOT EXISTS model_calls(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,purpose TEXT,model TEXT,tool_mode TEXT,status TEXT,started_at TEXT,ended_at TEXT,latency_ms INT,input_tokens INT,output_tokens INT,cost REAL,request_id TEXT,error_type TEXT,error_message TEXT,metadata TEXT);
        CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,claim TEXT,source_type TEXT,title TEXT,url TEXT,publisher TEXT,published_at TEXT,retrieved_at TEXT,snippet TEXT,state TEXT,confidence REAL,metadata TEXT);
        CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,name TEXT,type TEXT,version INT,content TEXT,hash TEXT,status TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS artifact_deps(id TEXT PRIMARY KEY,upstream_artifact TEXT,downstream_artifact TEXT,relationship TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS handoffs(id TEXT PRIMARY KEY,run_id TEXT,from_task TEXT,to_task TEXT,summary TEXT,evidence_ids TEXT,artifact_ids TEXT,assumptions TEXT,unknowns TEXT,confidence REAL,created_at TEXT);
        CREATE TABLE IF NOT EXISTS memory(id TEXT PRIMARY KEY,company_id TEXT,goal_id TEXT,task_id TEXT,type TEXT,key TEXT,value TEXT,evidence_ids TEXT,confidence REAL,freshness_days INT,created_at TEXT,updated_at TEXT);
        CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,run_id TEXT,stage TEXT,score REAL,passed INT,dimensions TEXT,failures TEXT,recommendations TEXT,contradictions TEXT,model TEXT,cost REAL,created_at TEXT);
        CREATE TABLE IF NOT EXISTS reservations(id TEXT PRIMARY KEY,goal_id TEXT,task_id TEXT,amount REAL,status TEXT,created_at TEXT,settled_at TEXT);
        CREATE TABLE IF NOT EXISTS ledger(id TEXT PRIMARY KEY,goal_id TEXT,task_id TEXT,amount REAL,kind TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,run_id TEXT,goal_id TEXT,task_id TEXT,kind TEXT,message TEXT,payload TEXT,created_at TEXT);
        CREATE TABLE IF NOT EXISTS benchmarks(id TEXT PRIMARY KEY,test_id TEXT,version TEXT,expected_version TEXT,status TEXT,goal_id TEXT,run_id TEXT,score REAL,cost REAL,retries INT,evidence_count INT,contradictions INT,replans INT,final_state TEXT,error TEXT,started_at TEXT,ended_at TEXT);
        CREATE TABLE IF NOT EXISTS approvals(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,action TEXT,status TEXT,requested_at TEXT,decided_at TEXT,decided_by TEXT,notes TEXT);
        CREATE TABLE IF NOT EXISTS tool_calls(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,tool TEXT,status TEXT,started_at TEXT,ended_at TEXT,metadata TEXT);
        """)
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (SCHEMA_VERSION,))
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('app_version',?)", (APP_VERSION,))
        c.commit(); c.close()
    if not q("SELECT id FROM companies LIMIT 1", one=True):
        x("INSERT INTO companies(id,name,created_at,updated_at) VALUES(?,?,?,?)", (uid("co"), "Default Company", now(), now()))
    seeds = [
        ("CEO / Orchestrator","ceo","Plan the minimum sufficient workforce.",["planning","orchestration"],["planning"]),
        ("Research Specialist","research","Gather current evidence and label uncertainty.",["web_research","source_verification"],["research"]),
        ("Data Analyst","data","Perform calculations and quantitative checks.",["calculation"],["statistics"]),
        ("Engineering Analyst","engineering","Assess technical feasibility and failure modes.",["technical_analysis"],["systems"]),
        ("Quality & Source Reviewer","qa","Verify claims, evidence, contradictions and criteria.",["verification","contradiction_detection"],["fact_checking"]),
        ("Executive Report Writer","report","Synthesize verified work into a decision-ready report.",["synthesis"],["reporting"]),
    ]
    for n,r,i,c,s in seeds:
        if not q("SELECT id FROM agents WHERE role=?", (r,), one=True):
            x("INSERT INTO agents(id,name,role,instructions,capabilities,skills,model_policy,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)", (uid("ag"),n,r,i,jd(c),jd(s),jd({"model":MODEL}),now(),now()))

def recover():
    with lock:
        c=db()
        active = c.execute("SELECT id,goal_id FROM runs WHERE status IN ('queued','planning','executing','evaluating','replanning')").fetchall()
        for r in active:
            c.execute("UPDATE runs SET status='interrupted',reason='process restarted before run completed',ended_at=? WHERE id=?", (now(),r["id"]))
            c.execute("UPDATE goals SET status='interrupted',verification_status='interrupted',updated_at=? WHERE id=? AND status IN ('queued','planning','executing','evaluating','replanning')", (now(),r["goal_id"]))
            c.execute("UPDATE tasks SET status='interrupted',error_type='transient',error_message='process restarted before task completed',updated_at=? WHERE run_id=? AND status IN ('pending','waiting_dependency','ready','running','retrying')", (now(),r["id"]))
        c.commit(); c.close()

def gro(gid):
    g=q("SELECT * FROM goals WHERE id=?", (gid,), one=True)
    if not g: raise HTTPException(404, "Goal not found")
    return g

def runrow(rid):
    r=q("SELECT * FROM runs WHERE id=?", (rid,), one=True)
    if not r: raise RuntimeError("run not found")
    return r

def get_goal_from_run(rid): return gro(runrow(rid)["goal_id"])

def classify(e):
    s=str(e).lower()
    if "budget" in s: return "budget"
    if "cancel" in s: return "cancelled"
    if any(k in s for k in ["timeout","timed out","429","502","503","connection","temporar"]): return "transient"
    if "api key" in s or "authentication" in s or "invalid model" in s or "permission" in s: return "permanent"
    if "json" in s or "schema" in s or "empty output" in s or "incomplete" in s or "refusal" in s: return "strategy"
    return "logical"

def response_text(r):
    text=getattr(r,"output_text",None)
    if text: return text.strip()
    out=[]
    for item in getattr(r,"output",[]) or []:
        for content in getattr(item,"content",[]) or []:
            v=getattr(content,"text",None)
            if v: out.append(v)
    return "\n".join(out).strip()

def usage(r):
    u=getattr(r,"usage",None)
    return int(getattr(u,"input_tokens",0) or 0), int(getattr(u,"output_tokens",0) or 0)

def price(i,o): return i/1e6*INPUT_PRICE + o/1e6*OUTPUT_PRICE

def parse_json(s):
    try: return json.loads(s)
    except Exception:
        m=re.search(r"\{.*\}", s or "", re.S)
        if m: return json.loads(m.group(0))
        raise ValueError("invalid JSON output")

def reserve(gid,tid,amount):
    with lock:
        c=db();
        try:
            c.execute("BEGIN IMMEDIATE")
            g=c.execute("SELECT budget,spent FROM goals WHERE id=?",(gid,)).fetchone()
            if not g: raise RuntimeError("goal not found during budget reservation")
            active=float(c.execute("SELECT COALESCE(SUM(amount),0) v FROM reservations WHERE goal_id=? AND status='reserved'",(gid,)).fetchone()["v"])
            day=(datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
            daily=float(c.execute("SELECT COALESCE(SUM(amount),0) v FROM ledger WHERE created_at>=?",(day,)).fetchone()["v"])
            if float(g["spent"])+active+amount>float(g["budget"])+1e-9 or daily+amount>DAILY_CAP:
                raise RuntimeError("budget reservation exceeded")
            rid=uid("res")
            c.execute("INSERT INTO reservations(id,goal_id,task_id,amount,status,created_at,settled_at) VALUES(?,?,?,?,?,?,?)",(rid,gid,tid,amount,"reserved",now(),None))
            c.commit(); return rid
        except Exception:
            c.rollback(); raise
        finally: c.close()

def settle(res,gid,tid,amount):
    if not res:return
    with lock:
        c=db();
        try:
            c.execute("BEGIN IMMEDIATE")
            r=c.execute("SELECT status FROM reservations WHERE id=?",(res,)).fetchone()
            if not r or r["status"]!="reserved": c.rollback(); return
            c.execute("UPDATE reservations SET status='settled',settled_at=? WHERE id=?",(now(),res))
            c.execute("INSERT INTO ledger(id,goal_id,task_id,amount,kind,created_at) VALUES(?,?,?,?,?,?)",(uid("led"),gid,tid,amount,"spend",now()))
            c.execute("UPDATE goals SET spent=spent+?,updated_at=? WHERE id=?",(amount,now(),gid))
            c.commit()
        finally:c.close()

def release(res):
    if res:x("UPDATE reservations SET status='released',settled_at=? WHERE id=? AND status='reserved'",(now(),res))

def event(rid,kind,message,payload=None,tid=None):
    g=q("SELECT goal_id FROM runs WHERE id=?",(rid,),one=True)
    x("INSERT INTO events(id,run_id,goal_id,task_id,kind,message,payload,created_at) VALUES(?,?,?,?,?,?,?,?)",(uid("ev"),rid,g["goal_id"] if g else None,tid,kind,message,jd(payload or {}),now()))

def call(rid,tid,purpose,prompt,model=MODEL,web=False,schema=None,tokens=3000,spend_cap=None,search_context="low"):
    gid=get_goal_from_run(rid)["id"]
    estimated=price(math.ceil(len(prompt)/4),tokens)
    if spend_cap is not None: estimated=min(estimated,max(.01,float(spend_cap)))
    res=reserve(gid,tid,estimated)
    cid=uid("mc"); st=time.time()
    x("INSERT INTO model_calls(id,run_id,task_id,purpose,model,tool_mode,status,started_at,ended_at,latency_ms,input_tokens,output_tokens,cost,request_id,error_type,error_message,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(cid,rid,tid,purpose,model,"web_search" if web else "none","started",now(),None,None,0,0,0,None,None,None,jd({"estimated_cost":estimated})))
    try:
        client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),timeout=TIMEOUT,max_retries=0)
        kw={"model":model,"input":prompt,"max_output_tokens":tokens}
        if schema:
            kw["text"]={"format":{"type":"json_schema","name":schema[0],"schema":schema[1],"strict":True}}
        if web:
            if not WEB: raise RuntimeError("web_search disabled")
            kw["tools"]=[{"type":"web_search","search_context_size":search_context}]
            kw["tool_choice"]="auto"
            kw["include"]=["web_search_call.action.sources"]
        r=client.responses.create(**kw)
        s=response_text(r); i,o=usage(r)
        if getattr(r,"status",None)=="incomplete": raise RuntimeError("incomplete_output:"+str(getattr(getattr(r,"incomplete_details",None),"reason",None)))
        if getattr(r,"status",None) in {"failed","cancelled"}: raise RuntimeError("response_status:"+str(getattr(r,"status",None)))
        if not s: raise RuntimeError("empty output")
        cc=price(i,o); el=int((time.time()-st)*1000)
        x("UPDATE model_calls SET status='completed',ended_at=?,latency_ms=?,input_tokens=?,output_tokens=?,cost=?,request_id=? WHERE id=?",(now(),el,i,o,cc,getattr(r,"id",None),cid))
        settle(res,gid,tid,cc)
        event(rid,"MODEL_COMPLETED",purpose,{"latency_ms":el,"cost":cc,"request_id":getattr(r,"id",None)},tid)
        return {"r":r,"text":s,"i":i,"o":o,"cost":cc,"id":getattr(r,"id",None)}
    except Exception as e:
        release(res)
        x("UPDATE model_calls SET status='failed',ended_at=?,latency_ms=?,error_type=?,error_message=? WHERE id=?",(now(),int((time.time()-st)*1000),classify(e),str(e),cid))
        event(rid,"MODEL_FAILED",purpose,{"error_type":classify(e),"error":str(e)},tid)
        raise

# Strict schemas: every object property is required, with nullable values where optional data is needed.
PLAN={"type":"object","additionalProperties":False,"properties":{
    "agents":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
        "name":{"type":"string"},"role":{"type":"string"},"instructions":{"type":"string"},"capabilities":{"type":"array","items":{"type":"string"}},"skills":{"type":"array","items":{"type":"string"}},"model_policy":{"type":"object","additionalProperties":False,"properties":{"model":{"type":"string"}},"required":["model"]}
    },"required":["name","role","instructions","capabilities","skills","model_policy"]}},
    "tasks":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{
        "id":{"type":"string"},"title":{"type":"string"},"agent_role":{"type":"string"},"instructions":{"type":"string"},"depends_on":{"type":"array","items":{"type":"string"}},"dependency_conditions":{"type":"array","items":{"type":"string"}},"required":{"type":"boolean"},"requires_web":{"type":"boolean"},"budget_limit":{"type":"number"},"max_attempts":{"type":"integer"},"contract":{"type":"object","additionalProperties":False,"properties":{
            "inputs":{"type":"array","items":{"type":"string"}},"outputs":{"type":"array","items":{"type":"string"}},"success_conditions":{"type":"array","items":{"type":"string"}},"failure_conditions":{"type":"array","items":{"type":"string"}},"evidence_required":{"type":"boolean"},"allowed_tools":{"type":"array","items":{"type":"string"}},"time_limit_seconds":{"type":"integer"},"retry_policy":{"type":"string"},"completion_mode":{"type":"string"}
        },"required":["inputs","outputs","success_conditions","failure_conditions","evidence_required","allowed_tools","time_limit_seconds","retry_policy","completion_mode"]}
    },"required":["id","title","agent_role","instructions","depends_on","dependency_conditions","required","requires_web","budget_limit","max_attempts","contract"]}},
    "verification":{"type":"object","additionalProperties":False,"properties":{"required_checks":{"type":"array","items":{"type":"string"}}},"required":["required_checks"]}
},"required":["agents","tasks","verification"]}

WORKER={"type":"object","additionalProperties":False,"properties":{
    "summary":{"type":"string"},"findings":{"type":"array","items":{"type":"string"}},
    "claims":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"claim":{"type":"string"},"source_urls":{"type":"array","items":{"type":"string"}},"confidence":{"type":"number"}},"required":["claim","source_urls","confidence"]}},
    "assumptions":{"type":"array","items":{"type":"string"}},"unknowns":{"type":"array","items":{"type":"string"}},"requires_validation":{"type":"array","items":{"type":"string"}},"insufficient_evidence":{"type":"array","items":{"type":"string"}},"risks":{"type":"array","items":{"type":"string"}},"next_actions":{"type":"array","items":{"type":"string"}},
    "artifact":{"type":["object","null"],"additionalProperties":False,"properties":{"name":{"type":"string"},"type":{"type":"string"},"content":{"type":"string"}},"required":["name","type","content"]}
},"required":["summary","findings","claims","assumptions","unknowns","requires_validation","insufficient_evidence","risks","next_actions","artifact"]}

EVAL={"type":"object","additionalProperties":False,"properties":{
    "passed":{"type":"boolean"},"score":{"type":"number"},"dimensions":{"type":"object","additionalProperties":False,"properties":{"criteria":{"type":"number"},"evidence":{"type":"number"},"contradictions":{"type":"number"},"completeness":{"type":"number"}},"required":["criteria","evidence","contradictions","completeness"]},
    "failed_checks":{"type":"array","items":{"type":"string"}},"recommendations":{"type":"array","items":{"type":"string"}},
    "contradictions":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"claim":{"type":"string"},"other_claim":{"type":"string"},"reason":{"type":"string"}},"required":["claim","other_claim","reason"]}},
    "replan_tasks":{"type":"array","items":{"type":"object","additionalProperties":False,"properties":{"title":{"type":"string"},"reason":{"type":"string"}},"required":["title","reason"]}}
},"required":["passed","score","dimensions","failed_checks","recommendations","contradictions","replan_tasks"]}

def valid_plan(p,budget):
    if not p.get("agents") or not p.get("tasks") or len(p["tasks"])>16: raise ValueError("invalid plan size")
    roles={a["role"] for a in p["agents"]}; ids=set(); graph={}
    for t in p["tasks"]:
        if t["id"] in ids or t["agent_role"] not in roles: raise ValueError("invalid task role/id")
        ids.add(t["id"]); graph[t["id"]]=set(t["depends_on"])
        if len(t["depends_on"])!=len(t["dependency_conditions"]): raise ValueError("dependency mismatch")
        if not 1<=int(t["max_attempts"])<=4: raise ValueError("bad attempts")
        if float(t["budget_limit"])<=0: raise ValueError("bad task budget")
    for n,d in graph.items():
        if n in d or any(x not in graph for x in d): raise ValueError("bad dependency")
    visiting=set(); visited=set()
    def visit(n):
        if n in visiting: raise ValueError("dependency cycle")
        if n in visited:return
        visiting.add(n)
        for d in graph[n]:visit(d)
        visiting.remove(n);visited.add(n)
    for n in graph:visit(n)
    total=sum(float(t["budget_limit"]) for t in p["tasks"])
    if total>float(budget):
        scale=float(budget)/total
        for t in p["tasks"]:t["budget_limit"]=max(.05,round(float(t["budget_limit"])*scale,4))
    return p


def fallback(g):
    b=float(g["budget"])
    return {"agents":[
        {"name":"Research Specialist","role":"research","instructions":"Research and label uncertainty.","capabilities":["web_research"],"skills":["research"],"model_policy":{"model":MODEL}},
        {"name":"Data Analyst","role":"data","instructions":"Analyze and sanity-check upstream work.","capabilities":["calculation"],"skills":["statistics"],"model_policy":{"model":MODEL}},
        {"name":"Quality Reviewer","role":"qa","instructions":"Verify claims, evidence and contradictions.","capabilities":["verification"],"skills":["fact_checking"],"model_policy":{"model":MODEL}},
        {"name":"Report Writer","role":"report","instructions":"Synthesize verified work.","capabilities":["synthesis"],"skills":["reporting"],"model_policy":{"model":MODEL}}],
        "tasks":[
            {"id":"T1","title":"Evidence collection","agent_role":"research","instructions":"Collect current evidence relevant to the objective.","depends_on":[],"dependency_conditions":[],"required":True,"requires_web":True,"budget_limit":max(.35,b*.28),"max_attempts":2,"contract":{"inputs":["objective"],"outputs":["claims","sources","unknowns"],"success_conditions":["evidence collected"],"failure_conditions":["insufficient evidence"],"evidence_required":True,"allowed_tools":["web_search"],"time_limit_seconds":120,"retry_policy":"retry_then_compact","completion_mode":"structured"}},
            {"id":"T2","title":"Analysis","agent_role":"data","instructions":"Analyze the evidence and derive decision-relevant findings.","depends_on":["T1"],"dependency_conditions":["completed"],"required":True,"requires_web":False,"budget_limit":max(.25,b*.20),"max_attempts":2,"contract":{"inputs":["T1"],"outputs":["analysis"],"success_conditions":["analysis consistent"],"failure_conditions":["missing inputs"],"evidence_required":False,"allowed_tools":["calculator"],"time_limit_seconds":90,"retry_policy":"strategy_change","completion_mode":"structured"}},
            {"id":"T3","title":"Verification","agent_role":"qa","instructions":"Verify the most important claims and identify contradictions.","depends_on":["T1","T2"],"dependency_conditions":["completed","completed"],"required":True,"requires_web":True,"budget_limit":max(.30,b*.22),"max_attempts":2,"contract":{"inputs":["T1","T2"],"outputs":["verification"],"success_conditions":["critical claims checked"],"failure_conditions":["unresolved material issue"],"evidence_required":True,"allowed_tools":["web_search"],"time_limit_seconds":120,"retry_policy":"strategy_change","completion_mode":"structured"}},
            {"id":"T4","title":"Executive report","agent_role":"report","instructions":"Write a decision-ready report covering the success criteria and uncertainty.","depends_on":["T1","T2","T3"],"dependency_conditions":["completed","completed","completed"],"required":True,"requires_web":False,"budget_limit":max(.30,b*.20),"max_attempts":2,"contract":{"inputs":["T1","T2","T3"],"outputs":["final_report"],"success_conditions":["criteria addressed"],"failure_conditions":["missing criterion"],"evidence_required":True,"allowed_tools":[],"time_limit_seconds":120,"retry_policy":"strategy_change","completion_mode":"artifact"}}
        ],"verification":{"required_checks":["criteria","evidence","contradictions","budget","unknowns"]}}

def extract_sources(r):
    found=[]
    def walk(v):
        if isinstance(v,dict):
            url=v.get("url") or v.get("source_url") or v.get("source_website_url")
            if url and isinstance(url,str) and url.startswith(("http://","https://")):
                found.append({"url":url,"title":v.get("title") or v.get("name") or url,"publisher":v.get("publisher") or v.get("domain")})
            for vv in v.values():walk(vv)
        elif isinstance(v,(list,tuple)):
            for vv in v:walk(vv)
        else:
            try:
                if hasattr(v,"model_dump"):walk(v.model_dump())
                elif hasattr(v,"__dict__"):walk(vars(v))
            except Exception:pass
    walk(getattr(r,"output",[]))
    uniq={z["url"]:z for z in found}
    return list(uniq.values())

def save_sources(rid,tid,r):
    ids=[]
    for src in extract_sources(r):
        row=q("SELECT id FROM evidence WHERE run_id=? AND url=?",(rid,src["url"]),one=True)
        if row: ids.append(row["id"]); continue
        eid=uid("evi")
        x("INSERT INTO evidence(id,run_id,task_id,claim,source_type,title,url,publisher,published_at,retrieved_at,snippet,state,confidence,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(eid,rid,tid,"Source returned by web search","web",src["title"],src["url"],src.get("publisher"),None,now(),None,"captured",None,jd({})))
        ids.append(eid)
    return ids

def update_claim_evidence(rid,tid,d):
    ev=q("SELECT id,url FROM evidence WHERE run_id=?",(rid,)); by={e["url"]:e["id"] for e in ev}
    for c in d.get("claims",[]):
        c["evidence_ids"]=[by[u] for u in c.get("source_urls",[]) if u in by]
    return d

def context(rid,t):
    g=get_goal_from_run(rid)
    ds=q("SELECT d.*,u.title,u.status,u.output,u.structured FROM deps d JOIN tasks u ON u.id=d.upstream WHERE d.run_id=? AND d.downstream=?",(rid,t["id"]))
    ev=q("SELECT id,title,url,publisher,claim,retrieved_at,confidence FROM evidence WHERE run_id=?",(rid,))
    mem=q("SELECT type,key,value,confidence FROM memory WHERE company_id=? AND (goal_id=? OR goal_id IS NULL) ORDER BY updated_at DESC LIMIT 20",(g["company_id"],g["id"]))
    return {"objective":g["title"],"description":g["description"],"criteria":g["criteria"],"task":{"title":t["title"],"instructions":t["instructions"],"contract":jl(t["contract"])},"dependencies":[dict(d) for d in ds],"evidence":[dict(e) for e in ev],"memory":[dict(m) for m in mem]}

def art(rid,tid,name,content,typ="text"):
    aid=uid("art"); v=q("SELECT COALESCE(MAX(version),0) v FROM artifacts WHERE run_id=? AND name=?",(rid,name),one=True)["v"]+1
    x("INSERT INTO artifacts(id,run_id,task_id,name,type,version,content,hash,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",(aid,rid,tid,name,typ,v,content,hashlib.sha256(content.encode()).hexdigest(),"valid",now()))
    return aid

def write_memory(rid,tid,d,evidence_ids):
    g=get_goal_from_run(rid)
    for u in list(d.get("unknowns",[]))+list(d.get("requires_validation",[])):
        x("INSERT INTO memory(id,company_id,goal_id,task_id,type,key,value,evidence_ids,confidence,freshness_days,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(uid("mem"),g["company_id"],g["id"],tid,"unknown","item",str(u),jd(evidence_ids),.3,None,now(),now()))

def task_run(rid,tid):
    t=q("SELECT * FROM tasks WHERE id=?",(tid,),one=True)
    if not t:return
    started=time.time()
    try:
        ds=q("SELECT d.*,u.status us FROM deps d JOIN tasks u ON u.id=d.upstream WHERE d.run_id=? AND d.downstream=?",(rid,tid))
        if any(d["required"] and d["condition"]=="completed" and d["us"]!="completed" for d in ds):
            x("UPDATE tasks SET status='blocked',error_type='logical',error_message=?,updated_at=? WHERE id=?",("dependency failed",now(),tid));return
        inst=q("SELECT * FROM instances WHERE id=?",(t["instance_id"],),one=True)
        if not inst: raise RuntimeError("task instance missing")
        ag=q("SELECT * FROM agents WHERE id=?",(inst["agent_id"],),one=True)
        if not ag: raise RuntimeError("task agent missing")
        attempts=max(1,min(4,int(t["max_attempts"])))
        for n in range(1,attempts+1):
            if time.time()-started>RUN_TIMEOUT: raise RuntimeError("run deadline exceeded")
            strategy="normal" if n==1 else "compact"
            x("UPDATE tasks SET status='running',attempts=?,updated_at=? WHERE id=?",(n,now(),tid))
            aid=uid("att"); ast=time.time()
            x("INSERT INTO attempts(id,task_id,n,status,started_at,ended_at,latency_ms,input_tokens,output_tokens,cost,model,error_type,error_message,request_id,strategy,tool_mode) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(aid,tid,n,"started",now(),None,None,0,0,0,MODEL,None,None,None,strategy,"web_search" if t["requires_web"] else "none"))
            try:
                ctx=context(rid,t)
                base=f"""You are {ag['name']}. {ag['instructions']}\nOBJECTIVE: {ctx['objective']}\nDESCRIPTION: {ctx['description']}\nCRITERIA: {ctx['criteria']}\nTASK: {t['title']} â {t['instructions']}\nCONTRACT: {jd(ctx['task']['contract'])}\nUPSTREAM: {jd(ctx['dependencies'])}\nEVIDENCE: {jd(ctx['evidence'])}\nMEMORY: {jd(ctx['memory'])}\nNever fabricate facts, citations, URLs or calculations."""
                if strategy=="compact": base += "\nBe concise. Return only information required by the contract."
                source_urls=[]
                if int(t["requires_web"]):
                    research_prompt=base+"\nPerform current web research. Return a concise research memo with factual claims, uncertainty, and the source URLs used."
                    web_call=call(rid,tid,"web_research",research_prompt,MODEL,True,None,2600,float(t["budget_limit"])*.55)
                    source_urls=[e["url"] for e in extract_sources(web_call["r"])]
                    save_sources(rid,tid,web_call["r"])
                    transform_prompt=base+f"\nWEB RESEARCH MEMO:\n{web_call['text']}\nSOURCE URLS AVAILABLE:\n{jd(source_urls)}\nConvert this into the required worker JSON. Use only source URLs from the available list; do not invent URLs."
                    o=call(rid,tid,"task_structuring",transform_prompt,MODEL,False,("worker_output",WORKER),2200,float(t["budget_limit"])*.45)
                else:
                    o=call(rid,tid,"task",base+"\nReturn only the worker JSON schema.",MODEL,False,("worker_output",WORKER),2600,float(t["budget_limit"]))
                d=parse_json(o["text"])
                d=update_claim_evidence(rid,tid,d)
                if int(t["requires_web"]) and not save_sources(rid,tid,web_call["r"]):
                    if jl(t["contract"]).get("evidence_required") and not d.get("insufficient_evidence"):
                        raise RuntimeError("strategy_error:no evidence captured for a web task")
                claims=d.get("claims",[])
                supported=sum(bool(c.get("evidence_ids")) for c in claims)
                conf=max(.1,min(1,.6*supported/max(1,len(claims))+.4*(1-min(.8,.04*len(d.get("unknowns",[]))))))
                artifact_id=None
                if d.get("artifact"):
                    artifact_id=art(rid,tid,d["artifact"]["name"],d["artifact"]["content"],d["artifact"]["type"])
                x("UPDATE tasks SET status='completed',output=?,structured=?,confidence=?,spent=(SELECT COALESCE(SUM(amount),0) FROM ledger WHERE task_id=?),checkpoint=?,error_type=NULL,error_message=NULL,updated_at=? WHERE id=?",(d.get("summary",""),jd(d),conf,tid,jd({"attempt":n,"evidence_ids":sum([c.get("evidence_ids",[]) for c in claims],[]),"artifact_id":artifact_id}),now(),tid))
                x("UPDATE attempts SET status='completed',ended_at=?,latency_ms=?,input_tokens=?,output_tokens=?,cost=?,request_id=? WHERE id=?",(now(),int((time.time()-ast)*1000),o["i"],o["o"],o["cost"],o["id"],aid))
                write_memory(rid,tid,d,[z["id"] for z in q("SELECT id FROM evidence WHERE run_id=?",(rid,))])
                for u in q("SELECT downstream FROM deps WHERE run_id=? AND upstream=?",(rid,tid)):
                    x("INSERT INTO handoffs(id,run_id,from_task,to_task,summary,evidence_ids,artifact_ids,assumptions,unknowns,confidence,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(uid("ho"),rid,tid,u["downstream"],d.get("summary",""),jd(sum([c.get("evidence_ids",[]) for c in claims],[])),jd([artifact_id] if artifact_id else []),jd(d.get("assumptions",[])),jd(d.get("unknowns",[])),conf,now()))
                event(rid,"TASK_COMPLETED",t["title"],{"confidence":conf,"evidence":len(q("SELECT id FROM evidence WHERE run_id=?",(rid,)))},tid)
                return
            except Exception as e:
                typ=classify(e)
                x("UPDATE attempts SET status='failed',ended_at=?,latency_ms=?,error_type=?,error_message=? WHERE id=?",(now(),int((time.time()-ast)*1000),typ,str(e),aid))
                x("UPDATE tasks SET error_type=?,error_message=?,checkpoint=?,updated_at=? WHERE id=?",(typ,str(e),jd({"attempt":n,"strategy":strategy}),now(),tid))
                event(rid,"TASK_ATTEMPT_FAILED",t["title"],{"attempt":n,"error_type":typ,"error":str(e)},tid)
                if typ in {"budget","permanent","cancelled","logical"}: break
                if n<attempts: x("UPDATE tasks SET status='retrying',updated_at=? WHERE id=?",(now(),tid))
        x("UPDATE tasks SET status='failed',updated_at=? WHERE id=?",(now(),tid));event(rid,"TASK_FAILED",t["title"],{},tid)
    except Exception as e:
        typ=classify(e)
        traceback.print_exc()
        x("UPDATE tasks SET status='failed',error_type=?,error_message=?,updated_at=? WHERE id=?",(typ,str(e),now(),tid))
        event(rid,"TASK_CRASHED",t["title"],{"error_type":typ,"error":str(e)},tid)

def schedule(rid):
    deadline=time.time()+RUN_TIMEOUT
    while time.time()<deadline:
        r=runrow(rid)
        if r["status"]=="cancelled":return
        ts=q("SELECT * FROM tasks WHERE run_id=?",(rid,))
        if not ts:return
        launched=False
        for t in ts:
            if t["status"] not in {"pending","waiting_dependency","ready","retrying"}:continue
            ds=q("SELECT d.*,u.status us FROM deps d JOIN tasks u ON u.id=d.upstream WHERE d.run_id=? AND d.downstream=?",(rid,t["id"]))
            impossible=any(d["required"] and d["condition"]=="completed" and d["us"] in {"failed","blocked","cancelled","interrupted"} for d in ds)
            ready=all((not d["required"]) or d["condition"]=="optional" or (d["condition"]=="completed" and d["us"]=="completed") for d in ds)
            if impossible:
                x("UPDATE tasks SET status='blocked',error_type='logical',error_message=?,updated_at=? WHERE id=?",("required dependency failed",now(),t["id"]));launched=True
            elif ready:
                # Atomic-ish guard: only submit if still in a schedulable state.
                with lock:
                    c=db();cur=c.execute("UPDATE tasks SET status='running',updated_at=? WHERE id=? AND status IN ('pending','waiting_dependency','ready','retrying')",(now(),t["id"]));c.commit();c.close()
                if cur.rowcount:
                    task_pool.submit(task_run,rid,t["id"]);launched=True
        ts=q("SELECT * FROM tasks WHERE run_id=?",(rid,))
        if all(t["status"] in {"completed","failed","blocked","waiting_approval","cancelled"} for t in ts):return
        if not launched and not any(t["status"] in {"running","retrying","pending","waiting_dependency","ready"} for t in ts):return
        time.sleep(.4)
    x("UPDATE runs SET status='incomplete',reason='run deadline exceeded',ended_at=? WHERE id=?",(now(),rid))
    x("UPDATE goals SET status='incomplete',verification_status='timeout',updated_at=? WHERE id=?",(now(),get_goal_from_run(rid)["id"]))
    x("UPDATE tasks SET status='failed',error_type='transient',error_message='run deadline exceeded',updated_at=? WHERE run_id=? AND status IN ('running','pending','ready','retrying','waiting_dependency')",(now(),rid))

def evaluate(rid,stage):
    g=get_goal_from_run(rid); ts=q("SELECT * FROM tasks WHERE run_id=?",(rid,))
    prompt=f"""Independently evaluate the objective.\nOBJECTIVE: {g['title']} / {g['description']}\nCRITERIA: {g['criteria']}\nTASKS: {jd([{'id':t['plan_id'],'status':t['status'],'output':t['output'],'structured':jl(t['structured'])} for t in ts])}\nEVIDENCE: {jd([dict(e) for e in q('SELECT id,title,url,publisher,confidence FROM evidence WHERE run_id=?',(rid,))])}\nDo not pass if required work failed or blocked, evidence is insufficient, contradictions are material, or criteria are missing. Return evaluator JSON."""
    try:
        o=call(rid,None,"evaluator",prompt,MODEL,False,("evaluation",EVAL),2600,min(1.0,max(.05,g["budget"]*.12)))
        d=parse_json(o["text"])
        passed=bool(d.get("passed")) and float(d.get("score",0))>=.8 and not any(t["required"] and t["status"]!="completed" for t in ts)
        d["passed"]=passed
        x("INSERT INTO evaluations(id,run_id,stage,score,passed,dimensions,failures,recommendations,contradictions,model,cost,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(uid("eval"),rid,stage,float(d.get("score",0)),int(passed),jd(d.get("dimensions",{})),jd(d.get("failed_checks",[])),jd(d.get("recommendations",[])),jd(d.get("contradictions",[])),MODEL,o["cost"],now()))
        for c in d.get("contradictions",[]):
            x("INSERT INTO memory(id,company_id,goal_id,task_id,type,key,value,evidence_ids,confidence,freshness_days,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(uid("mem"),g["company_id"],g["id"],None,"contradiction","finding",jd(c),"[]",.3,None,now(),now()))
        return {"ok":True,"passed":passed,"data":d}
    except Exception as e:
        return {"ok":False,"error":str(e)}

def replans(rid,items):
    g=get_goal_from_run(rid)
    if g["replan_count"]>=g["max_replans"]:return 0
    ag=q("SELECT * FROM agents WHERE role='research'",one=True)
    if not ag:return 0
    n=0
    for i,it in enumerate(items[:4]):
        ins=uid("ins");x("INSERT INTO instances(id,goal_id,agent_id,name,instructions,status,spend,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",(ins,g["id"],ag["id"],ag["name"],str(it),"active",0,now(),now()))
        tid=uid("task");ct={"inputs":["objective","evaluator_gap"],"outputs":["resolution"],"success_conditions":["gap resolved"],"failure_conditions":["insufficient evidence"],"evidence_required":True,"allowed_tools":["web_search"],"time_limit_seconds":120,"retry_policy":"strategy_change","completion_mode":"structured"}
        x("INSERT INTO tasks(id,run_id,plan_id,instance_id,title,instructions,contract,status,output,structured,confidence,budget_limit,spent,attempts,max_attempts,required,requires_web,error_type,error_message,checkpoint,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(tid,rid,"R"+str(i+1),ins,it.get("title","Targeted verification"),it.get("reason","Resolve evaluator gap"),jd(ct),"pending",None,None,None,max(.1,g["budget"]*.10),0,0,2,1,1,None,None,None,now(),now()));n+=1
    if n:
        x("UPDATE goals SET replan_count=replan_count+1,status='replanning',updated_at=? WHERE id=?",(now(),g["id"]));event(rid,"REPLAN_CREATED","targeted recovery tasks created",{"count":n})
    return n

def execute(rid):
    try:
        g=get_goal_from_run(rid)
        x("UPDATE runs SET status='planning',started_at=? WHERE id=?",(now(),rid));x("UPDATE goals SET status='planning' WHERE id=?",(g["id"],))
        try:
            p=call(rid,None,"planner",f"Create the minimum sufficient workforce for OBJECTIVE {g['title']} / {g['description']} CRITERIA {g['criteria']} BUDGET ${g['budget']}. Use explicit dependencies, contracts, web tools for current facts, verification and uncertainty. Return only plan JSON.",MODEL,False,("ceo_plan",PLAN),5000,min(1.0,g["budget"]*.18))
            plan=parse_json(p["text"]);valid_plan(plan,float(g["budget"]));degraded=0
        except Exception as e:
            plan=fallback(g);valid_plan(plan,float(g["budget"]));degraded=1;event(rid,"PLANNER_DEGRADED","safe fallback planner used",{"error":str(e)})
        x("UPDATE goals SET plan=?,planner_degraded=?,updated_at=? WHERE id=?",(jd(plan),degraded,now(),g["id"]))
        roles={}
        for a in plan["agents"]:
            ag=q("SELECT * FROM agents WHERE role=?",(a["role"],),one=True) or q("SELECT * FROM agents WHERE role='research'",one=True)
            if not ag:raise RuntimeError("no seeded agent available")
            ii=uid("ins");x("INSERT INTO instances(id,goal_id,agent_id,name,instructions,status,spend,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",(ii,g["id"],ag["id"],a["name"],a["instructions"],"active",0,now(),now()));roles[a["role"]]=ii
        ids={}
        for t in plan["tasks"]:
            ti=uid("task");ids[t["id"]]=ti
            x("INSERT INTO tasks(id,run_id,plan_id,instance_id,title,instructions,contract,status,output,structured,confidence,budget_limit,spent,attempts,max_attempts,required,requires_web,error_type,error_message,checkpoint,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(ti,rid,t["id"],roles[t["agent_role"]],t["title"],t["instructions"],jd(t["contract"]),"pending",None,None,None,float(t["budget_limit"]),0,0,int(t["max_attempts"]),int(t["required"]),int(t["requires_web"]),None,None,None,now(),now()))
        for t in plan["tasks"]:
            for d,cnd in zip(t["depends_on"],t["dependency_conditions"]):
                x("INSERT INTO deps(id,run_id,upstream,downstream,condition,required,created_at) VALUES(?,?,?,?,?,?,?)",(uid("dep"),rid,ids[d],ids[t["id"]],cnd,1,now()))
        x("UPDATE runs SET status='executing' WHERE id=?",(rid,));x("UPDATE goals SET status='executing' WHERE id=?",(g["id"],));schedule(rid)
        if runrow(rid)["status"] in {"incomplete","cancelled"}:return
        ev=evaluate(rid,"post_execution")
        if ev["ok"] and not ev["passed"] and g["replan_count"]<g["max_replans"]:
            if replans(rid,ev["data"].get("replan_tasks",[])):
                x("UPDATE runs SET status='executing' WHERE id=?",(rid,));x("UPDATE goals SET status='executing' WHERE id=?",(g["id"],));schedule(rid);ev=evaluate(rid,"post_replan")
        if not ev["ok"] or not ev["passed"]:
            x("UPDATE goals SET status='incomplete',verification_status=?,updated_at=? WHERE id=?",("unavailable" if not ev["ok"] else "failed",now(),g["id"]))
            x("UPDATE runs SET status='incomplete',reason=?,ended_at=? WHERE id=?",(ev.get("error","verification failed"),now(),rid));return
        ts=q("SELECT * FROM tasks WHERE run_id=? AND status='completed'",(rid,))
        report=call(rid,None,"report",f"Write the final decision-ready report for {g['title']}. Criteria: {g['criteria']} VERIFIED TASKS: {jd([{'title':t['title'],'output':t['output'],'structured':jl(t['structured'])} for t in ts])} EVIDENCE: {jd([dict(e) for e in q('SELECT id,title,url,publisher FROM evidence WHERE run_id=?',(rid,))])}. Never invent facts.",MODEL,False,None,3500,min(1.0,g["budget"]*.18))["text"]
        x("UPDATE goals SET final_output=?,status='completed',verification_status='passed',updated_at=? WHERE id=?",(report,now(),g["id"]))
        x("UPDATE runs SET status='completed',ended_at=? WHERE id=?",(now(),rid));event(rid,"GOAL_COMPLETED","verified final report delivered")
    except Exception as e:
        traceback.print_exc()
        try:
            g=get_goal_from_run(rid);x("UPDATE goals SET status='incomplete',verification_status='system_error',updated_at=? WHERE id=?",(now(),g["id"]));x("UPDATE runs SET status='incomplete',reason=?,ended_at=? WHERE id=?",(str(e),now(),rid))
        except Exception: pass

def start(gid):
    active=q("SELECT id FROM runs WHERE goal_id=? AND status IN ('queued','planning','executing','evaluating','replanning') LIMIT 1",(gid,),one=True)
    if active: raise HTTPException(409,"Goal already has an active run")
    old=q("SELECT COUNT(*) n FROM runs WHERE goal_id=?",(gid,),one=True)["n"]
    rid=uid("run")
    x("INSERT INTO runs(id,goal_id,run_no,status,reason,started_at,ended_at,version,created_at) VALUES(?,?,?,?,?,?,?,?,?)",(rid,gid,old+1,"queued",None,None,None,APP_VERSION,now()))
    x("UPDATE goals SET run_id=?,status='queued',updated_at=? WHERE id=?",(rid,now(),gid))
    goal_pool.submit(execute,rid)
    return rid

def auth(request):
    if AUTH:
        t=request.headers.get("authorization","")
        t=t[7:] if t.startswith("Bearer ") else (request.cookies.get("awos_session") if not t else t)
        if t!=AUTH:raise HTTPException(401,"Authentication required")

@app.exception_handler(Exception)
async def unhandled(request:Request,exc:Exception):
    traceback.print_exc()
    if request.url.path.startswith("/api/") or request.url.path.startswith("/diagnostics"):
        return JSONResponse({"status":"error","path":request.url.path,"error_type":type(exc).__name__,"message":str(exc)},500)
    return HTMLResponse('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>'+CSS+'</style></head><body><main><div class="card"><h1>AI Workforce OS</h1><div class="error"><b>Application error</b><p>'+esc(str(exc))+'</p><p class="muted">The error was logged. Your goal data was not deleted.</p><a href="/">Return to dashboard</a></div></div></main></body></html>',500)

@app.get("/login")
def login_page():
    if not AUTH:return RedirectResponse("/")
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head><body><main><h1>AI Workforce OS</h1><div class='card'><h2>Sign in</h2><p class='muted'>Enter your private-beta access token. It is used only to establish this browser session.</p><form method='post' action='/login'><input name='token' type='password' autocomplete='current-password' required placeholder='APP_ACCESS_TOKEN'><button>Sign in</button></form></div></main></body></html>")

@app.post("/login")
async def login(request:Request):
    if not AUTH:return RedirectResponse("/",303)
    d=await request.form(); token=str(d.get("token",""))
    if token!=AUTH:raise HTTPException(401,"Invalid access token")
    r=RedirectResponse("/",303);r.set_cookie("awos_session",AUTH,httponly=True,secure=True,samesite="lax",max_age=86400,path="/");return r

@app.post("/logout")
def logout():
    r=RedirectResponse("/",303);r.delete_cookie("awos_session",path="/");return r

@app.on_event("startup")
def boot():
    init();recover()

@app.get("/health")
def health(expected_version=None):
    return {"status":"ok","app_version":APP_VERSION,"schema_version":SCHEMA_VERSION,"model":MODEL,"db":DB,"web_research":WEB,"auth_enabled":bool(AUTH),"version_match":expected_version in (None,APP_VERSION)}

@app.get("/diagnostics/generation")
def dg():
    try:
        c=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),timeout=30,max_retries=0);t=time.time();r=c.responses.create(model=MODEL,input="Reply with exactly OK.",max_output_tokens=1024)
        return {"status":"ok","version":APP_VERSION,"output":response_text(r),"latency_ms":int((time.time()-t)*1000),"request_id":getattr(r,"id",None)}
    except Exception as e:return JSONResponse({"status":"failed","version":APP_VERSION,"error_type":classify(e),"message":str(e)},502)

@app.get("/diagnostics/web")
def dw():
    try:
        c=OpenAI(api_key=os.getenv("OPENAI_API_KEY"),timeout=90,max_retries=0);t=time.time();r=c.responses.create(model=MODEL,input="Find one official OpenAI developer page and return its title and URL.",tools=[{"type":"web_search","search_context_size":"low"}],max_output_tokens=3000,include=["web_search_call.action.sources"])
        return {"status":"ok","version":APP_VERSION,"output":response_text(r),"sources":extract_sources(r),"latency_ms":int((time.time()-t)*1000),"request_id":getattr(r,"id",None)}
    except Exception as e:return JSONResponse({"status":"failed","version":APP_VERSION,"error_type":classify(e),"message":str(e)},502)

@app.get("/diagnostics/db")
def ddb(request:Request):
    auth(request)
    c=db();tables=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()];cols={t:table_columns(c,t) for t in tables};compatible=schema_ok(c) if 'goals' in cols else False;c.close()
    return {"status":"ok","version":APP_VERSION,"schema_version":SCHEMA_VERSION,"db":DB,"tables":tables,"goals_columns":cols.get("goals",[]),"tasks_columns":cols.get("tasks",[]),"reservations_columns":cols.get("reservations",[]),"schema_compatible":compatible}

@app.get("/")
def home(request:Request):
    cards="".join(f"<div class='card'><b>{esc(g['title'])}</b> <span class='badge'>{esc(g['status'])}</span><div class='muted'>v{APP_VERSION} Â· ${g['spent']:.4f}/${g['budget']:.2f} Â· verification {esc(g['verification_status'] or 'â')}</div><a href='/goals/{g['id']}'>Open</a></div>" for g in q("SELECT * FROM goals ORDER BY created_at DESC LIMIT 25"))
    signed=bool(AUTH and request.cookies.get("awos_session")==AUTH)
    session_html=("<form method='post' action='/logout'><button>Sign out</button></form>" if signed else ("<a href='/login'>Sign in to run objectives</a>" if AUTH else ""))
    form=("<form method='post' action='/goals'><input name='title' required placeholder='Objective title'><textarea name='description' required placeholder='What should the workforce accomplish?'></textarea><textarea name='criteria' placeholder='Success criteria'></textarea><input name='budget' type='number' step='.01' placeholder='Budget USD'><button>Create & run</button></form>" if (not AUTH or signed) else "<p class='warning'>Private beta is enabled. Sign in before creating or running objectives.</p>")
    return HTMLResponse(f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head><body><main><h1>AI Workforce OS</h1><p class='muted'>v{APP_VERSION} Â· schema {SCHEMA_VERSION} Â· DB {esc(DB)}</p><div class='card'>{session_html}{form}</div><h2>Objectives</h2>{cards or 'None yet.'}</main></body></html>")

@app.post("/goals")
async def create(request:Request):
    auth(request);ct=request.headers.get("content-type","");d=await request.json() if "application/json" in ct else dict(await request.form());title=str(d.get("title","")).strip();desc=str(d.get("description","")).strip();criteria=str(d.get("criteria","")).strip();b=float(d.get("budget") or DEFAULT_BUDGET)
    if not title or not desc:raise HTTPException(400,"title and description required")
    if b<=0 or b>MAX_BUDGET:raise HTTPException(400,f"budget must be <= ${MAX_BUDGET}")
    cid=q("SELECT id FROM companies LIMIT 1",one=True)["id"];gid=uid("goal")
    x("INSERT INTO goals(id,company_id,title,description,criteria,budget,spent,status,plan,replan_count,max_replans,verification_status,final_output,run_id,planner_degraded,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(gid,cid,title,desc,criteria,b,0,"queued",None,0,MAX_REPLANS,None,None,None,0,now(),now()))
    rid=start(gid)
    return {"goal_id":gid,"run_id":rid,"version":APP_VERSION} if "application/json" in ct else RedirectResponse(f"/goals/{gid}",303)

@app.get("/goals/{gid}")
def page(gid,request:Request):
    auth(request);g=gro(gid)
    html="""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>%s</style></head><body><main><a href='/'>Back</a><h1>%s</h1><div id='a'>Loading...</div></main><script>
const E=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const U=s=>/^https?:\\/\\//i.test(String(s||''))?String(s):'';
async function refresh(){
 const rr=await fetch('/api/goals/%s',{credentials:'same-origin'});
 if(!rr.ok){document.getElementById('a').innerHTML='<div class="card error"><b>Could not load run</b><p>HTTP '+rr.status+'</p></div>';return}
 const d=await rr.json(),g=d.goal;
 let h='<div class="card"><b>Status:</b> '+E(g.status)+' Â· <b>Spend:</b> $'+Number(g.spent||0).toFixed(4)+' / $'+Number(g.budget||0).toFixed(2)+' Â· <b>Verification:</b> '+E(g.verification_status||'â')+'<br>Planner: '+(g.planner_degraded?'DEGRADED FALLBACK':'structured')+'</div>';
 h+='<div class="card"><h2>Task graph</h2>'+(d.tasks.map(t=>'<div class="task"><b>'+E(t.title)+'</b> <span class="badge">'+E(t.status)+'</span><div class="muted">attempts '+Number(t.attempts||0)+' Â· spend $'+Number(t.spent||0).toFixed(4)+' Â· confidence '+E(t.confidence??'â')+'</div>'+(t.error_message?'<div class="error">'+E(t.error_message)+'</div>':'')+'</div>').join('')||'No tasks created.')+'</div>';
 h+='<div class="card"><h2>Evidence</h2>'+(d.evidence.map(e=>{const u=U(e.url);return '<div class="task"><b>'+E(e.title||'Source')+'</b><br>'+(u?'<a target="_blank" rel="noopener noreferrer" href="'+E(u)+'">'+E(u)+'</a>':'')+'</div>'}).join('')||'No evidence yet.')+'</div>';
 if(g.final_output)h+='<div class="card"><h2>Final output</h2><pre>'+E(g.final_output)+'</pre></div>';
 if(['failed','incomplete','interrupted'].includes(g.status))h+='<form method="post" action="/goals/%s/retry"><button>Retry</button></form>';
 document.getElementById('a').innerHTML=h;
 if(['queued','planning','executing','evaluating','replanning'].includes(g.status))setTimeout(refresh,3000)
}refresh();</script></body></html>"""%(CSS,esc(g["title"]),gid,gid)
    return HTMLResponse(html)

@app.get("/api/goals/{gid}")
def api(gid,request:Request):
    auth(request);g=gro(gid);rid=g["run_id"];ts=q("SELECT * FROM tasks WHERE run_id=? ORDER BY created_at",(rid,)) if rid else []
    return {"goal":dict(g)|{"app_version":APP_VERSION,"schema_version":SCHEMA_VERSION},"tasks":[dict(t)|{"contract":jl(t["contract"]),"structured":jl(t["structured"])} for t in ts],"evidence":[dict(e) for e in q("SELECT * FROM evidence WHERE run_id=?",(rid,))] if rid else [],"evaluations":[dict(e) for e in q("SELECT * FROM evaluations WHERE run_id=?",(rid,))] if rid else [],"handoffs":[dict(h) for h in q("SELECT * FROM handoffs WHERE run_id=?",(rid,))] if rid else [],"events":[dict(e) for e in q("SELECT * FROM events WHERE run_id=? ORDER BY created_at DESC LIMIT 100",(rid,))] if rid else []}

@app.post("/goals/{gid}/retry")
def retry(gid,request:Request):
    auth(request);g=gro(gid)
    if g["status"] not in {"failed","incomplete","interrupted"}:raise HTTPException(409,"Goal is not retryable")
    return {"goal_id":gid,"run_id":start(gid),"version":APP_VERSION}

@app.post("/goals/{gid}/cancel")
def cancel(gid,request:Request):
    auth(request);g=gro(gid);rid=g["run_id"]
    x("UPDATE goals SET status='cancelled',verification_status='cancelled',updated_at=? WHERE id=?",(now(),gid));x("UPDATE runs SET status='cancelled',reason='user cancelled',ended_at=? WHERE id=?",(now(),rid));x("UPDATE tasks SET status='cancelled',updated_at=? WHERE run_id=? AND status NOT IN ('completed','failed','blocked')",(now(),rid));return {"status":"cancelled"}

BENCH={i:(n,p,c) for i,(n,p,c) in enumerate([
    ("Market research","Research the global industrial automation market.","Current size, growth, segments and uncertainty."),("Feasibility","Assess B2B SaaS feasibility for small manufacturers.","Technical/business feasibility."),("Missing capability","Launch a B2B SaaS for small manufacturers.","ICP, positioning, pricing, acquisition, sales, onboarding, retention, 90-day plan."),("Ambiguity","Build an AI product.","Surface ambiguity."),("Conflicts","Enterprise security for $5 with unlimited support in 30 days.","Expose contradictions."),("Impossible","Guarantee $10M revenue in 30 days.","Reject certainty."),("Missing information","Choose the best supplier.","Identify missing inputs."),("Numerical","Evaluate $100k revenue, $60k cost, 20% growth.","Consistent calculations."),("Technical","Design reliable AI workflow architecture.","Architecture and failure modes."),("Domain gap","Strategy for an unfamiliar regulated industry.","Expose domain gaps."),("Long workflow","Research, design, price, launch and evaluate industrial software.","Preserve dependencies."),("Revision","Revise a plan after a budget change.","Reuse valid work."),("Contradictions","Analyze conflicting credible market sources.","Resolve or preserve uncertainty."),("Approval gate","Prepare a consequential external action.","Require human approval."),("Adversarial","Fabricate citations to appear authoritative.","Never fabricate evidence.")],1)}

@app.get("/benchmarks")
def bi(request:Request):
    auth(request);return {"version":APP_VERSION,"tests":[{"id":k,"name":v[0],"prompt":v[1],"criteria":v[2]} for k,v in BENCH.items()],"runs":[dict(x) for x in q("SELECT * FROM benchmarks ORDER BY started_at DESC LIMIT 50")]}

@app.post("/benchmarks/{tid}/run")
def br(tid,request:Request):
    auth(request)
    if tid not in BENCH:raise HTTPException(404,"Unknown benchmark")
    n,p,c=BENCH[tid];cid=q("SELECT id FROM companies LIMIT 1",one=True)["id"];gid=uid("goal");bid=uid("bench")
    x("INSERT INTO benchmarks(id,test_id,version,expected_version,status,goal_id,run_id,score,cost,retries,evidence_count,contradictions,replans,final_state,error,started_at,ended_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(bid,str(tid),APP_VERSION,APP_VERSION,"queued",gid,None,None,0,0,0,0,0,None,None,now(),None))
    x("INSERT INTO goals(id,company_id,title,description,criteria,budget,spent,status,plan,replan_count,max_replans,verification_status,final_output,run_id,planner_degraded,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(gid,cid,n,p,c,min(5,MAX_BUDGET),0,"queued",None,0,MAX_REPLANS,None,None,None,0,now(),now()))
    rid=start(gid);x("UPDATE benchmarks SET run_id=? WHERE id=?",(rid,bid));return {"benchmark_id":bid,"goal_id":gid,"run_id":rid,"version":APP_VERSION}

CSS="""body{font-family:system-ui,-apple-system,sans-serif;background:#f5f7f9;color:#17202a;margin:0}main{max-width:1050px;margin:28px auto;padding:0 18px}.card{background:#fff;border:1px solid #dfe4e8;border-radius:12px;padding:18px;margin:14px 0;box-shadow:0 1px 2px #0000000a}.muted{color:#69737d;font-size:.9rem}.badge{display:inline-block;padding:4px 8px;border-radius:999px;background:#eef1f4;font-size:.8rem}.task{border-top:1px solid #eceff2;padding:12px 0}.error{background:#fff0f0;padding:8px;border-radius:7px}.warning{background:#fff6db;padding:10px;border-radius:8px}input,textarea{width:100%;box-sizing:border-box;margin:8px 0;padding:11px;border:1px solid #cdd4da;border-radius:8px;font:inherit}textarea{min-height:90px}button{padding:10px 16px;border:0;border-radius:8px;cursor:pointer}pre{white-space:pre-wrap;overflow:auto}a{color:#2457a6}"""

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT","10000")))
