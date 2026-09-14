import os, sqlite3, uuid, json, html
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse

DB = Path(__file__).with_name("workforce_v02.db")
app = FastAPI(title="AI Workforce OS", version="0.2.1")

def now(): return datetime.now(timezone.utc).isoformat()
def uid(): return str(uuid.uuid4())
def esc(x): return html.escape(str(x or ""))

def db():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def fetch(sql, args=()):
    c=db(); rows=c.execute(sql,args).fetchall(); c.close(); return rows

def write(sql,args=()):
    c=db(); c.execute(sql,args); c.commit(); c.close()

def init():
    c=db()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS company(id TEXT PRIMARY KEY,name TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS agents(id TEXT PRIMARY KEY,name TEXT NOT NULL,role TEXT UNIQUE NOT NULL,instructions TEXT NOT NULL,capabilities TEXT NOT NULL,status TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY,title TEXT NOT NULL,description TEXT NOT NULL,criteria TEXT NOT NULL,budget REAL NOT NULL,status TEXT NOT NULL,plan TEXT,final_output TEXT,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,goal_id TEXT NOT NULL,agent_id TEXT NOT NULL,title TEXT NOT NULL,instructions TEXT NOT NULL,status TEXT NOT NULL,output TEXT,confidence REAL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS handoffs(id TEXT PRIMARY KEY,goal_id TEXT NOT NULL,from_agent_id TEXT NOT NULL,to_agent_id TEXT NOT NULL,summary TEXT NOT NULL,evidence TEXT NOT NULL,confidence REAL,created_at TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,goal_id TEXT,kind TEXT NOT NULL,message TEXT NOT NULL,created_at TEXT NOT NULL);
    """)
    if not c.execute("SELECT id FROM company LIMIT 1").fetchone():
        c.execute("INSERT INTO company VALUES(?,?)",(uid(),"My AI Company"))
        seed=[
        ("CEO","CEO / Orchestrator","Turn objectives into plans, delegate work, monitor progress, and synthesize results.","plan,delegate,review,synthesize"),
        ("Research Agent","Research Specialist","Identify facts, evidence requirements, assumptions, gaps, and research questions.","research,evidence,verification"),
        ("Engineering Agent","Engineering Analyst","Analyze technical requirements, constraints, compatibility, risks, and alternatives.","technical_analysis,risk,verification"),
        ("Procurement Agent","Procurement Analyst","Analyze sourcing, supply, availability, commercial risks, and mitigations.","sourcing,supply,risk"),
        ("Report Agent","Executive Report Writer","Synthesize specialist work into a concise decision-ready executive report.","synthesis,report")]
        for n,r,i,cap in seed:
            c.execute("INSERT INTO agents VALUES(?,?,?,?,?,?)",(uid(),n,r,i,cap,"idle"))
    c.commit(); c.close()

def agent(role): return fetch("SELECT * FROM agents WHERE role=?",(role,))[0]
def log(gid,kind,msg): write("INSERT INTO events VALUES(?,?,?,?,?)",(uid(),gid,kind,msg,now()))

def model(system,prompt):
    key=os.getenv("OPENAI_API_KEY")
    model_name=os.getenv("OPENAI_MODEL","gpt-5-mini")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    try:
        from openai import OpenAI
        print(f"OPENAI_CALL_START model={model_name}", flush=True)
        client=OpenAI(api_key=key, timeout=30.0, max_retries=0)
        response=client.responses.create(
            model=model_name,
            input=f"SYSTEM:\n{system}\n\nUSER:\n{prompt}")
        print(f"OPENAI_CALL_OK model={model_name} request_id={getattr(response, '_request_id', None)}", flush=True)
        return response.output_text
    except Exception as e:
        print(f"OPENAI_CALL_ERROR type={type(e).__name__} message={e}", flush=True)
        raise

def make_plan(g):
    live=model("""You are the CEO of an AI company. Return ONLY valid JSON:
{"tasks":[{"title":"string","instructions":"string","role":"string"}]}
Allowed roles: Research Specialist, Engineering Analyst, Procurement Analyst, Executive Report Writer.
Create 3-6 useful tasks; the final task must be Executive Report Writer.""",
        f"Objective: {g['title']}\nDescription: {g['description']}\nSuccess: {g['criteria']}\nBudget: ${g['budget']:.2f}")
    try:
        d=json.loads(live[live.find("{"):live.rfind("}")+1])
        allowed={"Research Specialist","Engineering Analyst","Procurement Analyst","Executive Report Writer"}
        tasks=[x for x in d.get("tasks",[]) if x.get("role") in allowed and x.get("title") and x.get("instructions")]
        if tasks:
            if tasks[-1]["role"]!="Executive Report Writer":
                tasks.append({"title":"Prepare final executive report","instructions":"Synthesize all work into conclusions, risks, uncertainties and recommendations.","role":"Executive Report Writer"})
            return {"tasks":tasks}
    except Exception as e:
        print(f"PLAN_PARSE_ERROR type={type(e).__name__} message={e}", flush=True)
        raise RuntimeError("CEO returned invalid plan JSON") from e
    raise RuntimeError("CEO returned no usable plan")

def specialist(a,item,context):
    live=model(a["instructions"],f"Task: {item['title']}\nInstructions: {item['instructions']}\n\nPrior findings:\n{context}\n\nReturn summary, key findings, evidence/assumptions, uncertainties, and next action.")
    if live: return live,.75
    raise RuntimeError(f"{a['name']} returned no output")

def execute_goal(gid):
    g=fetch("SELECT * FROM goals WHERE id=?",(gid,))[0]
    ceo=agent("CEO / Orchestrator")
    write("UPDATE goals SET status=? WHERE id=?",("planning",gid))
    write("UPDATE agents SET status=? WHERE id=?",("working",ceo["id"]))
    log(gid,"planning","CEO started planning the objective.")
    try:
        plan=make_plan(g)
        write("UPDATE goals SET status=?,plan=? WHERE id=?",("executing",json.dumps(plan,indent=2),gid))
        log(gid,"plan",f"CEO created {len(plan['tasks'])} work items.")
        context=[]; final=None
        for item in plan["tasks"]:
            a=agent(item["role"]); tid=uid()
            write("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)",(tid,gid,a["id"],item["title"],item["instructions"],"running",None,None,now()))
            write("UPDATE agents SET status=? WHERE id=?",("working",a["id"]))
            log(gid,"task_started",f"{a['name']} started: {item['title']}")
            out,conf=specialist(a,item,"\n\n".join(context[-5:]))
            write("UPDATE tasks SET status=?,output=?,confidence=? WHERE id=?",("completed",out,conf,tid))
            write("UPDATE agents SET status=? WHERE id=?",("idle",a["id"]))
            log(gid,"task_completed",f"{a['name']} completed: {item['title']}")
            context.append(f"[{a['name']}]\n{out}")
            if item["role"]!="Executive Report Writer":
                report=agent("Executive Report Writer")
                write("INSERT INTO handoffs VALUES(?,?,?,?,?,?,?,?)",(uid(),gid,a["id"],report["id"],f"Completed {item['title']}",out,conf,now()))
                log(gid,"handoff",f"{a['name']} handed findings to Report Agent.")
            else: final=out
        write("UPDATE goals SET status=?,final_output=? WHERE id=?",("completed",final or "No final output produced.",gid))
        log(gid,"completed","CEO completed the objective.")
    except Exception as e:
        message=f"{type(e).__name__}: {e}"
        write("UPDATE goals SET status=?,final_output=? WHERE id=?",("failed",f"EXECUTION FAILED\n\n{message}\n\nCheck Render logs for OPENAI_CALL_ERROR details.",gid))
        log(gid,"failed",f"Execution failed: {message}")
        print(f"GOAL_FAILED gid={gid} {message}", flush=True)
    finally:
        write("UPDATE agents SET status=? WHERE status=?",("idle","working"))

@app.get("/",response_class=HTMLResponse)
def home():
    gs=fetch("SELECT * FROM goals ORDER BY created_at DESC"); ag=fetch("SELECT * FROM agents ORDER BY role"); ev=fetch("SELECT * FROM events ORDER BY created_at DESC LIMIT 40")
    goals="".join(f"<div class='row'><a href='/goals/{g['id']}'><b>{esc(g['title'])}</b></a><div class='muted'>{esc(g['status'])} Â· ${g['budget']:.2f} budget</div></div>" for g in gs) or "<p class='muted'>No objectives yet.</p>"
    events="".join(f"<div class='row'><small>{esc(e['created_at'][11:19])}</small> {esc(e['message'])}</div>" for e in ev) or "<p class='muted'>Activity will appear here.</p>"
    agents="".join(f"<div class='agent'><b>{esc(a['name'])}</b><div class='muted'>{esc(a['role'])}</div><p>{esc(a['instructions'])}</p><small>Status: {esc(a['status'])}</small></div>" for a in ag)
    return f"""<!doctype html><meta name=viewport content='width=device-width,initial-scale=1'><title>AI Workforce OS v0.2.1</title>
<style>body{{margin:0;background:#f4f5f7;color:#15171a;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:#111318;color:#fff;padding:18px 22px}}main{{max-width:1200px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}section,.agent{{background:#fff;border:1px solid #dfe3e7;border-radius:14px;padding:16px}}.agents{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}input,textarea,button{{width:100%;box-sizing:border-box;padding:11px;margin:7px 0;border-radius:9px;font:inherit}}input,textarea{{border:1px solid #ccd2d8}}textarea{{min-height:90px}}button{{background:#111318;color:#fff;border:0;font-weight:600}}.row{{padding:10px 0;border-bottom:1px solid #eceef1}}.muted,small{{color:#69717c}}a{{color:#145ac6;text-decoration:none}}@media(max-width:720px){{.grid{{grid-template-columns:1fr}}main{{padding:12px}}}}</style>
<header><b>AI WORKFORCE OS</b> Â· v0.2.1</header><main><div class=grid>
<section><h2>Give the company an objective</h2><form method=post action=/goals><input name=title placeholder='Objective title' required><textarea name=description placeholder='Describe what you want the AI company to accomplish.' required></textarea><textarea name=criteria placeholder='What does success look like?' required></textarea><input name=budget type=number min=0 step=.01 value=5 required><button>Create objective and run</button></form><h2>Objectives</h2>{goals}</section>
<section><h2>Activity</h2>{events}</section></div><section style='margin-top:16px'><h2>AI workforce</h2><div class=agents>{agents}</div></section></main>"""

@app.post("/goals")
def create_goal(title:str=Form(...),description:str=Form(...),criteria:str=Form(...),budget:float=Form(...)):
    gid=uid()
    write("INSERT INTO goals VALUES(?,?,?,?,?,?,?,?,?)",(gid,title,description,criteria,budget,"queued",None,None,now()))
    log(gid,"goal_created",f"Human assigned objective: {title}")
    execute_goal(gid)
    return RedirectResponse(f"/goals/{gid}",303)

@app.get("/goals/{gid}",response_class=HTMLResponse)
def detail(gid:str):
    g=fetch("SELECT * FROM goals WHERE id=?",(gid,))[0]
    ts=fetch("SELECT t.*,a.name agent_name FROM tasks t JOIN agents a ON a.id=t.agent_id WHERE t.goal_id=? ORDER BY t.created_at",(gid,))
    hs=fetch("SELECT h.*,a.name from_name,b.name to_name FROM handoffs h JOIN agents a ON a.id=h.from_agent_id JOIN agents b ON b.id=h.to_agent_id WHERE h.goal_id=? ORDER BY h.created_at",(gid,))
    tasks="".join(f"<details><summary><b>{esc(t['title'])}</b> â {esc(t['agent_name'])}</summary><pre>{esc(t['output'])}</pre></details>" for t in ts)
    hands="".join(f"<div class=row><b>{esc(h['from_name'])}</b> â <b>{esc(h['to_name'])}</b><br><small>Confidence: {h['confidence']}</small><pre>{esc(h['evidence'])}</pre></div>" for h in hs)
    return f"""<!doctype html><meta name=viewport content='width=device-width'><title>{esc(g['title'])}</title><style>body{{margin:0;background:#f4f5f7;font-family:-apple-system,system-ui}}main{{max-width:1000px;margin:auto;padding:18px}}section{{background:#fff;border:1px solid #dfe3e7;border-radius:14px;padding:16px;margin:14px 0}}pre{{white-space:pre-wrap;background:#f7f8fa;padding:12px;border-radius:9px;overflow:auto}}.row{{padding:10px 0;border-bottom:1px solid #eee}}a{{color:#145ac6;text-decoration:none}}</style><main><a href='/'>â Workforce dashboard</a><h1>{esc(g['title'])}</h1><p>{esc(g['description'])}</p><section><b>Status:</b> {esc(g['status'])}<br><b>Success:</b> {esc(g['criteria'])}<br><b>Budget:</b> ${g['budget']:.2f}</section><section><h2>CEO plan</h2><pre>{esc(g['plan'])}</pre></section><section><h2>Work completed</h2>{tasks}</section><section><h2>Agent handoffs</h2>{hands or '<p>No handoffs.</p>'}</section><section><h2>Final output</h2><pre>{esc(g['final_output'])}</pre></section></main>"""

@app.get("/api/goals/{gid}")
def api_goal(gid:str):
    g=fetch("SELECT * FROM goals WHERE id=?",(gid,))
    if not g: return JSONResponse({"error":"not found"},404)
    ts=fetch("SELECT * FROM tasks WHERE goal_id=?",(gid,))
    return {"goal":dict(g[0]),"tasks":[dict(t) for t in ts]}

@app.get("/health")
def health(): return {"status":"ok","version":"0.2.1"}

init()
