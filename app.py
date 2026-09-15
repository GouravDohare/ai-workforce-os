import os,re,json,html,time,uuid,sqlite3,threading
from datetime import datetime,timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI,Form
from fastapi.responses import HTMLResponse,RedirectResponse,JSONResponse
from openai import OpenAI

APP_VERSION='0.3.13'; SCHEMA_VERSION='313-1'
DB=Path(__file__).resolve().parent/os.getenv('WORKFORCE_DB','workforce_v0313.db')
MODEL=os.getenv('OPENAI_MODEL','gpt-5-mini'); TIMEOUT=float(os.getenv('OPENAI_TIMEOUT_SECONDS','150'))
MAX_RETRIES=int(os.getenv('OPENAI_MAX_RETRIES','1')); DEFAULT_BUDGET=float(os.getenv('DEFAULT_GOAL_BUDGET_USD','5.0'))
MAX_BUDGET=float(os.getenv('MAX_GOAL_BUDGET_USD','50.0')); WEB=os.getenv('ENABLE_WEB_RESEARCH','true').lower()=='true'
executor=ThreadPoolExecutor(max_workers=int(os.getenv('MAX_CONCURRENT_TASKS','3'))); app=FastAPI(title='AI Workforce OS',version=APP_VERSION)

def now(): return datetime.now(timezone.utc).isoformat()
def uid(): return str(uuid.uuid4())
def db():
 c=sqlite3.connect(DB,timeout=30,check_same_thread=False); c.row_factory=sqlite3.Row; c.execute('PRAGMA journal_mode=WAL'); return c
def one(sql,a=()):
 c=db();
 try:return c.execute(sql,a).fetchone()
 finally:c.close()
def rows(sql,a=()):
 c=db();
 try:return c.execute(sql,a).fetchall()
 finally:c.close()
def write(sql,a=()):
 c=db(); c.execute(sql,a); c.commit(); c.close()
def event(run,kind,msg,task=None):
 g=one('SELECT goal_id FROM runs WHERE id=?',(run,)); write('INSERT INTO events VALUES(?,?,?,?,?,?,?)',(uid(),run,g[0] if g else None,task,kind,msg,now()))

def init():
 c=db(); c.executescript('''
 CREATE TABLE IF NOT EXISTS companies(id TEXT PRIMARY KEY,name TEXT,created_at TEXT);
 CREATE TABLE IF NOT EXISTS goals(id TEXT PRIMARY KEY,company_id TEXT,title TEXT,description TEXT,criteria TEXT,budget REAL,spent REAL DEFAULT 0,status TEXT,current_run_id TEXT,plan_json TEXT,final_output TEXT,created_at TEXT,updated_at TEXT);
 CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,goal_id TEXT,run_number INTEGER,status TEXT,reason TEXT,started_at TEXT,ended_at TEXT,created_at TEXT);
 CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,run_id TEXT,plan_id TEXT,role TEXT,title TEXT,instructions TEXT,required INTEGER,web INTEGER,phase TEXT,status TEXT,output TEXT,attempts INTEGER DEFAULT 0,max_attempts INTEGER DEFAULT 2,error_type TEXT,error_message TEXT,created_at TEXT,updated_at TEXT);
 CREATE TABLE IF NOT EXISTS deps(id TEXT PRIMARY KEY,run_id TEXT,upstream TEXT,downstream TEXT,created_at TEXT);
 CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,title TEXT,url TEXT,created_at TEXT);
 CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY,run_id TEXT,passed INTEGER,score REAL,checks_json TEXT,failures_json TEXT,created_at TEXT);
 CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,run_id TEXT,goal_id TEXT,task_id TEXT,kind TEXT,message TEXT,created_at TEXT);
 CREATE TABLE IF NOT EXISTS model_calls(id TEXT PRIMARY KEY,run_id TEXT,task_id TEXT,purpose TEXT,started_at TEXT,ended_at TEXT,latency_ms INTEGER,error_type TEXT,error_message TEXT);
 ''');
 if not c.execute('SELECT id FROM companies LIMIT 1').fetchone(): c.execute('INSERT INTO companies VALUES(?,?,?)',(uid(),'My AI Company',now()))
 c.commit(); c.close()
 # Honest restart recovery.
 c=db(); active=c.execute("SELECT id,goal_id FROM runs WHERE status IN ('running','queued')").fetchall()
 for r in active:
  c.execute("UPDATE runs SET status='interrupted',ended_at=?,reason=? WHERE id=?",(now(),'Process restarted; worker state was lost.',r[0]))
  c.execute("UPDATE tasks SET status='interrupted',error_type='process_restart',error_message=?,updated_at=? WHERE run_id=? AND status IN ('running','queued')",('Worker process restarted.',now(),r[0]))
  c.execute("UPDATE goals SET status='interrupted',updated_at=? WHERE id=? AND status IN ('running','queued')",(now(),r[1]))
 c.commit(); c.close()
init()

def get_goal_from_run(run_id):
 r=one('SELECT goal_id FROM runs WHERE id=?',(run_id,)); return one('SELECT * FROM goals WHERE id=?',(r[0],)) if r else None

def text_of(resp):
 if getattr(resp,'output_text',None): return resp.output_text.strip()
 out=[]
 for item in getattr(resp,'output',[]) or []:
  for part in getattr(item,'content',[]) or []:
   t=getattr(part,'text',None)
   if t: out.append(t)
 return '\n'.join(out).strip()

def model(run,purpose,prompt,web=False,tokens=3000,task=None):
 start=time.time(); event(run,'model_start',f'Model call started: {purpose}',task)
 try:
  client=OpenAI(timeout=TIMEOUT,max_retries=MAX_RETRIES); kw={'model':MODEL,'input':prompt,'max_output_tokens':tokens}
  if web and WEB: kw['tools']=[{'type':'web_search'}]
  resp=client.responses.create(**kw); txt=text_of(resp)
  if getattr(resp,'status',None)=='incomplete': raise RuntimeError(f"Model response incomplete: {getattr(getattr(resp,'incomplete_details',None),'reason','unknown')}")
  if not txt: raise RuntimeError('Model returned empty output')
  write('INSERT INTO model_calls VALUES(?,?,?,?,?,?,?,?,?)',(uid(),run,task,purpose,datetime.fromtimestamp(start,tz=timezone.utc).isoformat(),now(),int((time.time()-start)*1000),None,None)); event(run,'model_ok',f'Model call completed: {purpose}',task); return resp,txt
 except Exception as e:
  write('INSERT INTO model_calls VALUES(?,?,?,?,?,?,?,?,?)',(uid(),run,task,purpose,datetime.fromtimestamp(start,tz=timezone.utc).isoformat(),now(),int((time.time()-start)*1000),type(e).__name__,str(e))); event(run,'model_error',f'Model call failed: {purpose}: {e}',task); raise

def capture(resp,run,task):
 for item in getattr(resp,'output',[]) or []:
  for part in getattr(item,'content',[]) or []:
   for a in getattr(part,'annotations',[]) or []:
    u=getattr(a,'url',None)
    if u and not one('SELECT id FROM evidence WHERE run_id=? AND url=?',(run,u)): write('INSERT INTO evidence VALUES(?,?,?,?,?,?)',(uid(),run,task,getattr(a,'title',None) or 'Web source',u,now()))

def fallback():
 return {'tasks':[
 {'id':'T1','title':'Research the objective','instructions':'Collect current evidence and key facts. Separate facts, assumptions and unknowns. Use a small number of high-quality sources.','role':'research','depends':[],'required':1,'web':1,'phase':'work'},
 {'id':'T2','title':'Analyze the research','instructions':'Review T1, reconcile important claims and derive implications. Flag uncertainty.','role':'analyst','depends':['T1'],'required':1,'web':0,'phase':'work'},
 {'id':'T3','title':'Quality review','instructions':'Challenge T1 and T2 for unsupported claims, omissions and contradictions.','role':'qa','depends':['T1','T2'],'required':1,'web':0,'phase':'qa'},
 {'id':'T4','title':'Final report','instructions':'Synthesize T1-T3 into a concise decision-ready report with sources, assumptions, risks and recommendations.','role':'report','depends':['T2','T3'],'required':1,'web':0,'phase':'report'}]}

def plan(run,g):
 p=f'''You are the CEO of a governed AI workforce. Plan the objective into 3-7 focused tasks. Allowed roles: research, analyst, engineering, product, procurement, data, qa, report. Keep each task small enough for one model call. Return ONLY JSON in this exact shape: {{"tasks":[{{"id":"T1","title":"...","instructions":"...","role":"research","depends":[],"required":1,"web":1,"phase":"work"}}]}}. Objective: {g["title"]}. Description: {g["description"]}. Success criteria: {g["criteria"]}.'''
 try:
  _,t=model(run,'planner',p,False,3000); m=re.search(r'\{.*\}',t,re.S); x=json.loads(m.group(0)) if m else None
  if not x or not x.get('tasks'): raise ValueError('Planner returned no tasks')
  ids={t['id'] for t in x['tasks']}
  for t in x['tasks']:
   if any(d not in ids for d in t.get('depends',[])): raise ValueError('Planner returned unknown dependency')
  return {'tasks':x['tasks'][:10]}
 except Exception as e: event(run,'planner_fallback',f'Planner fallback used: {e}'); return fallback()

def upstream(task):
 rs=rows('SELECT t.plan_id,t.title,t.output,t.status FROM tasks t JOIN deps d ON d.upstream=t.id WHERE d.downstream=?',(task['id'],))
 return '\n\n'.join(f"{r['plan_id']} {r['title']} [{r['status']}]\n{r['output'] or ''}" for r in rs)

def do_task(t,run):
 for attempt in range(1,t['max_attempts']+1):
  write('UPDATE tasks SET status=\'running\',attempts=?,updated_at=? WHERE id=?',(attempt,now(),t['id'])); event(run,'task_start',f"{t['role']} started {t['title']}",t['id']); g=get_goal_from_run(run)
  prompt=f'''You are the {t['role']} agent. Objective: {g['title']}. Task: {t['title']}. Instructions: {t['instructions']}. Upstream work: {upstream(t) or 'none'}. Be concise but useful. Clearly label sourced facts, assumptions and unknowns. Never invent sources.'''
  try:
   resp,out=model(run,'research' if t['web'] else t['role'],prompt,bool(t['web']),3000 if t['web'] else 3500,t['id'])
   if t['web']: capture(resp,run,t['id'])
   write('UPDATE tasks SET status=\'completed\',output=?,updated_at=? WHERE id=?',(out,now(),t['id'])); event(run,'task_ok',f"Task completed: {t['title']}",t['id']); return True
  except Exception as e:
   transient=any(x in str(e).lower() for x in ('timeout','timed out','rate limit','connection'))
   final=attempt>=t['max_attempts']; write('UPDATE tasks SET status=?,error_type=?,error_message=?,updated_at=? WHERE id=?',('failed' if final else 'queued','transient_timeout' if transient else type(e).__name__,str(e),now(),t['id']))
   event(run,'task_failed' if final else 'task_retry',f"{t['title']}: {e}",t['id'])
   if final or not transient: return False
 return False

def schedule(run):
 while True:
  pending=rows("SELECT * FROM tasks WHERE run_id=? AND status IN ('queued','pending') ORDER BY rowid",(run,))
  if not pending: return rows('SELECT * FROM tasks WHERE run_id=?',(run,))
  ready=[]; progress=False
  for t in pending:
   ds=rows('SELECT t.status FROM tasks t JOIN deps d ON d.upstream=t.id WHERE d.downstream=?',(t['id'],))
   if any(x[0] in ('failed','interrupted','blocked') for x in ds):
    write('UPDATE tasks SET status=\'blocked\',error_type=\'dependency_blocked\',error_message=?,updated_at=? WHERE id=?',(str([x[0] for x in ds]),now(),t['id'])); progress=True
   elif all(x[0]=='completed' for x in ds): ready.append(t)
  if ready:
   fs=[executor.submit(do_task,t,run) for t in ready]
   for f in fs: f.result()
   progress=True
  if not progress: time.sleep(.5)

def evaluate(run):
 ts=rows('SELECT * FROM tasks WHERE run_id=?',(run,)); req=[t for t in ts if t['required']]; fails=[]; checks=[]
 for t in req:
  ok=t['status']=='completed' and bool(t['output']); checks.append({'task':t['plan_id'],'passed':ok});
  if not ok: fails.append(f"{t['plan_id']} is {t['status']}")
 if any(t['web'] for t in req) and not rows('SELECT * FROM evidence WHERE run_id=?',(run,)): fails.append('Required web research produced no captured evidence.')
 passed=not fails; score=1.0 if passed else max(0,1-len(fails)/max(1,len(req)))
 write('INSERT INTO evaluations VALUES(?,?,?,?,?,?,?)',(uid(),run,int(passed),score,json.dumps(checks),json.dumps(fails),now())); return passed,fails

def final(run):
 ev=one('SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 1',(run,)); ts=rows('SELECT * FROM tasks WHERE run_id=? ORDER BY rowid',(run,)); src=rows('SELECT * FROM evidence WHERE run_id=?',(run,))
 if not ev or not ev['passed']: return 'RUN INCOMPLETE: required work did not pass evaluation.'
 report=next((t['output'] for t in reversed(ts) if t['phase']=='report' and t['output']),None) or next((t['output'] for t in reversed(ts) if t['output']), '')
 return report+'\n\nSources captured:\n'+'\n'.join(f"- {s['title']}: {s['url']}" for s in src)

def run_goal(run):
 g=get_goal_from_run(run)
 try:
  p=plan(run,g); write('UPDATE goals SET plan_json=?,updated_at=? WHERE id=?',(json.dumps(p),now(),g['id'])); event(run,'planner_ok','CEO plan created.')
  ids={}
  for t in p['tasks']:
   tid=uid(); ids[t['id']]=tid; write('INSERT INTO tasks(id,run_id,plan_id,role,title,instructions,required,web,phase,status,max_attempts,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(tid,run,t['id'],t.get('role','analyst'),t['title'],t['instructions'],int(t.get('required',1)),int(t.get('web',0)),t.get('phase','work'),'queued',2,now(),now()))
  for t in p['tasks']:
   for d in t.get('depends',[]): write('INSERT INTO deps VALUES(?,?,?,?,?)',(uid(),run,ids[d],ids[t['id']],now()))
  schedule(run); passed,fails=evaluate(run); out=final(run); status='completed' if passed else 'incomplete'
  write('UPDATE goals SET status=?,final_output=?,updated_at=? WHERE id=?',(status,out,now(),g['id'])); write('UPDATE runs SET status=?,ended_at=?,reason=? WHERE id=?',(status,now(),'passed' if passed else '; '.join(fails),run)); event(run,'run_complete',f'Run {status}.')
 except Exception as e:
  write('UPDATE goals SET status=\'failed\',final_output=?,updated_at=? WHERE id=?',(f'RUN FAILED: {type(e).__name__}: {e}',now(),g['id'])); write('UPDATE runs SET status=\'failed\',ended_at=?,reason=? WHERE id=?',(now(),str(e),run)); event(run,'run_failed',f'Run failed: {type(e).__name__}: {e}')

def start(gid):
 g=one('SELECT * FROM goals WHERE id=?',(gid,)); n=(one('SELECT COALESCE(MAX(run_number),0)+1 FROM runs WHERE goal_id=?',(gid,))[0]); r=uid(); write('INSERT INTO runs VALUES(?,?,?,?,?,?,?,?)',(r,gid,n,'running',None,now(),None,now())); write('UPDATE goals SET status=\'running\',current_run_id=?,updated_at=? WHERE id=?',(r,now(),gid)); event(r,'run_start','Workforce run started.'); executor.submit(run_goal,r)

@app.get('/health')
def health(): return {'status':'ok','app_version':APP_VERSION,'schema_version':SCHEMA_VERSION,'model':MODEL,'db':DB.name}
@app.get('/diagnostics/generation')
def gen():
 try:
  r=uid(); s=time.time(); _,t=model(r,'diagnostic_generation','Reply with exactly OK.',False,4096); return {'status':'ok','message':'Real Responses API generation succeeded.','version':APP_VERSION,'model':MODEL,'output':t,'latency_ms':int((time.time()-s)*1000),'max_output_tokens':4096}
 except Exception as e:return JSONResponse({'status':'failed','error_type':type(e).__name__,'message':str(e),'version':APP_VERSION,'model':MODEL},500)
@app.get('/diagnostics/web')
def wdiag():
 try:
  r=uid(); s=time.time(); resp,t=model(r,'diagnostic_web','Find one current official OpenAI Responses API documentation page. Return its title and URL.',True,8000); capture(resp,r,None); return {'status':'ok','message':'Real Responses API web search succeeded.','version':APP_VERSION,'model':MODEL,'output':t,'latency_ms':int((time.time()-s)*1000),'max_output_tokens':8000}
 except Exception as e:return JSONResponse({'status':'failed','error_type':type(e).__name__,'message':str(e),'version':APP_VERSION,'model':MODEL},500)
@app.get('/api/goals/{gid}')
def api_goal(gid):
 g=one('SELECT * FROM goals WHERE id=?',(gid,));
 if not g:return JSONResponse({'error':'not found'},404)
 r=one('SELECT * FROM runs WHERE id=?',(g['current_run_id'],)) if g['current_run_id'] else None
 return {'goal':dict(g),'run':dict(r) if r else None,'tasks':[dict(x) for x in rows('SELECT * FROM tasks WHERE run_id=? ORDER BY rowid',(r['id'],))] if r else [],'events':[dict(x) for x in rows('SELECT * FROM events WHERE goal_id=? ORDER BY created_at DESC LIMIT 100',(gid,))]}

def esc(v):return html.escape(str(v or ''))
def shell(body):return '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font-family:system-ui;margin:0;background:#f5f6f8}main{max-width:1100px;margin:30px auto;padding:0 18px}section{background:#fff;border:1px solid #ddd;border-radius:16px;padding:22px;margin:18px 0}input,textarea{width:100%;box-sizing:border-box;padding:13px;border:1px solid #ccc;border-radius:10px;margin:7px 0 16px}button{background:#111;color:#fff;border:0;border-radius:10px;padding:12px 18px;font-weight:700}pre{white-space:pre-wrap;background:#f6f7f8;padding:14px;border-radius:10px}.row{padding:13px 0;border-bottom:1px solid #eee}.muted{color:#667}</style></head><body><main>'+body+'</main></body></html>'
@app.get('/',response_class=HTMLResponse)
def home():
 gs=rows('SELECT * FROM goals ORDER BY created_at DESC LIMIT 20'); cards=''.join(f'<div class="row"><a href="/goals/{g["id"]}"><b>{esc(g["title"])}</b></a> â {g["status"]}</div>' for g in gs) or '<p class="muted">No goals yet.</p>'
 return shell(f'<section><h1>AI Workforce OS <small>v{APP_VERSION}</small></h1><p class="muted">Governed multi-agent orchestration: plan, execute, verify, report.</p><form method="post" action="/goals"><label>Objective</label><input name="title" required placeholder="e.g. Research the global industrial automation market"><label>Description</label><textarea name="description" rows="4"></textarea><label>Success criteria</label><textarea name="criteria" rows="4"></textarea><label>Budget (USD, optional)</label><input name="budget" placeholder="System default $5.00"><button>Create goal</button></form></section><section><h2>Recent goals</h2>{cards}</section>')
@app.post('/goals')
def create(title:str=Form(...),description:str=Form(''),criteria:str=Form(''),budget:str=Form('')):
 try:b=float(budget) if budget.strip() else DEFAULT_BUDGET
 except:b=DEFAULT_BUDGET
 b=max(.1,min(b,MAX_BUDGET)); gid=uid(); cid=one('SELECT id FROM companies LIMIT 1')[0]; t=now(); write('INSERT INTO goals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',(gid,cid,title,description,criteria,b,0,'queued',None,None,None,t,t)); start(gid); return RedirectResponse('/goals/'+gid,303)
@app.post('/goals/{gid}/retry')
def retry_goal(gid):
 g=one('SELECT * FROM goals WHERE id=?',(gid,))
 if not g or g['status'] not in ('failed','incomplete','interrupted'): return RedirectResponse('/goals/'+gid,303)
 start(gid); return RedirectResponse('/goals/'+gid,303)

@app.get('/goals/{gid}',response_class=HTMLResponse)
def goal_page(gid):
 g=one('SELECT * FROM goals WHERE id=?',(gid,));
 if not g:return HTMLResponse('Not found',404)
 r=one('SELECT * FROM runs WHERE id=?',(g['current_run_id'],)) if g['current_run_id'] else None; ts=rows('SELECT * FROM tasks WHERE run_id=? ORDER BY rowid',(r['id'],)) if r else []
 ev=rows('SELECT * FROM events WHERE goal_id=? ORDER BY created_at DESC LIMIT 40',(gid,)); taskhtml=''.join(f'<div class="row"><b>{esc(t["title"])}</b><br><span class="muted">{t["phase"]} | {t["status"]}</span>{("<pre>"+esc(t["error_type"]+": "+t["error_message"])+"</pre>") if t["error_message"] else ""}{("<pre>"+esc(t["output"][:3000])+"</pre>") if t["output"] else ""}</div>' for t in ts) or '<p class="muted">No tasks yet.</p>'
 act=''.join(f'<div class="row"><b>{esc(e["kind"])}</b> {esc(e["message"])}<br><small>{esc(e["created_at"])}</small></div>' for e in ev); poll=''
 if g['status'] in ('running','queued'): poll=f'<script>setInterval(()=>fetch("/api/goals/{gid}").then(r=>r.json()).then(x=>{{if(!["running","queued"].includes(x.goal.status))location.reload()}}).catch(()=>{{}}),3000)</script>'
 return shell(f'<a href="/">Back to dashboard</a><section><h1>{esc(g["title"])}</h1><b>Version:</b> {APP_VERSION} &nbsp; <b>Status:</b> {g["status"]}<br><b>Budget:</b> ${g["budget"]:.2f} &nbsp; <b>Run:</b> {r["run_number"] if r else "-"}{("<form method='post' action='/goals/"+gid+"/retry' style='margin-top:15px'><button>Retry goal</button></form>" if g['status'] in ('failed','incomplete','interrupted') else '')}</section><section><h2>CEO plan</h2><pre>{esc(g["plan_json"] or "Planning...")}</pre></section><section><h2>Task execution</h2>{taskhtml}</section><section><h2>Final output</h2><pre>{esc(g["final_output"] or "Waiting for workforce execution...")}</pre></section><section><h2>Activity</h2>{act}</section>{poll}')
