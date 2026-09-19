import pathlib, shutil, hashlib

p = pathlib.Path("app.py")
if not p.exists():
    raise SystemExit("app.py not found. Run from the AI Workforce OS repository root.")
s = p.read_text(encoding="utf-8")
if 'APP_VERSION = "0.4.21"' not in s:
    raise SystemExit("Expected v0.4.21 app.py was not found; refusing to patch an unknown version.")

backup = p.with_name("app.py.v0.4.21.bak")
shutil.copy2(p, backup)
s = s.replace('APP_VERSION = "0.4.21"', 'APP_VERSION = "0.4.22"', 1)
s = s.replace('SCHEMA_VERSION = "046-3"', 'SCHEMA_VERSION = "046-4"', 1)

needle = 'MAX_MODEL_REQUEST_ESTIMATED_TOKENS = int(os.getenv("MAX_MODEL_REQUEST_ESTIMATED_TOKENS", "24000"))\n'
extra = '''# v0.4.22: bounded final-report context.
FINAL_REPORT_CONTEXT_CHARS = int(os.getenv("FINAL_REPORT_CONTEXT_CHARS", "36000"))
FINAL_REPORT_OUTPUT_TOKENS = int(os.getenv("FINAL_REPORT_OUTPUT_TOKENS", "5200"))
FINAL_REPORT_FALLBACK_CHARS = int(os.getenv("FINAL_REPORT_FALLBACK_CHARS", "30000"))
FINAL_REPORT_FALLBACK_OUTPUT_TOKENS = int(os.getenv("FINAL_REPORT_FALLBACK_OUTPUT_TOKENS", "6500"))
FINAL_TASK_OUTPUT_CHARS = int(os.getenv("FINAL_TASK_OUTPUT_CHARS", "700"))
FINAL_TASK_STRUCTURED_CHARS = int(os.getenv("FINAL_TASK_STRUCTURED_CHARS", "900"))
FINAL_EVIDENCE_ITEMS = int(os.getenv("FINAL_EVIDENCE_ITEMS", "8"))
FINAL_CLAIM_ITEMS = int(os.getenv("FINAL_CLAIM_ITEMS", "20"))
FINAL_CONTRADICTION_ITEMS = int(os.getenv("FINAL_CONTRADICTION_ITEMS", "10"))
FINAL_BUDGET_ITEMS = int(os.getenv("FINAL_BUDGET_ITEMS", "10"))
FINAL_CLAIM_CHARS = int(os.getenv("FINAL_CLAIM_CHARS", "350"))
FINAL_CONTRADICTION_CHARS = int(os.getenv("FINAL_CONTRADICTION_CHARS", "650"))
FINAL_CONTRADICTION_REASON_CHARS = int(os.getenv("FINAL_CONTRADICTION_REASON_CHARS", "500"))
FINAL_EVAL_CHARS = int(os.getenv("FINAL_EVAL_CHARS", "3000"))
'''
if needle not in s:
    raise SystemExit("Could not locate request-token constants.")
s = s.replace(needle, needle + extra, 1)

a = s.index("def report_call(rid, prompt, spend_cap):")
b = s.index("\ndef planner_call(", a)

new_report = '''def compact_report_text(value, limit):
    return clip(value, max(200, int(limit)))

def build_report_context(task_rows, evidence_rows, claim_rows, conflict_rows, budget_rows, eval_summary):
    context = {
        "tasks": [{
            "id": t["id"], "title": compact_report_text(t["title"], 180),
            "output": compact_report_text(t["output"], FINAL_TASK_OUTPUT_CHARS),
            "structured": compact_report_text(t["structured"], FINAL_TASK_STRUCTURED_CHARS),
        } for t in task_rows],
        "evidence": [{
            "id": e["id"], "title": compact_report_text(e["title"], 160),
            "url": compact_report_text(e["url"], 420),
            "publisher": compact_report_text(e["publisher"], 100),
            "published_at": e["published_at"], "provenance": e["provenance"],
        } for e in evidence_rows[:FINAL_EVIDENCE_ITEMS]],
        "claims": [{
            "id": c["id"], "task_id": c["task_id"],
            "claim": compact_report_text(c["claim"], FINAL_CLAIM_CHARS),
            "claim_type": c["claim_type"],
            "values_json": compact_report_text(c["values_json"], 400),
            "source_ids": c["source_ids"], "confidence": c["confidence"],
            "status": c["status"],
        } for c in claim_rows[:FINAL_CLAIM_ITEMS]],
        "contradictions": [{
            "claim_a": compact_report_text(c.get("claim_a"), FINAL_CONTRADICTION_CHARS),
            "claim_b": compact_report_text(c.get("claim_b"), FINAL_CONTRADICTION_CHARS),
            "reason": compact_report_text(c.get("reason"), FINAL_CONTRADICTION_REASON_CHARS),
            "severity": c.get("severity"), "status": c.get("status"),
            "resolution": compact_report_text(c.get("resolution"), 400),
        } for c in conflict_rows[:FINAL_CONTRADICTION_ITEMS]],
        "business_budget_items": [{
            "id": b["id"], "task_id": b["task_id"],
            "label": compact_report_text(b["label"], 160),
            "amount": b["amount"], "currency": b["currency"], "period": b["period"],
            "basis": compact_report_text(b["basis"], 300),
            "source_ids": b["source_ids"], "status": b["status"],
        } for b in budget_rows[:FINAL_BUDGET_ITEMS]],
        "verification_summary": compact_report_text(jd(eval_summary), FINAL_EVAL_CHARS),
    }
    return compact_json(context, FINAL_REPORT_CONTEXT_CHARS)

def report_call(rid, prompt, spend_cap):
    try:
        return call(rid,None,"report",prompt,MODEL,False,None,FINAL_REPORT_OUTPUT_TOKENS,spend_cap)
    except Exception as e:
        msg = str(e)
        if "strategy_error:model request too large" in msg:
            event(rid,"REPORT_CONTEXT_COMPACTED",
                  "final report context exceeded the request budget; retrying with a smaller context",
                  {"output_tokens":FINAL_REPORT_OUTPUT_TOKENS,
                   "context_chars":FINAL_REPORT_CONTEXT_CHARS})
            compact = (
                "Write a concise but complete decision-ready final report. Address EVERY success criterion. "
                "Use only the supplied bounded workforce context. Do not repeat evidence unnecessarily. "
                "Clearly label facts, estimates, assumptions, unknowns, and risks. "
                "Preserve unresolved contradictions rather than silently choosing a number. "
                "End with a criterion-by-criterion conclusion.\n" +
                clip(prompt, FINAL_REPORT_FALLBACK_CHARS)
            )
            return call(rid,None,"report_compacted",compact,MODEL,False,None,
                        FINAL_REPORT_FALLBACK_OUTPUT_TOKENS,spend_cap)
        if "incomplete_output:max_output_tokens" not in msg:
            raise
        event(rid,"REPORT_ESCALATED",
              "final report output ceiling reached; retrying with a compact prompt",
              {"initial_tokens":FINAL_REPORT_OUTPUT_TOKENS,
               "escalated_tokens":FINAL_REPORT_FALLBACK_OUTPUT_TOKENS})
        compact = (
            "Write a concise but complete decision-ready final report. Address EVERY success criterion. "
            "Use headings and bullets/tables where useful. Do not repeat evidence unnecessarily. "
            "Clearly label facts, estimates, assumptions, unknowns, and risks. "
            "End with a criterion-by-criterion conclusion.\n" +
            clip(prompt, FINAL_REPORT_FALLBACK_CHARS)
        )
        return call(rid,None,"report_escalated",compact,MODEL,False,None,
                    FINAL_REPORT_FALLBACK_OUTPUT_TOKENS,spend_cap)

'''
s = s[:a] + new_report + s[b+1:]

a = s.index('        ts=q("SELECT * FROM tasks WHERE run_id=? AND status=\'completed\'",(rid,))')
b = s.index('        report=report_call(rid,report_prompt,min(1.0,g["budget"]*.18))["text"]', a)
b = s.index("\n", b) + 1

replacement = '''        ts=q("SELECT * FROM tasks WHERE run_id=? AND status='completed'",(rid,))
        report_tasks=[{"id":t["plan_id"],"title":t["title"],"output":t["output"],
                       "structured":jd(jl(t["structured"]))} for t in ts]
        report_evidence=[{"id":e["id"],"title":e["title"],"url":e["url"],
                          "publisher":e["publisher"],"published_at":e["published_at"],
                          "provenance":jl(e["metadata"],{})}
                         for e in q("SELECT id,title,url,publisher,published_at,metadata FROM evidence WHERE run_id=? ORDER BY retrieved_at DESC LIMIT ?",(rid,MAX_EVIDENCE_ITEMS))]
        report_claims=[dict(c) for c in q("SELECT id,task_id,claim,claim_type,values_json,source_ids,confidence,status FROM claims WHERE run_id=? ORDER BY created_at DESC LIMIT ?",(rid,MAX_CLAIMS))]
        report_conflicts=contradiction_summary(rid)
        report_budgets=[dict(b) for b in q("SELECT * FROM budget_items WHERE run_id=? ORDER BY created_at",(rid,))]
        latest_eval=q("SELECT * FROM evaluations WHERE run_id=? ORDER BY created_at DESC LIMIT 1",(rid,),one=True)
        eval_summary=jl(latest_eval["failures"],{}) if latest_eval else {}
        report_context=build_report_context(report_tasks,report_evidence,report_claims,
                                            report_conflicts,report_budgets,eval_summary)
        report_prompt=f'''Write the final decision-ready report for {clip(g["title"],500)}.
SUCCESS CRITERIA: {clip(g["criteria"],5000)}
WORKFORCE EXECUTION BUDGET: ${float(g["budget"]):.4f}. This is ONLY the AI/API spend cap for producing this report. It is NOT the business launch budget, funding requirement, staffing budget, or customer acquisition budget. Never describe it as one.
BOUNDED WORKFORCE DECISION CONTEXT: {report_context}
Never invent facts or source details. For every important numerical/current factual claim, cite the supporting source ID from the supplied evidence/claim registry and distinguish verified facts, estimates, assumptions, recommendations, unknowns, and unresolved contradictions. If sources disagree, explain the scope/date/definition difference or explicitly preserve the conflict; never silently choose one number. Treat the workforce execution budget as unrelated to the business budget. Do not turn proposed validation steps into claims that validation already occurred.'''
        report=report_call(rid,report_prompt,min(1.0,g["budget"]*.18))["text"]
'''
s = s[:a] + replacement + s[b:]

needle2 = '"max_model_request_estimated_tokens":MAX_MODEL_REQUEST_ESTIMATED_TOKENS,'
if needle2 not in s:
    raise SystemExit("Could not locate health response.")
s = s.replace(needle2, needle2+'"final_report_context_chars":FINAL_REPORT_CONTEXT_CHARS,"final_report_output_tokens":FINAL_REPORT_OUTPUT_TOKENS,"final_report_fallback_output_tokens":FINAL_REPORT_FALLBACK_OUTPUT_TOKENS,',1)

p.write_text(s,encoding="utf-8")
print("Updated app.py to v0.4.22")
print("Backup:", backup)
compile(s, str(p), "exec")
print("Syntax check: PASS")
print("SHA256:", hashlib.sha256(s.encode()).hexdigest())
