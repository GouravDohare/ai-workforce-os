# AI Workforce OS v0.2

Development prototype:
Human objective -> CEO plan -> specialist tasks -> handoffs -> executive report.

Run locally:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload

Open http://127.0.0.1:8000

For browser deployment, push this directory to GitHub and deploy it as a Docker Web Service using render.yaml. Add OPENAI_API_KEY as a secret for live model execution. Without a key, deterministic demo mode works.

This is development software: no authentication, external side effects, purchasing, email, arbitrary shell execution, or real web/document tools.
