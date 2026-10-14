import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from textprep import build_text  # noqa: E402

MODEL_PATH = ROOT / "models" / "model.joblib"
REVIEW_THRESHOLD = 0.5   # placeholder, not yet validated (see notes below)

app = FastAPI(title="Issue triage classifier")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model file missing. Run: python3 src/train_final.py")
        _model = joblib.load(MODEL_PATH)
    return _model


class Issue(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    body: str = Field("", max_length=20000)


class Prediction(BaseModel):
    label: str
    confidence: float
    needs_review: bool
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(issue: Issue):
    model = get_model()
    proba = model.predict_proba([build_text(issue.title, issue.body)])[0]
    best = int(proba.argmax())
    return Prediction(
        label=str(model.classes_[best]),
        confidence=round(float(proba[best]), 3),
        needs_review=bool(proba[best] < REVIEW_THRESHOLD),
        probabilities={str(c): round(float(p), 3) for c, p in zip(model.classes_, proba)},
    )


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Issue triage demo</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 1rem;color:#222}
 input,textarea{width:100%;box-sizing:border-box;font:inherit;padding:.5rem;margin:.25rem 0 .75rem}
 textarea{height:11rem}
 button{font:inherit;padding:.5rem 1rem;margin:0 .5rem .5rem 0;cursor:pointer}
 .bar{background:#eee;border-radius:4px;margin:.25rem 0}
 .bar div{background:#4c7;height:1.3rem;border-radius:4px;color:#fff;font-size:.8rem;
          padding-left:.4rem;white-space:nowrap;box-sizing:border-box}
 .warn{color:#a60}
 small{color:#666}
</style></head><body>
<h1>Issue triage demo</h1>
<p>Paste a GitHub issue and the model suggests a label.
<small>Trained only on closed scikit-learn issues, so it may not work on other projects.</small></p>
<div>
 <button onclick="fill('bug')">Example: bug</button>
 <button onclick="fill('feature')">Example: feature</button>
 <button onclick="fill('doc')">Example: docs</button>
 <button onclick="fill('ci')">Example: CI</button>
</div>
<label>Title<input id="title" maxlength="300"></label>
<label>Body<textarea id="body" maxlength="20000"></textarea></label>
<button onclick="run()">Classify</button>
<div id="out"></div>
<script>
const EX = {
 bug: ["KMeans raises ValueError with sparse input",
       "### Steps/Code to Reproduce import numpy as np from sklearn.cluster import KMeans ... ### Expected Results No error ### Actual Results ValueError: setting an array element with a sequence"],
 feature: ["Add support for sample_weight in the new estimator",
       "### Describe the workflow you want to enable I would like to pass sample weights ... ### Describe your proposed solution Add a sample_weight argument to fit"],
 doc: ["Typo in the user guide for pipelines",
       "### Describe the issue linked to the documentation The section on pipelines has a wrong link and a misspelled word"],
 ci: ["Wheel builds failing on Linux nightly",
       "The nightly wheel build job has been failing since yesterday, probably a change in the build image."]
};
function fill(k){ title.value = EX[k][0]; body.value = EX[k][1]; run(); }
async function run(){
  const out = document.getElementById('out');
  out.textContent = 'Working...';
  const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({title: title.value, body: body.value})});
  if(!r.ok){ out.textContent = 'Error ' + r.status + (r.status === 422 ? ': a title is required' : ''); return; }
  const d = await r.json();
  out.replaceChildren();
  const h = document.createElement('h2');
  h.textContent = d.label + ' (' + Math.round(d.confidence*100) + '%)';
  out.appendChild(h);
  if(d.needs_review){
    const w = document.createElement('p'); w.className = 'warn';
    w.textContent = 'Low confidence: a human should check this one.'; out.appendChild(w);
  }
  Object.entries(d.probabilities).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>{
    const bar = document.createElement('div'); bar.className = 'bar';
    const fillEl = document.createElement('div');
    fillEl.style.width = Math.max(v*100, 12) + '%';
    fillEl.textContent = k + ' ' + Math.round(v*100) + '%';
    bar.appendChild(fillEl); out.appendChild(bar);
  });
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE