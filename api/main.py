import os
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from textprep import build_text  

MODEL_PATH = ROOT / "models" / "model.joblib"

REVIEW_THRESHOLD = float(os.environ.get("REVIEW_THRESHOLD", "0.5"))

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
 .bar{position:relative;background:#eee;border-radius:4px;margin:.25rem 0;height:1.4rem;overflow:hidden}
 .bar .fill{background:#8c9;height:100%}
 .bar .lbl{position:absolute;left:.4rem;top:0;line-height:1.4rem;font-size:.85rem;white-space:nowrap}
 .warn{color:#a60}
 .err{color:#b00}
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
const titleEl = document.getElementById('title');
const bodyEl = document.getElementById('body');
const out = document.getElementById('out');

function fill(k){ titleEl.value = EX[k][0]; bodyEl.value = EX[k][1]; run(); }

function showError(msg){
  out.replaceChildren();
  const p = document.createElement('p'); p.className = 'err'; p.textContent = msg;
  out.appendChild(p);
}

async function run(){
  out.textContent = 'Working...';
  let r;
  try {
    r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({title: titleEl.value, body: bodyEl.value})});
  } catch (e) { showError('Could not reach the server.'); return; }
  if(!r.ok){
    let msg = 'Error ' + r.status;
    if(r.status === 422){ msg += ': a title is required (max 300 characters; body max 20000).'; }
    else { try { msg += ': ' + (await r.json()).detail; } catch (e) {} }
    showError(msg); return;
  }
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
    const fillEl = document.createElement('div'); fillEl.className = 'fill';
    fillEl.style.width = (v*100) + '%';
    const lbl = document.createElement('span'); lbl.className = 'lbl';
    lbl.textContent = k + ' ' + Math.round(v*100) + '%';
    bar.appendChild(fillEl); bar.appendChild(lbl); out.appendChild(bar);
  });
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE