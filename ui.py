import sys
import os
import gradio as gr

# Ensure local project root is first on sys.path so our
# local agents/ folder wins over any installed 'agents' package.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from env.base_env import ResumeEnv
from env.models import DecisionAction, RankingAction, Observation


# ---------------------------------------------------------------------------
# Inline baseline agent (avoids agents/ import collision with site-packages)
# ---------------------------------------------------------------------------

_SHORTLIST_KW = ["python", "machine learning", "kubernetes", "docker", "sql"]
_REJECT_KW    = ["photoshop", "illustrator", "crm", "helpdesk", "printer", "sales"]


def _score_resume(resume: str) -> int:
    text = resume.lower()
    return (sum(1 for kw in _SHORTLIST_KW if kw in text)
          - sum(2 for kw in _REJECT_KW    if kw in text))


def act(observation: Observation):
    if observation.task_type in ("easy", "medium"):
        decisions = []
        for resume in observation.resumes:
            s = _score_resume(resume)
            decisions.append("shortlist" if s >= 2 else "maybe" if s == 1 else "reject")
        return DecisionAction(decisions=decisions)
    else:
        scored = sorted(enumerate(observation.resumes),
                        key=lambda x: _score_resume(x[1]), reverse=True)
        return RankingAction(ranking=[i for i, _ in scored])


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_env = None
_observation = None

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --border:    #1e1e2e;
    --accent:    #7c6aff;
    --accent2:   #ff6a9b;
    --green:     #3dffa0;
    --yellow:    #ffe066;
    --text:      #e8e6ff;
    --muted:     #6b6a8a;
    --radius:    12px;
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Mono', monospace !important;
    color: var(--text) !important;
}

/* ---- header ---- */
#smarthire-header {
    background: linear-gradient(135deg, #0d0d1a 0%, #13101f 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px 40px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
#smarthire-header::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(124,106,255,0.18) 0%, transparent 70%);
    pointer-events: none;
}
#smarthire-header::after {
    content: "";
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(255,106,155,0.12) 0%, transparent 70%);
    pointer-events: none;
}

/* ---- cards ---- */
.sh-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 24px;
}

/* ---- labels ---- */
.sh-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}

/* ---- textboxes ---- */
textarea, .gr-textbox textarea {
    background: #0d0d16 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    padding: 14px 16px !important;
    resize: vertical !important;
    transition: border-color 0.2s;
}
textarea:focus {
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(124,106,255,0.12) !important;
}

/* ---- dropdown ---- */
.gr-dropdown select, select {
    background: #0d0d16 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
}

/* ---- buttons ---- */
#btn-start {
    background: linear-gradient(135deg, #5c4fff 0%, #7c6aff 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100% !important;
}
#btn-start:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

#btn-run {
    background: linear-gradient(135deg, #cc3d72 0%, #ff6a9b 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.04em !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    width: 100% !important;
}
#btn-run:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ---- score badge area ---- */
#result-box textarea {
    background: linear-gradient(160deg, #0e1221 0%, #0d0d16 100%) !important;
    border: 1px solid rgba(61,255,160,0.25) !important;
    color: var(--green) !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
}

/* ---- status box ---- */
#status-box textarea {
    border: 1px solid rgba(255,224,102,0.2) !important;
    color: var(--yellow) !important;
    font-size: 12px !important;
}

/* ---- section divider markdown ---- */
.sh-divider { 
    border-top: 1px solid var(--border); 
    margin: 4px 0 16px; 
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
"""

# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------

def reset_env(task: str):
    global _env, _observation
    _env = ResumeEnv(task_type=task)
    _observation = _env.reset()

    jd = _observation.job_description
    resumes_text = "\n\n" + ("─" * 52) + "\n\n".join(
        f"  CANDIDATE {i+1}\n\n{r}"
        for i, r in enumerate(_observation.resumes)
    )
    status = f"✅  Environment ready  ·  Task: {task.upper()}  ·  {len(_observation.resumes)} candidates loaded"
    return jd, resumes_text, status, ""


def run_step():
    global _env, _observation
    if _env is None or _observation is None:
        return "⚠️  Run 'Start Screening' first.", "─"

    action = act(_observation)
    obs, reward, done, info = _env.step(action)

    score = reward.score
    task_type = _observation.task_type

    # Score bar
    filled = int(score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    score_pct = f"{score * 100:.1f}%"

    lines = [
        f"  SCORE   [{bar}]  {score_pct}",
        f"  RAW     {score:.4f} / 1.0000",
        "",
        f"  TASK    {task_type.upper()}",
        "",
    ]

    if task_type in ("easy", "medium"):
        decisions = action.decisions
        labels = {"shortlist": "🟢 SHORTLIST", "maybe": "🟡 MAYBE   ", "reject": "🔴 REJECT  "}
        lines.append("  ── Per-Candidate Decisions ──────────────")
        for i, (resume, dec) in enumerate(zip(_observation.resumes, decisions)):
            name_line = resume.split("\n")[0]
            lines.append(f"  {labels.get(dec, dec)}  Candidate {i+1}: {name_line}")
    else:
        lines.append("  ── Ranking (Best → Worst) ───────────────")
        for rank, idx in enumerate(action.ranking):
            name_line = _observation.resumes[idx].split("\n")[0]
            lines.append(f"  #{rank+1}  →  Candidate {idx+1}: {name_line}")

    lines += [
        "",
        "  ── Reasoning ────────────────────────────",
        f"  {reward.reasoning}",
    ]

    status = f"✅  Evaluation complete  ·  Score: {score_pct}  ·  Done: {done}"
    return "\n".join(lines), status


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="SmartHire") as demo:

    # Header
    gr.HTML("""
    <div id="smarthire-header">
        <div style="font-family:'Syne',sans-serif; font-size:28px; font-weight:800;
                    letter-spacing:-0.02em; color:#e8e6ff; margin-bottom:6px;">
            ◈ SmartHire
            <span style="background:linear-gradient(90deg,#7c6aff,#ff6a9b);
                         -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                AI Resume Evaluator
            </span>
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:12px; color:#6b6a8a;
                    letter-spacing:0.06em;">
            INTELLIGENT CANDIDATE SCREENING  ·  3 TASK LEVELS  ·  DETERMINISTIC SCORING
        </div>
    </div>
    """)

    with gr.Row():
        # ── Left panel ────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220):
            gr.HTML('<div class="sh-label">⚙️ &nbsp;Configuration</div>')

            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task Level",
                interactive=True,
            )

            gr.HTML('<div style="height:12px"></div>')
            btn_start = gr.Button("▶  Start Screening", elem_id="btn-start")
            gr.HTML('<div style="height:8px"></div>')
            btn_run = gr.Button("⚡  Run AI Screening", elem_id="btn-run")

            gr.HTML('<div style="height:20px"></div>')
            gr.HTML('<div class="sh-label">📡 &nbsp;Status</div>')
            status_box = gr.Textbox(
                value="Idle — select a task and press Start.",
                lines=3,
                interactive=False,
                show_label=False,
                elem_id="status-box",
            )

        # ── Right panel ───────────────────────────────────────────
        with gr.Column(scale=3):

            gr.HTML('<div class="sh-label">📋 &nbsp;Job Description</div>')
            jd_box = gr.Textbox(
                value="",
                lines=4,
                interactive=False,
                show_label=False,
                placeholder="Job description will appear here after Start Screening…",
            )

            gr.HTML('<div style="height:16px"></div>')
            gr.HTML('<div class="sh-label">👤 &nbsp;Candidate Resumes</div>')
            resumes_box = gr.Textbox(
                value="",
                lines=10,
                interactive=False,
                show_label=False,
                placeholder="Resumes will appear here…",
            )

            gr.HTML('<div style="height:16px"></div>')
            gr.HTML('<div class="sh-label">🏆 &nbsp;Evaluation Result</div>')
            result_box = gr.Textbox(
                value="",
                lines=14,
                interactive=False,
                show_label=False,
                placeholder="Results will appear here after Run AI Screening…",
                elem_id="result-box",
            )

    # ── Wire up buttons ───────────────────────────────────────────
    btn_start.click(
        fn=reset_env,
        inputs=[task_dropdown],
        outputs=[jd_box, resumes_box, status_box, result_box],
    )

    btn_run.click(
        fn=run_step,
        inputs=[],
        outputs=[result_box, status_box],
    )

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
