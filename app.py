import os
import io
from fastapi import FastAPI, Request, Depends, Form, Query, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# Local modules
from models import (
    init_db,
    fetch_recent,
    fetch_analytics,
    fetch_last_analysis,
    fetch_latest_for_contact,
    log_analysis,
)
from scheduler import (
    start_scheduler,
    compute_insights_score,
    update_fub_insights_score,
    score_lead,
    detect_borough,
    gpt_analyze,
    TONE_PROFILES,
    fub_get,
    get_notes,
    get_emails,
    get_revaluate_score,
)
from market_news import fetch_news, fetch_market_snapshot

# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------
load_dotenv()
app = FastAPI(title="insights.ai – Command Hub (v7)")

# Ensure 'static' exists (Git ignores empty folders in some deploys like Render)
if not os.path.isdir("static"):
    os.makedirs("static/css", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "Bernese2025!")

app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=60 * 60 * 24)


def require_auth(request: Request):
    """
    Gate dashboard routes behind a simple password check.
    Allow /widget and /api/* without login so it can be embedded in FUB.
    """
    path = request.url.path
    if path.startswith("/widget") or path.startswith("/api/") or path == "/login":
        return
    if request.session.get("authed"):
        return
    # Redirect via an exception so FastAPI issues a proper 307 with Location
    raise HTTPException(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/login"},
    )


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler(app)


# -----------------------------------------------------------------------------
# Auth + Dashboard
# -----------------------------------------------------------------------------
@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})


@app.post("/login")
def login_submit(request: Request, password: str = Form(...)):
    if password == DASHBOARD_PASSWORD:
        request.session["authed"] = True
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Invalid password"}
    )


@app.get("/", dependencies=[Depends(require_auth)])
def index(request: Request):
    entries = fetch_recent(limit=25)
    return templates.TemplateResponse("index.html", {"request": request, "entries": entries})


@app.get("/analytics", dependencies=[Depends(require_auth)])
def analytics(request: Request):
    rows = fetch_analytics(limit=500)
    cats = {"HOT": 0, "WARM": 0, "COOL": 0, "DORMANT": 0}
    tones = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for r in rows:
        cats[r[3]] = cats.get(r[3], 0) + 1
        t = (r[4] or "").lower()
        if "pos" in t:
            tones["Positive"] += 1
        elif "neg" in t:
            tones["Negative"] += 1
        else:
            tones["Neutral"] += 1
    valid = [r for r in rows if r[6] is not None]
    avg = round(sum([r[6] for r in valid]) / max(1, len(valid)), 2) if valid else 0
    metrics = {"count": len(rows), "avg_score": avg, "cats": cats, "tones": tones}
    return templates.TemplateResponse(
        "analytics.html", {"request": request, "rows": rows, "metrics": metrics}
    )


@app.get("/export-csv", dependencies=[Depends(require_auth)])
def export_csv():
    from models import _conn

    conn = _conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT contact_name,score,category,tone,tone_mode,next_action,draft,insights_score,revaluate_score,created_at
        FROM analyses
        ORDER BY created_at DESC
        """
    )
    data = c.fetchall()
    conn.close()

    output = io.StringIO()
    import csv as _csv

    w = _csv.writer(output)
    w.writerow(
        [
            "contact_name",
            "lead_score",
            "category",
            "tone",
            "tone_mode",
            "next_action",
            "draft",
            "insights_ai_score",
            "revaluate_score",
            "created_at",
        ]
    )
    for row in data:
        w.writerow(row)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=insights_ai_all_analyses.csv"},
    )


# -----------------------------------------------------------------------------
# Widget (for FUB) + Sandbox
# -----------------------------------------------------------------------------
@app.get("/widget")
def widget(
    request: Request,
    contact_id: int,
    include_market: int = 1,
    include_news: int = 1,
    auto: int = 0,   # NEW: toggle auto-analysis on first load
):
    from market_news import fetch_news, fetch_market_snapshot

    # 1) Check last analysis for this contact
    last = fetch_last_analysis(contact_id)

    # 2) If none and auto=1, run a first-pass analysis now
    if not last and auto:
        person = fub_get(f"/people/{contact_id}")
        person = person.get("person", person) if isinstance(person, dict) else person
        notes = get_notes(contact_id)
        emails = get_emails(contact_id)
        lead = score_lead(person, notes, emails)
        borough_hint = detect_borough(person)

        news_items = fetch_news(max_items=6, nyc_only=True, borough_hint=borough_hint) if include_news else []
        market = fetch_market_snapshot(area="NYC") if include_market else {"bullets": []}

        tone_txt, action, draft, _raw = gpt_analyze(
            person, notes, emails, lead,
            tone_mode="Recommended",
            market_bullets=market.get("bullets"),
            news_items=news_items,
            borough_hint=borough_hint
        )

        insight = compute_insights_score(person, lead, get_revaluate_score(person), contact_id)
        update_fub_insights_score(contact_id, insight)

        log_analysis(
            contact_id,
            person.get("name"),
            person.get("stage"),
            person.get("tags"),
            lead,
            "WARM" if lead >= 5 else "COOL",
            tone_txt,
            "Recommended",
            action,
            draft,
            insight,
            get_revaluate_score(person),
        )

        # refresh "last" after logging
        last = fetch_last_analysis(contact_id)

    # 3) Render the widget with news/market (as before)
    news = fetch_news(max_items=4, nyc_only=True)
    market = fetch_market_snapshot(area="NYC")

    if not last:
        # still nothing? (e.g., FUB API error) -> show empty state
        return templates.TemplateResponse(
            "widget.html",
            {
                "request": request,
                "data": None,
                "spark": [],
                "news": news,
                "market": market,
                "include_market": include_market,
                "include_news": include_news,
            },
        )

    spark = fetch_latest_for_contact(contact_id, limit=5)
    data = {
        "contact_id": contact_id,
        "name": last[0],
        "stage": last[1],
        "tags": last[2],
        "lead_score": last[3],
        "category": last[4],
        "tone": last[5],
        "tone_mode": last[6],
        "next_action": last[7],
        "draft": last[8],
        "insights_score": last[9],
        "reval": last[10],
        "created_at": last[11],
    }

    return templates.TemplateResponse(
        "widget.html",
        {
            "request": request,
            "data": data,
            "spark": spark,
            "news": news,
            "market": market,
            "include_market": include_market,
            "include_news": include_news,
        },
    )



@app.get("/sandbox", dependencies=[Depends(require_auth)])
def sandbox(request: Request):
    return templates.TemplateResponse("sandbox.html", {"request": request})


# -----------------------------------------------------------------------------
# APIs used by the UI
# -----------------------------------------------------------------------------
@app.get("/api/get_news")
def api_get_news(q: str = Query("", description="Optional keyword"), max_items: int = 6, borough: str = ""):
    items = fetch_news(max_items=max_items, nyc_only=True, borough_hint=borough or None)
    if q:
        ql = q.lower()
        items = [i for i in items if ql in (i.get("title", "").lower() + " " + i.get("summary", "").lower())]
    return JSONResponse({"items": items})


@app.get("/api/get_market_data")
def api_market(area: str = "NYC"):
    return JSONResponse(fetch_market_snapshot(area=area))


@app.post("/api/regen_draft")
async def api_regen_draft(contact_id: int, tone: str = "Recommended", include_news: int = 1, include_market: int = 1):
    # Fetch FUB data
    person = fub_get(f"/people/{contact_id}")
    person = person.get("person", person) if isinstance(person, dict) else person
    notes = get_notes(contact_id)
    emails = get_emails(contact_id)

    # Scoring + context
    lead = score_lead(person, notes, emails)
    borough_hint = detect_borough(person)
    news = fetch_news(max_items=6, nyc_only=True, borough_hint=borough_hint) if include_news else []
    market = fetch_market_snapshot(area="NYC") if include_market else {"bullets": []}

    # Generate draft
    tone_mode = tone if tone in TONE_PROFILES else "Recommended"
    tone_txt, action, draft, _raw = gpt_analyze(
        person,
        notes,
        emails,
        lead,
        tone_mode=tone_mode,
        market_bullets=market.get("bullets"),
        news_items=news,
        borough_hint=borough_hint,
    )

    # Compute and write-back Insights.ai Score to FUB custom field
    insight = compute_insights_score(person, lead, get_revaluate_score(person), contact_id)
    update_fub_insights_score(contact_id, insight)

    # Log locally for analytics
    log_analysis(
        contact_id,
        person.get("name"),
        person.get("stage"),
        person.get("tags"),
        lead,
        "WARM" if lead >= 5 else "COOL",
        tone_txt,
        tone_mode,
        action,
        draft,
        insight,
        get_revaluate_score(person),
    )

    return JSONResponse(
        {"tone": tone_txt, "tone_mode": tone_mode, "next_action": action, "draft": draft, "insights_score": insight}
    )


@app.post("/api/refine_draft")
async def api_refine_draft(contact_id: int, current_draft: str, tone: str = "Recommended"):
    import openai

    tone_mode = tone if tone in TONE_PROFILES else "Recommended"
    prompt = (
        "Refine and slightly tighten the following outreach draft without changing the tone or meaning. "
        f"Keep <= 90 words. Tone style: {TONE_PROFILES.get(tone_mode, '')}\n\n---\n{current_draft}\n---"
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        new_text = resp["choices"][0]["message"]["content"].strip()
    except Exception:
        new_text = current_draft
    return JSONResponse({"draft": new_text, "tone_mode": tone_mode})

