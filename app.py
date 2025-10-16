import os
import io
import json
from base64 import urlsafe_b64decode

from fastapi import FastAPI, Request, Depends, Form, Query, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# OpenAI (new SDK)
OPENAI_DISABLED = os.getenv("OPENAI_DISABLED", "0") == "1"
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # allows running without the SDK when disabled

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
    TONE_PROFILES,
    fub_get,
    get_notes,
    get_emails,
)
from market_news import fetch_news, fetch_market_snapshot

# -----------------------------------------------------------------------------
# App bootstrap
# -----------------------------------------------------------------------------
load_dotenv()
app = FastAPI(title="insights.ai – Command Hub (v7.3)")

# CORS for XHRs from the widget (not strictly needed for iframe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security headers so FUB can embed us
@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    # Remove any X-Frame-Options that could block embedding
    for k in list(resp.headers.keys()):
        if k.lower() == "x-frame-options":
            resp.headers.pop(k, None)
    # Allow FUB (and subdomains) to iframe this app
    resp.headers["Content-Security-Policy"] = (
        "frame-ancestors 'self' https://app.followupboss.com https://*.followupboss.com https://*.followupboss.net;"
    )
    return resp

# Ensure 'static' exists (Render/Git sometimes ship empty dirs)
if not os.path.isdir("static"):
    os.makedirs("static/css", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "Bernese2025!")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Session cookie works inside FUB iframe: SameSite=None; Secure
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=60 * 60 * 24,
    same_site="none",
    https_only=True,
)

# OpenAI client (new SDK) if enabled
client = OpenAI(api_key=OPENAI_API_KEY) if (not OPENAI_DISABLED and OPENAI_API_KEY and OpenAI) else None

# -----------------------------------------------------------------------------
# Auth helper
# -----------------------------------------------------------------------------
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
    # Redirect via exception so FastAPI issues a 307 with Location
    raise HTTPException(
        status_code=status.HTTP_307_TEMPORARY_REDIRECT,
        headers={"Location": "/login"},
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_contact_id(request: Request, contact_id):
    # If integer-like, use it
    try:
        if contact_id is not None and str(contact_id).isdigit():
            return int(contact_id)
    except Exception:
        pass
    # Fallback: parse FUB context
    ctx = request.query_params.get("context")
    if not ctx:
        return None
    try:
        padded = ctx + "=" * (-len(ctx) % 4)  # base64url padding
        data = json.loads(urlsafe_b64decode(padded.encode()).decode())
        pid = data.get("person", {}).get("id")
        if pid:
            return int(pid)
    except Exception:
        return None
    return None

def _merge_person_with_custom_fields(person: dict, cid: int) -> dict:
    """
    Some FUB accounts only return custom fields when included explicitly.
    We re-fetch with ?include=customFields and merge.
    """
    try:
        pf = fub_get(f"/people/{cid}", params={"include": "customFields"})
        pf = pf.get("person", pf) if isinstance(pf, dict) else pf
        if isinstance(pf, dict):
            # prefer keys from pf when present
            base = dict(person or {})
            for k, v in pf.items():
                if k not in base or (k in ("custom", "customFields") and v):
                    base[k] = v
            return base
    except Exception:
        pass
    return person or {}

def _extract_revaluate(person: dict) -> float | None:
    if not person:
        return None
    # 1) person['custom'] (dict or dict with 'fields' list)
    custom = person.get("custom")
    if isinstance(custom, dict):
        # direct keys
        for k in list(custom.keys()):
            kl = str(k).strip().lower()
            if "revaluate" in kl and "score" in kl:
                try:
                    return float(custom[k])
                except Exception:
                    pass
        # nested fields list
        fields = custom.get("fields")
        if isinstance(fields, list):
            for f in fields:
                name = str(f.get("name", "")).lower()
                if "revaluate" in name and "score" in name:
                    try:
                        return float(f.get("value"))
                    except Exception:
                        pass
    # 2) person['customFields'] (list of maps)
    cf = person.get("customFields")
    if isinstance(cf, list):
        for f in cf:
            cand = f.get("name") or f.get("label") or f.get("fieldName") or f.get("field")
            if cand and isinstance(cand, str):
                kl = cand.strip().lower()
                if "revaluate" in kl and "score" in kl:
                    try:
                        return float(f.get("value"))
                    except Exception:
                        pass
    # 3) rare 'properties'
    props = person.get("properties")
    if isinstance(props, dict):
        for k in props.keys():
            kl = str(k).strip().lower()
            if "revaluate" in kl and "score" in kl:
                try:
                    return float(props[k])
                except Exception:
                    pass
    return None

def _fallback_draft(person, lead, borough_hint, market_bullets, news_items, tone_mode):
    """Used when OpenAI is unavailable (429/disabled)."""
    name = person.get("firstName") or person.get("name") or "there"
    borough_line = f" Given the {borough_hint} trend," if borough_hint else ""
    market_line = f" {market_bullets[0]}" if market_bullets else ""
    news_line = f" Also, saw: {news_items[0].get('title')}" if news_items else ""
    return (
        tone_mode or "Recommended",
        "Follow up to re-engage.",
        f"Hi {name}, quick pulse check on your plans.{borough_line}{market_line}{news_line} "
        f"If helpful, I can send a 2-min update on inventory and pricing. Want me to share that?",
    )

def _generate_draft(person, notes, emails, lead, tone_mode, market_bullets=None, news_items=None, borough_hint=None):
    """Calls OpenAI if available; otherwise falls back."""
    mb = [f"- {b}" for b in (market_bullets or [])]
    ni = [f"- {n.get('title')} — {n.get('summary','')[:160]}..." for n in (news_items or [])]
    if OPENAI_DISABLED or not client:
        return _fallback_draft(person, lead, borough_hint, mb, news_items or [], tone_mode)

    tone_desc = TONE_PROFILES.get(tone_mode or "Recommended", "")
    texts = []
    for n in notes or []: texts.append(n.get("body", ""))
    for e in emails or []: texts.extend([e.get("body", ""), e.get("subject", "")])
    combined = "\n".join([t for t in texts if t])[-3000:] or "(No recent text found.)"
    b_focus = f"The client’s market context is {borough_hint.title()}." if borough_hint else ""
    prompt = f"""
You are Roger's "Insights.ai" assistant. Draft a concise (≤ 90 words), human, NYC-savvy outreach.

{b_focus}
Contact: {person.get('name')}
Stage: {person.get('stage')}
Lead score: {lead}
Tone style: {tone_mode} — {tone_desc}

Recent history:
{combined}

Current Market Highlights:
{os.linesep.join(mb)}

Recent Headlines:
{os.linesep.join(ni)}

Task:
1) Write one short message that naturally weaves 1–2 contextual insights above.
2) Keep it confident, helpful, and specific (no clichés).
3) End with a soft next step.
""".strip()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.55,
            max_tokens=450,
        )
        draft = resp.choices[0].message.content.strip()
        return (tone_mode or "Recommended"), "Follow up to re-engage.", draft
    except Exception:
        # e.g., 429 insufficient_quota — fallback so the widget still loads
        return _fallback_draft(person, lead, borough_hint, mb, news_items or [], tone_mode)

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
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid password"})

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
        if "pos" in t: tones["Positive"] += 1
        elif "neg" in t: tones["Negative"] += 1
        else: tones["Neutral"] += 1
    valid = [r for r in rows if r[6] is not None]
    avg = round(sum([r[6] for r in valid]) / max(1, len(valid)), 2) if valid else 0
    metrics = {"count": len(rows), "avg_score": avg, "cats": cats, "tones": tones}
    return templates.TemplateResponse("analytics.html", {"request": request, "rows": rows, "metrics": metrics})

@app.get("/export-csv", dependencies=[Depends(require_auth)])
def export_csv():
    from models import _conn
    conn = _conn(); c = conn.cursor()
    c.execute("""
        SELECT contact_name,score,category,tone,tone_mode,next_action,draft,insights_score,revaluate_score,created_at
        FROM analyses ORDER BY created_at DESC
    """)
    data = c.fetchall(); conn.close()
    output = io.StringIO()
    import csv as _csv
    w = _csv.writer(output)
    w.writerow(["contact_name","lead_score","category","tone","tone_mode","next_action","draft","insights_ai_score","revaluate_score","created_at"])
    for row in data: w.writerow(row)
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=insights_ai_all_analyses.csv"})

# -----------------------------------------------------------------------------
# Widget (FUB) + Sandbox
# -----------------------------------------------------------------------------
@app.get("/widget")
def widget(
    request: Request,
    contact_id: str | None = None,
    include_market: int = 1,
    include_news: int = 1,
    auto: int = 0,
):
    cid = _resolve_contact_id(request, contact_id)
    news = fetch_news(max_items=4, nyc_only=True)
    market = fetch_market_snapshot(area="NYC")

    if not cid:
        return templates.TemplateResponse(
            "widget.html",
            {"request": request, "data": None, "spark": [], "news": news, "market": market,
             "include_market": include_market, "include_news": include_news},
        )

    last = fetch_last_analysis(cid)

    if not last and auto:
        # Pull person, merging in customFields explicitly
        person = fub_get(f"/people/{cid}")
        person = person.get("person", person) if isinstance(person, dict) else person
        person = _merge_person_with_custom_fields(person, cid)

        notes = get_notes(cid)
        emails = get_emails(cid)
        lead = score_lead(person, notes, emails)
        borough_hint = detect_borough(person)
        news_items = fetch_news(max_items=6, nyc_only=True, borough_hint=borough_hint) if include_news else []
        market_full = fetch_market_snapshot(area="NYC") if include_market else {"bullets": []}
        reval = _extract_revaluate(person)

        tone_txt, action, draft = _generate_draft(
            person, notes, emails, lead, tone_mode="Recommended",
            market_bullets=market_full.get("bullets"), news_items=news_items, borough_hint=borough_hint
        )

        insight = compute_insights_score(person, lead, reval, cid)
        update_fub_insights_score(cid, insight)
        log_analysis(
            cid, person.get("name"), person.get("stage"), person.get("tags"),
            lead, "WARM" if lead>=5 else "COOL", tone_txt, "Recommended",
            action, draft, insight, reval
        )
        last = fetch_last_analysis(cid)

    if not last:
        return templates.TemplateResponse(
            "widget.html",
            {"request": request, "data": None, "spark": [], "news": news, "market": market,
             "include_market": include_market, "include_news": include_news},
        )

    # Ensure Revaluate displays even if not stored earlier
    display_reval = last[10]
    if display_reval is None:
        try:
            p_now = fub_get(f"/people/{cid}", params={"include": "customFields"})
            p_now = p_now.get("person", p_now) if isinstance(p_now, dict) else p_now
            display_reval = _extract_revaluate(p_now)
        except Exception:
            pass

    spark = fetch_latest_for_contact(cid, limit=5)
    data = {
        "contact_id": cid,
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
        "reval": display_reval,
        "created_at": last[11],
    }
    return templates.TemplateResponse(
        "widget.html",
        {"request": request, "data": data, "spark": spark, "news": news, "market": market,
         "include_market": include_market, "include_news": include_news},
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
    person = fub_get(f"/people/{contact_id}")
    person = person.get("person", person) if isinstance(person, dict) else person
    person = _merge_person_with_custom_fields(person, contact_id)

    notes = get_notes(contact_id)
    emails = get_emails(contact_id)
    lead = score_lead(person, notes, emails)
    borough_hint = detect_borough(person)
    news = fetch_news(max_items=6, nyc_only=True, borough_hint=borough_hint) if include_news else []
    market = fetch_market_snapshot(area="NYC") if include_market else {"bullets": []}
    reval = _extract_revaluate(person)

    tone_mode = tone if tone in TONE_PROFILES else "Recommended"
    tone_txt, action, draft = _generate_draft(
        person, notes, emails, lead, tone_mode=tone_mode,
        market_bullets=market.get("bullets"), news_items=news, borough_hint=borough_hint
    )

    insight = compute_insights_score(person, lead, reval, contact_id)
    update_fub_insights_score(contact_id, insight)
    log_analysis(
        contact_id, person.get("name"), person.get("stage"), person.get("tags"),
        lead, "WARM" if lead>=5 else "COOL", tone_txt, tone_mode, action, draft, insight, reval
    )
    return JSONResponse({"tone": tone_txt, "tone_mode": tone_mode, "next_action": action,
                         "draft": draft, "insights_score": insight, "revaluate_score": reval})

@app.post("/api/refine_draft")
async def api_refine_draft(contact_id: int, current_draft: str, tone: str = "Recommended"):
    tone_mode = tone if tone in TONE_PROFILES else "Recommended"
    prompt = (
        "Refine and slightly tighten the following outreach draft without changing the tone or meaning. "
        f"Keep <= 90 words. Tone style: {TONE_PROFILES.get(tone_mode, '')}\n\n---\n{current_draft}\n---"
    )
    # If OpenAI is disabled/unavailable, just return the current draft
    if OPENAI_DISABLED or not client:
        return JSONResponse({"draft": current_draft, "tone_mode": tone_mode})
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        new_text = resp.choices[0].message.content.strip()
    except Exception:
        new_text = current_draft
    return JSONResponse({"draft": new_text, "tone_mode": tone_mode})
