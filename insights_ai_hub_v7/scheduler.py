import os, statistics
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from models import init_db, log_analysis, fetch_latest_for_contact
import openai

NY_TZ = ZoneInfo("America/New_York")
load_dotenv()

FUB_API_KEY = os.getenv("FUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DIGEST_TIME = os.getenv("DIGEST_TIME", "08:00")
MAX_CONTACTS_PER_RUN = int(os.getenv("MAX_CONTACTS_PER_RUN", "300"))
ALLOWED_TAGS = [t.strip().lower() for t in os.getenv("ANALYZE_ONLY_TAGS", "buyer,seller").split(",") if t.strip()]
UPDATE_FUB_SCORE = os.getenv("UPDATE_FUB_SCORE", "True").lower() == "true"
FUB_SCORE_FIELD = os.getenv("FUB_SCORE_FIELD", "Insights.ai Score")

FUB_BASE = "https://api.followupboss.com/v1"
AUTH = (FUB_API_KEY, "")
openai.api_key = OPENAI_API_KEY

TONE_PROFILES = {
    "Recommended": "",
    "Professional": "Maintain a confident, polished, clear tone with strong guidance.",
    "Friendly": "Warm and conversational, concise and approachable.",
    "Analytical": "Data-driven and logical. Reference facts and trends succinctly.",
    "Empathetic": "Supportive, understanding, and patient; reduce friction and stress.",
    "Persuasive": "Motivational tone, emphasize value, timing, and next-step urgency.",
    "Luxury Buyer": "Refined, elegant, aspirational; emphasize discretion and quality.",
    "High-Energy": "Upbeat, momentum-building, action-oriented."
}

def fub_get(path, params=None):
    r = requests.get(f"{FUB_BASE}{path}", auth=AUTH, params=params or {})
    r.raise_for_status(); return r.json()

def fub_put(path, json_body):
    r = requests.put(f"{FUB_BASE}{path}", auth=AUTH, json=json_body)
    r.raise_for_status(); return r.json()

def normalize(val, minv, maxv):
    try: v = float(val)
    except Exception: return 0.0
    if v <= minv: return 0.0
    if v >= maxv: return 100.0
    return (v - minv) / (maxv - minv) * 100.0

def get_notes(contact_id):
    try: return fub_get(f"/people/{contact_id}/notes").get("notes", [])
    except Exception: return []

def get_emails(contact_id):
    try: return fub_get(f"/people/{contact_id}/emails").get("emails", [])
    except Exception: return []

def get_revaluate_score(person):
    custom = person.get("custom", {}) if isinstance(person.get("custom"), dict) else {}
    for key in ["revaluate score", "Revaluate Score", "revaluate_score", "RevaluateScore"]:
        if key in custom:
            try: return float(custom[key])
            except Exception: return None
    return None

def score_lead(person, notes, emails):
    score = 0
    now_utc = datetime.now(timezone.utc)
    last_activity = person.get("lastActivityDate")
    if last_activity:
        try:
            dt = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
            delta_days = (now_utc - dt).days
            if delta_days < 7: score += 3
            elif delta_days > 45: score -= 3
        except Exception: pass
    stage = (person.get("stage") or "").lower()
    if "hot" in stage: score += 4
    elif "active" in stage: score += 3
    elif "nurture" in stage: score += 2
    blobs = []
    for n in notes: blobs.append(n.get("body", "") or "")
    for e in emails:
        blobs.extend([e.get("body", "") or "", e.get("subject", "") or ""])
    text = " ".join(blobs).lower()
    positive_kw = ["excited","love","great","ready","perfect","amazing","move forward","offer"]
    negative_kw = ["concern","issue","problem","worry","not sure","delay","wait","pause","stop","no longer"]
    score += sum(2 for kw in positive_kw if kw in text)
    score -= sum(2 for kw in negative_kw if kw in text)
    intent_kw = ["buy","sell","mortgage","pre-approval","timeline","closing","offer","list","showing","tour"]
    score += sum(1 for kw in intent_kw if kw in text)
    return score

def tone_to_numeric(tone):
    t = (tone or "").lower()
    if "pos" in t: return 85
    if "neg" in t: return 25
    return 55

def compute_insights_score(person, lead_score, revaluate_value, contact_id):
    hist = fetch_latest_for_contact(contact_id, limit=5)
    insights_hist = [h[0] for h in hist if h[0] is not None]
    tone_hist = [tone_to_numeric(h[4]) for h in hist if len(h) >= 5]
    if len(insights_hist) >= 2:
        slope = max(min(insights_hist[0] - insights_hist[-1], 20), -20)
        trend_norm = normalize(slope, -20, 20)
    else:
        trend_norm = 50.0
    lead_norm = normalize(lead_score, -10, 40)
    reval_norm = normalize(revaluate_value, 0, 100) if revaluate_value is not None else 50.0
    rec_norm = 50.0
    try:
        if person.get("lastActivityDate"):
            dt = datetime.fromisoformat(person["lastActivityDate"].replace("Z","+00:00"))
            days = (datetime.now(timezone.utc) - dt).days
            rec_norm = normalize(max(0, 30 - days), 0, 30)
    except Exception: pass
    if len(tone_hist) >= 2:
        var = statistics.pstdev(tone_hist)
        tone_norm = normalize(10 - var, 0, 10)
    else:
        tone_norm = 55.0
    stage = (person.get("stage") or "").lower()
    stage_boost = 100.0 if "active" in stage else (70.0 if "nurture" in stage else 50.0)
    insight = (trend_norm * 0.20 + lead_norm * 0.25 + reval_norm * 0.25 + rec_norm * 0.15 + tone_norm * 0.10 + stage_boost * 0.05)
    return round(min(max(insight, 0.0), 100.0), 2)

def update_fub_insights_score(person_id, score):
    if not UPDATE_FUB_SCORE: return
    body = {"id": person_id, "custom": {FUB_SCORE_FIELD: score}}
    try: fub_put(f"/people/{person_id}", body)
    except Exception: pass

def detect_borough(person):
    borough_hint = None
    addr = person.get("address") or {}
    city = (addr.get("city") or "").lower()
    if city in ["brooklyn","manhattan","queens","bronx","staten island"]:
        borough_hint = city
    else:
        tags = [t.lower() for t in (person.get("tags") or [])]
        for b in ["brooklyn","manhattan","queens","bronx","staten island"]:
            if b in tags:
                borough_hint = b; break
    return borough_hint or "nyc"

def build_prompt(person, notes, emails, lead_score, tone_mode, market_bullets=None, news_items=None, borough_hint=None):
    tone_desc = TONE_PROFILES.get(tone_mode or "Recommended", "")
    recent_texts = []
    for n in notes: recent_texts.append(n.get("body",""))
    for e in emails: recent_texts.append(e.get("body",""))
    combined = "\n".join(recent_texts[-5:])[:3000] or "(No recent text found.)"
    mb = "\n".join(f"- {b}" for b in (market_bullets or []))
    ni = "\n".join(f"- {n.get('title')} — {n.get('summary','')[:160]}..." for n in (news_items or []))
    b_focus = f"The client’s market context is {borough_hint.title()}." if borough_hint else ""
    return f"""
You are Roger's "Insights.ai" assistant. Draft a concise (≤ 90 words), human, NYC-savvy outreach.

{b_focus}
Contact: {person.get('name')}
Stage: {person.get('stage')}
Lead score: {lead_score}
Tone style: {tone_mode} — {tone_desc}

Recent history:
{combined}

Current Market Highlights:
{mb}

Recent Headlines:
{ni}

Task:
1) Write one short message that naturally weaves 1–2 contextual insights above.
2) Keep it confident, helpful, and specific (no clichés).
3) End with a soft next step.
"""

def gpt_analyze(person, notes, emails, lead_score, tone_mode="Recommended", market_bullets=None, news_items=None, borough_hint=None):
    prompt = build_prompt(person, notes, emails, lead_score, tone_mode, market_bullets, news_items, borough_hint)
    try:
        resp = openai.ChatCompletion.create(model="gpt-4-turbo", messages=[{"role":"user","content":prompt}], temperature=0.55, max_tokens=450)
        content = resp["choices"][0]["message"]["content"].strip()
        draft = content; tone = tone_mode; next_action = "Follow up to re-engage."
        return tone, next_action, draft, prompt
    except Exception as e:
        return "Neutral", "Follow up manually.", f"[Error: {e}]", prompt

def start_scheduler(app):
    if getattr(app.state, "scheduler_started", False): return
    app.state.scheduler_started = True
    scheduler = BackgroundScheduler(timezone=str(NY_TZ))
    hh, mm = DIGEST_TIME.split(":")
    scheduler.add_job(lambda: None, "cron", hour=int(hh), minute=int(mm), id="noop", replace_existing=True)
    scheduler.start()
