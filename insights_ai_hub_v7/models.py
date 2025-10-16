import sqlite3
from zoneinfo import ZoneInfo
DB_PATH = "insights_ai.db"
NY_TZ = ZoneInfo("America/New_York")
def _conn(): return sqlite3.connect(DB_PATH)
def init_db():
    conn=_conn(); c=conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        contact_id INTEGER, contact_name TEXT, stage TEXT, tags TEXT,
        score INTEGER, category TEXT, tone TEXT, tone_mode TEXT,
        next_action TEXT, draft TEXT, insights_score REAL, revaluate_score REAL,
        created_at TEXT)''')
    conn.commit(); conn.close()
def log_analysis(contact_id, name, stage, tags, score, category, tone, tone_mode, next_action, draft, insights_score, revaluate_score):
    conn=_conn(); c=conn.cursor()
    c.execute('''INSERT INTO analyses(contact_id,contact_name,stage,tags,score,category,tone,tone_mode,next_action,draft,insights_score,revaluate_score,created_at)
                 VALUES(?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))''',
        (contact_id,name or "",stage or "",", ".join(tags or []) if isinstance(tags,list) else (tags or ""),score,category,tone or "",tone_mode or "Recommended",next_action or "",draft or "",float(insights_score) if insights_score is not None else None,float(revaluate_score) if revaluate_score is not None else None))
    conn.commit(); conn.close()
def fetch_recent(limit=25):
    conn=_conn(); c=conn.cursor()
    c.execute("SELECT contact_id,contact_name,score,category,tone,next_action,draft,insights_score,revaluate_score,created_at,tone_mode FROM analyses ORDER BY created_at DESC LIMIT ?",(limit,))
    rows=c.fetchall(); conn.close(); return rows
def fetch_latest_for_contact(contact_id, limit=5):
    conn=_conn(); c=conn.cursor()
    c.execute("SELECT insights_score,created_at,next_action,draft,tone,tone_mode FROM analyses WHERE contact_id=? ORDER BY created_at DESC LIMIT ?",(contact_id,limit))
    rows=c.fetchall(); conn.close(); return rows
def fetch_last_analysis(contact_id):
    conn=_conn(); c=conn.cursor()
    c.execute("SELECT contact_name,stage,tags,score,category,tone,tone_mode,next_action,draft,insights_score,revaluate_score,created_at FROM analyses WHERE contact_id=? ORDER BY created_at DESC LIMIT 1",(contact_id,))
    row=c.fetchone(); conn.close(); return row
def fetch_analytics(limit=500):
    conn=_conn(); c=conn.cursor()
    c.execute("SELECT contact_id,contact_name,created_at,category,tone,score,insights_score,tone_mode FROM analyses ORDER BY created_at DESC LIMIT ?",(limit,))
    rows=c.fetchall(); conn.close(); return rows
