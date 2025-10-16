import time, requests, feedparser
from bs4 import BeautifulSoup
CACHE={'news':{'data':[],'ts':0},'market':{'data':{},'ts':0}}; TTL=60*60*6
def _within_ny(t):
    if not t: return False
    s=t.lower()
    keys=['nyc','new york','manhattan','brooklyn','queens','bronx','staten island','tribeca','soho','ues','uws','williamsburg','dumbo','lic','chelsea','flatiron','astoria','greenpoint']
    return any(k in s for k in keys)
def feeds():
    return [
        "https://therealdeal.com/section/new-york/feed/",
        "https://www.curbed.com/rss/index.xml",
        "https://www.zillow.com/research/feed/",
        "https://www.mansionglobal.com/feeds/rss",
        "https://blog.marketproof.com/feed/",
        "https://zillow.mediaroom.com/rss?rsspage=research",
        "https://millersamuel.com/feed/"
    ]
def _weight_sources(items, borough_hint=None):
    if not borough_hint: return items
    b=borough_hint.lower()
    out=[]
    for it in items:
        t=(it.get('title') or '').lower()
        if 'brooklyn' in b and any(k in t for k in ['brooklyn','williamsburg','greenpoint','dumbo']): out.insert(0,it)
        elif 'manhattan' in b and any(k in t for k in ['manhattan','chelsea','soho','tribeca','ues','uws']): out.insert(0,it)
        elif 'queens' in b and any(k in t for k in ['queens','astoria','lic','long island city']): out.insert(0,it)
        else: out.append(it)
    return out
def fetch_news(max_items=6, nyc_only=True, borough_hint=None):
    now=time.time()
    # Always rebuild (keep fresh) within session; keep simple cache
    items=[]
    for url in feeds():
        try:
            fp=feedparser.parse(url)
            for e in fp.entries[:12]:
                title=e.get('title',''); link=e.get('link',''); summary=BeautifulSoup(e.get('summary',''), 'html.parser').get_text(' ', strip=True)
                if nyc_only and not (_within_ny(title) or _within_ny(summary)):
                    if 'zillow.com/research' in url or 'millersamuel.com' in url:
                        pass
                    else:
                        continue
                items.append({'title':title,'link':link,'summary':summary})
        except Exception: pass
    try:
        html=requests.get("https://www.urbandigs.com/blog/", timeout=10).text
        soup=BeautifulSoup(html,'html.parser')
        h=soup.find('h2') or soup.find('h1')
        if h:
            a=h.find('a')
            if a and a.get('href'): items.append({'title':h.get_text(strip=True),'link':a.get('href'),'summary':''})
    except Exception: pass
    seen=set(); dedup=[]
    for it in items:
        ttl=it.get('title','')
        if ttl in seen: continue
        seen.add(ttl); dedup.append(it)
    dedup=_weight_sources(dedup, borough_hint)
    return dedup[:max_items]
def fetch_market_snapshot(area="NYC"):
    snap={'area':area,'bullets':[]}
    try:
        fp=feedparser.parse("https://blog.marketproof.com/feed/")
        for e in fp.entries[:8]:
            if _within_ny(e.get('title','')) or _within_ny(e.get('summary','')):
                snap['bullets'].append(f"Marketproof: {e.get('title','').strip()}"); break
    except Exception: pass
    try:
        fp=feedparser.parse("https://www.zillow.com/research/feed/")
        for e in fp.entries[:8]:
            ttl=e.get('title','')
            if any(k in ttl.lower() for k in ['nyc','new york','manhattan','brooklyn','queens']):
                snap['bullets'].append(f"Zillow: {ttl.strip()}"); break
    except Exception: pass
    try:
        fp=feedparser.parse("https://millersamuel.com/feed/")
        if fp.entries:
            e=fp.entries[0]; soup=BeautifulSoup(e.get('summary',''), 'html.parser')
            excerpt=soup.get_text(' ', strip=True)[:160]
            snap['bullets'].append(f"Housing Notes: {e.get('title','').strip()} — {excerpt}...")
    except Exception: pass
    snap['bullets']=snap['bullets'][:3] or ["NYC market insights fetched at runtime."]
    return snap
