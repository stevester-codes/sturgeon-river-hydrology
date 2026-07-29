#!/usr/bin/env python3
from __future__ import annotations
import csv,json,re
from datetime import datetime,timezone,timedelta
from pathlib import Path

STATIONS=['05EA002','05EA005','05EA006','05EA004','05EA010','05EA011','05EA012']
VALID_RE=re.compile(r'_(\d{10})_000_\d{2}\.dbf$')


def norm(s): return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

def find_key(fields,*needles):
    n={norm(f):f for f in fields}
    for needle in needles:
        q=norm(needle)
        for k,v in n.items():
            if q==k or q in k: return v
    return None

def parse_dt(v):
    v=(v or '').strip()
    for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%d %H:%M','%Y-%m-%dT%H:%M:%SZ'):
        try:
            d=datetime.strptime(v,fmt)
            return d.replace(tzinfo=timezone.utc)
        except ValueError: pass
    try:
        d=datetime.fromisoformat(v.replace('Z','+00:00'))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception: return None

def parse_float(v):
    try: return float(str(v).strip())
    except Exception: return None

def read_gauge(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows: return []
    fields=list(rows[0])
    dk=find_key(fields,'Date')
    pk=find_key(fields,'Parameter')
    vk=find_key(fields,'Value')
    if not (dk and pk and vk):
        return [{'error':f'Unrecognized columns: {fields}'}]
    data=[]
    for r in rows:
        d=parse_dt(r.get(dk)); v=parse_float(r.get(vk)); p=str(r.get(pk,'')).strip()
        if d is None or v is None: continue
        data.append((d,p,v))
    out=[]
    for code,label in [('46','water_level_m'),('47','discharge_m3s')]:
        series=sorted((d,v) for d,p,v in data if p==code or (code=='46' and 'level' in p.lower()) or (code=='47' and ('discharge' in p.lower() or 'flow' in p.lower())))
        if not series: continue
        latest_t,latest_v=series[-1]
        def nearest_before(hours):
            target=latest_t-timedelta(hours=hours)
            prior=[x for x in series if x[0]<=target]
            return prior[-1] if prior else None
        p24=nearest_before(24); p72=nearest_before(72)
        out.append({'metric':label,'latest_utc':latest_t.isoformat(),'latest':latest_v,
                    'change_24h':None if not p24 else latest_v-p24[1],
                    'change_72h':None if not p72 else latest_v-p72[1]})
    return out

def precip_summary(path):
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    parsed=[]
    for r in rows:
        m=VALID_RE.search(r.get('_source_file',''))
        if not m: continue
        d=datetime.strptime(m.group(1),'%Y%m%d%H').replace(tzinfo=timezone.utc)
        p=parse_float(r.get('PR_mm')); c=parse_float(r.get('CFIA')); a=parse_float(r.get('Shp_Area'))
        if p is None: continue
        parsed.append((str(r.get('Station','')).strip(),d,p,c,a))
    if not parsed: return []
    latest=max(x[1] for x in parsed)
    start=latest-timedelta(hours=24)
    out=[]
    for st in STATIONS:
        xs=[x for x in parsed if x[0]==st and start < x[1] <= latest]
        if not xs: continue
        out.append({'station':st,'period_end_utc':latest.isoformat(),'period_hours':24,
                    'precip_mm':sum(x[2] for x in xs),'n_6h_periods':len(xs),
                    'mean_cfia':sum(x[3] for x in xs if x[3] is not None)/max(1,sum(x[3] is not None for x in xs)),
                    'area_km2':next((x[4] for x in xs if x[4] is not None),None)})
    return out

def write_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(''); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    root=Path('sturgeon_pipeline_output'); summary=root/'summary'; summary.mkdir(parents=True,exist_ok=True)
    gauges=[]
    for st in STATIONS:
        p=root/'raw/wateroffice'/f'{st}.csv'
        if not p.exists(): continue
        for r in read_gauge(p): r['station']=st; gauges.append(r)
    rain=precip_summary(root/'processed/watershed_precip_06h.csv') if (root/'processed/watershed_precip_06h.csv').exists() else []
    write_csv(gauges,summary/'latest_gauges.csv'); write_csv(rain,summary/'recent_precip_24h.csv')
    target=next((r for r in gauges if r.get('station')=='05EA002' and r.get('metric')=='water_level_m'),None)
    payload={'generated_utc':datetime.now(timezone.utc).isoformat(),'target_05EA002':target,'gauges':gauges,'recent_precip_24h':rain}
    (summary/'summary.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2))

if __name__=='__main__': main()
