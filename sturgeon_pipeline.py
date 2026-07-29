#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,io,json,logging,re,time,zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
import requests,shapefile
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STATIONS=['05EA002','05EA005','05EA006','05EA004','05EA010','05EA011','05EA012']
WO='https://wateroffice.ec.gc.ca/services/real_time_data/csv/inline'
HOSTS=['dd.weather.gc.ca','dd.meteo.gc.ca']
FNAME=re.compile(r'CMC_HRDPA_WATERSHED-(?P<accum>\d{3})-(?P<cutoff>\d{4})cutoff_SFC_0_ps2\.5km_(?P<valid>\d{10})_000_(?P<group>\d{2})\.dbf')
HREF=re.compile(r'href="([^"?][^"]*)"')


def session(contact:str)->requests.Session:
    s=requests.Session(); s.headers['User-Agent']=f'sturgeon-river-hydrology/1.0 ({contact})'
    r=Retry(total=3,backoff_factor=2,status_forcelist=[403,429,500,502,503,504],allowed_methods=['GET'],respect_retry_after_header=True)
    a=HTTPAdapter(max_retries=r); s.mount('https://',a); return s


def get_text(s,url):
    r=s.get(url,timeout=60); r.raise_for_status(); return r.text


def get_bytes(s,url):
    r=s.get(url,timeout=90); r.raise_for_status(); return r.content


def dates(a,b):
    d=a
    while d<=b:
        yield d; d+=timedelta(days=1)


def fetch_wateroffice(s,start,end,out):
    out.mkdir(parents=True,exist_ok=True); warnings=[]
    for st in STATIONS:
        params=[('stations[]',st),('parameters[]','46'),('parameters[]','47'),('start_date',start.strftime('%Y-%m-%d %H:%M:%S')),('end_date',end.strftime('%Y-%m-%d %H:%M:%S'))]
        try:
            r=s.get(WO,params=params,timeout=90); r.raise_for_status()
            p=out/f'{st}.csv'; p.write_bytes(r.content)
            if r.text.count('\n')<3: warnings.append(f'{st}: WaterOffice response empty')
        except Exception as e: warnings.append(f'{st}: WaterOffice failed: {e}')
        time.sleep(.5)
    return warnings


def datamart_fallback(s,out):
    out.mkdir(parents=True,exist_ok=True); warnings=[]
    for freq in ('hourly','daily'):
        ok=False
        for host in HOSTS:
            url=f'https://{host}/today/hydrometric/csv/AB/{freq}/AB_{freq}_hydrometric.csv'
            try:
                (out/f'AB_{freq}_hydrometric.csv').write_bytes(get_bytes(s,url)); ok=True; break
            except Exception as e: warnings.append(f'{url}: {e}')
        if not ok: warnings.append(f'No {freq} Datamart hydrometric fallback')
    return warnings


def discover_hrdpa(s,start,end,accum):
    found={}
    for pub in dates(start,end+timedelta(days=1)):
        done=False
        for host in HOSTS:
            roots=[f'https://{host}/{pub:%Y%m%d}/WXO-DD/']
            if pub.date()==datetime.now(timezone.utc).date(): roots.append(f'https://{host}/today/')
            for root in roots:
                url=f'{root}analysis/precip/hrdpa_watershed/shapefile/{accum}/'
                try: links=HREF.findall(get_text(s,url))
                except Exception: continue
                for link in links:
                    m=FNAME.search(link)
                    if not m or m.group('accum')!=f'{int(accum):03d}' or m.group('cutoff')!='0700': continue
                    valid=datetime.strptime(m.group('valid'),'%Y%m%d%H').replace(tzinfo=timezone.utc)
                    if start<=valid<=end+timedelta(days=1): found[(valid,m.group('group'))]=(url,link)
                done=True; break
            if done: break
        time.sleep(.3)
    return found


def read_dbf(path):
    r=shapefile.Reader(dbf=str(path)); fields=[x[0] for x in r.fields[1:]]
    return fields,[x.as_dict() for x in r.records()]


def fetch_hrdpa(s,start,end,accum,out):
    out.mkdir(parents=True,exist_ok=True); matches=[]; warnings=[]
    files=discover_hrdpa(s,start,end,accum)
    preferred=[v for k,v in files.items() if k[1]=='05'] or list(files.values())
    for base,name in preferred:
        p=out/name
        try: p.write_bytes(get_bytes(s,base+name))
        except Exception as e: warnings.append(f'{name}: {e}'); continue
        try: fields,rows=read_dbf(p)
        except Exception as e: warnings.append(f'{name} DBF read: {e}'); continue
        sf='Station' if 'Station' in fields else next((f for f in fields if 'stat' in f.lower()),None)
        if not sf: warnings.append(f'{name}: no station field; fields={fields}'); continue
        for row in rows:
            if str(row.get(sf,'')).strip() in STATIONS:
                row['_source_file']=name; matches.append(row)
    if not matches and preferred!=list(files.values()):
        warnings.append('Group 05 had no matches; scan-all fallback required')
    return matches,warnings


def write_csv(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(''); return
    fields=sorted({k for r in rows for k in r})
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def zipdir(src,dst):
    with zipfile.ZipFile(dst,'w',zipfile.ZIP_DEFLATED) as z:
        for p in src.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(src))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--start-date',default='2026-07-01'); ap.add_argument('--end-date'); ap.add_argument('--output-dir',default='sturgeon_pipeline_output'); ap.add_argument('--contact',default='github-actions@users.noreply.github.com'); args=ap.parse_args()
    start=datetime.strptime(args.start_date,'%Y-%m-%d').replace(tzinfo=timezone.utc)
    end=datetime.strptime(args.end_date,'%Y-%m-%d').replace(tzinfo=timezone.utc) if args.end_date else datetime.now(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    s=session(args.contact); warnings=[]
    warnings+=fetch_wateroffice(s,start,end,out/'raw/wateroffice')
    if any('failed' in x.lower() for x in warnings): warnings+=datamart_fallback(s,out/'raw/datamart_hydrometric')
    for accum in ('06','24'):
        rows,w=fetch_hrdpa(s,start,end,accum,out/f'raw/hrdpa_watershed/{accum}'); warnings+=w; write_csv(rows,out/f'processed/watershed_precip_{accum}h.csv')
    (out/'logs').mkdir(exist_ok=True); (out/'logs/missing_periods.log').write_text('\n'.join(warnings) if warnings else 'No warnings.\n')
    status={'run_utc':datetime.now(timezone.utc).isoformat(),'start':str(start.date()),'end':str(end.date()),'warnings':len(warnings)}; (out/'status.json').write_text(json.dumps(status,indent=2))
    z=out.parent/f'{out.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.zip'; zipdir(out,z); print(z)

if __name__=='__main__': main()
