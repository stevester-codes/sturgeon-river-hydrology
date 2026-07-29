from datetime import datetime,timezone
from sturgeon_pipeline import FNAME,HREF
html='<a href="CMC_HRDPA_WATERSHED-006-0700cutoff_SFC_0_ps2.5km_2026072800_000_05.dbf">x</a>'
links=HREF.findall(html)
assert len(links)==1
m=FNAME.search(links[0]); assert m
assert m.group('accum')=='006' and m.group('cutoff')=='0700' and m.group('group')=='05'
assert datetime.strptime(m.group('valid'),'%Y%m%d%H').replace(tzinfo=timezone.utc)==datetime(2026,7,28,0,tzinfo=timezone.utc)
print('Parsing tests passed')
