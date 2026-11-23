"""Validate published source-table structure; this does not reclassify Gaussian logs."""
from pathlib import Path
import csv,collections,math,json,hashlib
P=Path(__file__).resolve().parent
read=lambda n:list(csv.DictReader((P/n).open()))
c=read('cohort_ids.csv');o=read('optimization_outcomes.csv');t=read('recorded_search_times.csv')
assert len(c)==200 and len({(x['dataset'],x['stem']) for x in c})==200
assert len(o)==800 and len({(x['dataset'],x['reaction'],x['workflow']) for x in o})==800
assert all(x['opt_freq_accepted'] in ['0','1'] for x in o)
expected={'reactot':[99,97,98,90],'gsm':[89,95,87,86]};ws=['Gaussian--CalcFC','Gaussian--CalcAll','OneShot','External--CalcAll']
for ds,counts in expected.items():
 for wf,count in zip(ws,counts):
  rows=[x for x in o if x['dataset']==ds and x['workflow']==wf];assert len(rows)==100;assert sum(int(x['opt_freq_accepted']) for x in rows)==count
assert len(t)==len(read('recorded_timing_events.csv'))==600
for x in t:
 if x['recorded_search_s']:assert math.isfinite(float(x['recorded_search_s'])) and float(x['recorded_search_s'])>=0
meta=json.loads((P/'provenance.json').read_text())
for n,h in meta['public_file_sha256'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
print('PASS: cohort keys, all attempted/nonaccepted outcomes, timing coverage and published file integrity.')
