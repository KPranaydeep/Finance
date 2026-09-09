"""Synthetic in-process UI test only. Never a production data source."""
from datetime import datetime, timezone
import streamlit as st
from public_review.ui import render_events
from public_review.core import evaluate, decision
from review_fixtures import policy, baseline

now=datetime(2026,9,9,13,tzinfo=timezone.utc)
b=baseline(); p=policy()
m=evaluate(b,{'A.NS':105,'B.NS':51},'2026-09-09',p)
d=decision(m,b,b['weights'],p,10000)
payload={'checked_at':now.isoformat(),'as_of':'2026-09-09','metrics':m,'decision':d,
         'forecast':{'status':'RESEARCH_ONLY'},'policy':p,'validation':{'passed':False},
         'comparisons':[{'option':'No trade','cash_raised':0.,'fees':0.,'tax':0.,'orders':[]}],
         'mmi':{'status':'UNAVAILABLE'},'dividend_assumption':'Synthetic test'}
events=[{'kind':'BASELINE','baseline_id':b['baseline_id'],'seq':1,'payload':b},
        {'kind':'ASSESSMENT','baseline_id':b['baseline_id'],'seq':2,'payload':payload,'event_hash':'test-only'}]
mode=st.session_state.get('fixture_mode','normal')
if mode=='stale': now=datetime(2026,9,12,13,tzinfo=timezone.utc)
if mode=='failure': events.append({'kind':'FAILURE','baseline_id':b['baseline_id'],'seq':3,'payload':{'reason':'STALE_DATA'}})
if mode=='recovered':
    events.append({'kind':'FAILURE','baseline_id':b['baseline_id'],'seq':3,'payload':{'reason':'STALE_DATA'}})
    events.append({'kind':'HEARTBEAT','baseline_id':b['baseline_id'],'seq':4,'payload':{'at':now.isoformat(),'alerts_enabled':False}})
if mode=='empty': events=[]
if mode=='second':
    from copy import deepcopy
    other=deepcopy(b); other['baseline_id']='other'; other['portfolio_version']=3; other['publication_id']='PUB-OTHER'
    events.append({'kind':'BASELINE','baseline_id':'other','seq':3,'payload':other})
render_events(events,now=now)

