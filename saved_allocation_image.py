"""On-demand images of private saved weights; no optimizer or market requests."""
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
import math
import hashlib
import streamlit as st
from threading import RLock
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo
from textwrap import wrap
from allocation_image_metrics import saved_image_metrics, definition_ribbon, image_metric_config
from selected_image_metrics import selected_image_statistics

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


WIDTH = 1272
ROWS_PER_PANEL = 24
_RENDER_LOCK = RLock()
_IMAGE_STATS_VERSION = "selected-intact-blocks-v2"


@dataclass(frozen=True)
class SavedWeightsImage:
    rows: tuple  # exact listing, native currency, saved fraction, active pause
    saved_label: str
    date_slug: str
    history_label: str
    scope_label: str
    review_only: bool
    statistics: tuple = ()
    horizon: int | None = None
    stats_retained: bool = False
    include_ribbon: bool = True
    view: str = "Global"
    stats_recomputed: bool = False
    stats_detail: str = ""

    def panels(self):
        currencies=sorted({r[1] for r in self.rows},key=lambda c:({"INR":0,"USD":1}.get(c,2),c))
        result=[]
        for currency in currencies:
            group=sorted((r for r in self.rows if r[1]==currency),key=lambda r:(-r[2],r[0]))
            for start in range(0,len(group),ROWS_PER_PANEL):
                result.append((currency,tuple(group[start:start+ROWS_PER_PANEL]),start>0))
        return tuple(result)


def saved_weights_image_spec(payload, paused=()):
    currencies={}
    for row in payload.get("current_allocation") or []:
        ticker=str(row.get("Yahoo Ticker") or "").strip().upper()
        currency=str(row.get("Currency") or "").strip().upper()
        if ticker and currency:
            currencies.setdefault(ticker,set()).add(currency)
    active={str(t).strip().upper() for t in paused}
    rows=[];seen=set()
    for row in payload.get("optimal_allocation") or []:
        ticker=str(row.get("ticker") or "").strip().upper()
        weight=float(row.get("target_weight"))
        if not ticker or any(ord(c)<32 for c in ticker) or ticker in seen or not math.isfinite(weight) or not 0<=weight<=1:
            raise ValueError("Saved weights contain an invalid or duplicate listing")
        seen.add(ticker)
        if not weight:
            continue
        codes=currencies.get(ticker,set())
        currency=next(iter(codes)) if len(codes)==1 else "UNAVAILABLE"
        if not codes and ticker.endswith((".NS",".BO")):
            currency="INR"
        rows.append((ticker,currency,weight,ticker in active))
    if not rows:
        raise ValueError("No positive saved optimal weights are available")
    try:
        date=datetime.fromisoformat(str(payload.get("saved_at")).replace("Z","+00:00"))
        if date.tzinfo:
            date=date.astimezone(ZoneInfo("Asia/Kolkata"))
            saved_label=date.strftime("%d %b %Y · %H:%M IST")
        else:
            saved_label=date.strftime("%d %b %Y")+" · timezone unrecorded"
        slug=date.strftime("%Y-%m-%d")
    except (ValueError,TypeError):
        saved_label="Saved date unavailable";slug="saved"
    cutoff=str(payload.get("portfolio_metrics_as_of") or (payload.get("history") or {}).get("valid_end") or "")[:10]
    try:
        history_label=datetime.strptime(cutoff,"%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        history_label="Unavailable"
    scope=payload.get("optimization_scope") or {}
    scoped=scope.get("mode")=="unpaused-capital-v1" and bool(scope.get("excluded_tickers"))
    conditional=any((payload.get(k) or {}).get("status")=="conditional_research" for k in ("pause_allocation_research","allocation_policy"))
    review=conditional or (payload.get("allocation_validation") or {}).get("passed") is not True
    review=review or not math.isclose(sum(r[2] for r in rows),1.,abs_tol=1e-6)
    statistics,horizon=saved_image_metrics(payload)
    return SavedWeightsImage(tuple(rows),saved_label,slug,history_label,
        "Unpaused capital only · paused holdings excluded" if scoped else "Saved allocation · portfolio weights",review,
        statistics,horizon,bool(payload.get("portfolio_metrics_refresh_error")))


def render_saved_weights_page(spec, page=0):
    """Same cream/serif share-card style, with exact weights and bounded pages."""
    panels=spec.panels();pages=(len(panels)+1)//2
    if not 0<=page<pages:
        raise ValueError("Invalid allocation-image page")
    selected=panels[page*2:page*2+2]
    stats_height=334
    height=max(1740 if spec.include_ribbon else 1300,
               1414+max(len(p[1]) for p in selected)*52-(0 if spec.include_ribbon else 440))
    paper,ink,muted,accent,rule="#f5f0e6","#29251f","#6b665e","#9f4339","#d8cfbf"
    # Keep concurrent session renders and download workers isolated in Matplotlib.
    with _RENDER_LOCK:
        figure=Figure(figsize=(WIDTH/100,height/100+1e-6),dpi=100,facecolor=paper)
        FigureCanvasAgg(figure)
        def text(x,y,value,size=14,*,color=ink,bold=False,align="left",family="sans-serif",max_width=None):
            item=figure.text(x,1-y/height,value,fontsize=size,color=color,ha=align,va="center",
                             fontweight="bold" if bold else "normal",family=family,parse_math=False)
            if max_width:
                renderer=figure.canvas.get_renderer()
                extent=item.get_window_extent(renderer).width
                if extent>max_width:
                    item.set_fontsize(size*max_width/extent)
            return item
        def line(left,right,y,color=rule):
            figure.lines.append(Line2D((left,right),(1-y/height,)*2,
                transform=figure.transFigure,color=color,linewidth=.8))
        left,right=.06,.94
        text(left,60,"PRIVATE PORTFOLIO",15,color=accent,bold=True)
        text(right,60,"REVIEW ONLY" if spec.review_only else "SAVED MODEL",12,color=accent,bold=True,align="right")
        text(left,118,"SAVED OPTIMAL WEIGHTS",27,bold=True,family="serif")
        text(left,170,spec.saved_label+(" · "+spec.view+" view" if spec.view!="Global" else ""),13,color=muted)
        line(left,right,210,color=ink)
        for x,label,value in ((left,"SECURITIES",str(len(spec.rows))),(.36,"TOTAL ALLOCATION" if spec.view=="Global" else "SHOWN ALLOCATION",f"{sum(r[2] for r in spec.rows):.2%}"),(.67,"HISTORY THROUGH",spec.history_label)):
            text(x,254,label,11,color=muted,bold=True)
            text(x,295,value,20,bold=True)
        text(left,349,spec.scope_label,13,color=muted)
        line(left,right,375)
        text(left,402,"PORTFOLIO STATISTICS · CURRENT → OPTIMAL",14,color=accent,bold=True)
        text(left,430,("Recomputed selected mix · each side normalized to 100% · INR returns · before costs/taxes" if spec.stats_recomputed else
             "Saved analysis · before costs/taxes · fitted weights may differ from rounded allocations" if spec.view=="Global" else
             "Full saved-allocation statistics · before costs/taxes · view weights retain their original portfolio base")+
             (" · prior dated statistics retained" if spec.stats_retained else ""),11.5,color=muted,max_width=(right-left)*WIDTH)
        for i,(label,current,optimal,note) in enumerate(spec.statistics):
            x=left+(i%4)*.225;y=467+(i//4)*110
            text(x,y,label,10.5,color=muted,bold=True,max_width=.21*WIDTH)
            value=f"{current} → {optimal}" if current is not None else optimal
            text(x,y+34,value,16,bold=True,max_width=.21*WIDTH)
            text(x,y+66,note,10.2,color=muted,max_width=.21*WIDTH)
        if spec.stats_recomputed:
            text(left,671,spec.stats_detail,10.2,color=muted,max_width=(right-left)*WIDTH)
        elif spec.view!="Global":
            text(left,671,"Saved fitted weights can differ from rounded allocations.",10.2,color=muted)
        line(left,right,686)
        panel_width=.415 if len(selected)==2 else .88
        for index,(currency,rows,continued) in enumerate(selected):
            x=left+index*.465;end=x+panel_width
            heading=(currency if currency!="UNAVAILABLE" else "CURRENCY UNAVAILABLE")+(" · continued" if continued else "")
            text(x,412+stats_height,heading,15,color=accent,bold=True)
            text(x,452+stats_height,"SECURITY",11,color=muted,bold=True)
            text(end,452+stats_height,"ALLOCATION",11,color=muted,bold=True,align="right")
            line(x,end,476+stats_height,color=ink)
            for n,(ticker,_,weight,paused) in enumerate(rows):
                y=512+stats_height+n*52
                if n%2:
                    figure.patches.append(Rectangle((x,1-(y+22)/height),panel_width,44/height,
                        transform=figure.transFigure,facecolor="#f0ebe1",edgecolor="none"))
                text(x+.005,y,ticker+(" *" if paused else ""),19,bold=True,max_width=(panel_width-.14)*WIDTH)
                text(end-.005,y,"<0.01%" if weight<.0001 else f"{weight:.2%}",19,color=accent,bold=True,align="right")
        if spec.include_ribbon:
            figure.patches.append(Rectangle((left-.012,155/height),right-left+.024,440/height,
                transform=figure.transFigure,facecolor="#eee6d7",edgecolor="none"))
            text(left,height-568,"HOW TO READ THESE NUMBERS",14,color=accent,bold=True)
            for i,(term,definition) in enumerate(definition_ribbon(spec.horizon,recalculated=spec.stats_recomputed)):
                x=left+(i%2)*.465;y=height-526+(i//2)*90
                text(x,y,term,12,bold=True,max_width=.415*WIDTH)
                for j,part in enumerate(wrap(definition,66)):
                    text(x,y+23+j*20,part,11.1,color=muted,max_width=.415*WIDTH)
            text(left,height-142,"Concepts: Sharpe (1994) · Künsch (1989) · Rockafellar & Uryasev (2000). App conventions noted above.",9.5,color=muted)
        line(left,right,height-116)
        text(left,height-82,"Saved model weights · not executed trades. Historical and resampled risks are not loss limits.",11.5,color=muted)
        if any(r[3] for r in spec.rows):
            text(left,height-49,"* Pause marked in this view · saved weights retained; trading remains subject to current checks.",11.5,color=muted)
        text(right,height-32,f"{page+1} / {pages}",11,color=muted,align="right")
        output=BytesIO()
        figure.canvas.print_png(output)
        figure.clear()
        return output.getvalue()


def saved_weights_image_download(spec):
    pages=(len(spec.panels())+1)//2
    if pages==1:
        return render_saved_weights_page(spec)
    output=BytesIO()
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for page in range(pages):
            archive.writestr(f"optimal-weights-{spec.date_slug}-{page+1:02d}.png",render_saved_weights_page(spec,page))
    return output.getvalue()


def render_weight_views(spec, views, include_ribbon=True, *, payload=None, history=None):
    """One image set per view with independently recalculated selected statistics."""
    if not isinstance(include_ribbon,bool) or any(v not in ("INR","US","Global") for v in views):
        raise ValueError("Choose INR, US or Global and a definitions-ribbon setting")
    images=[];missing=[]
    for view in dict.fromkeys(views):
        rows=tuple(r for r in spec.rows if view=="Global" or r[1]==("INR" if view=="INR" else "USD"))
        if not rows:
            missing.append(view);continue
        selected=replace(spec,rows=rows,include_ribbon=include_ribbon,view="US / USD quotes" if view=="US" else view)
        statistics,horizon,details=selected_image_statistics(payload or {},rows,history)
        detail="Original-base allocation weights · "+str(details["observations"])+" common returns"
        if details["excluded_returns"]:
            detail+=f" · {details['excluded_returns']} incomplete returns excluded; intact-block resampling"
        if details["error"]:
            detail=details["error"]
        selected=replace(selected,statistics=statistics,horizon=horizon,stats_recomputed=True,
                         stats_retained=False,history_label=details["history_label"],stats_detail=detail)
        pages=(len(selected.panels())+1)//2
        for page in range(pages):
            name=f"optimal-weights-{view.lower()}-{spec.date_slug}"+(f"-{page+1:02d}" if pages>1 else "")+".png"
            images.append((name,render_saved_weights_page(selected,page),view+ (f" · page {page+1}/{pages}" if pages>1 else "")))
    return tuple(images),tuple(missing)


def pack_weight_views(images, date_slug):
    if len(images)==1:
        return images[0][0],"image/png",images[0][1]
    if not images:
        raise ValueError("No rendered allocation images")
    output=BytesIO()
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for name,data,_ in images:
            archive.writestr(name,data)
    return f"optimal-weights-views-{date_slug}.zip","application/zip",output.getvalue()


@st.fragment
def render_saved_weights_download(payload, paused=(), *, key, history_loader=None):
    try:
        spec=saved_weights_image_spec(payload,paused)
    except (ValueError,TypeError,OverflowError) as exc:
        st.caption(f"Weights image unavailable: {exc}")
        return
    signature=hashlib.sha256(repr((_IMAGE_STATS_VERSION,spec,payload.get("portfolio_evidence_history"),
        payload.get("current_allocation"),image_metric_config(payload))).encode()).hexdigest()
    cache=st.session_state.setdefault("saved_weights_image_previews",{})
    if key in cache and cache[key]["signature"]!=signature:
        cache.pop(key)
    with st.form(key+"_options",border=False):
        views=st.multiselect("Image coverage",["INR","US","Global"],default=["Global"],key=key+"_views",
            help="Choose one or more image sets. INR = INR quotes; US = USD quotes (including USD-quoted crypto); Global = all currencies. Allocation weights retain their original base. Statistics are recalculated for the selected listings, each mix normalized to 100%, using INR-valued history.")
        ribbon=st.checkbox("Include definitions ribbon",value=True,key=key+"_ribbon")
        render=st.form_submit_button("Render preview",icon=":material/image:")
    if render:
        cache.pop(key,None)
        if not views:
            st.info("Select at least one view, then render the preview.")
            return
        with st.spinner("Rendering saved weights…"):
            history=payload.get("portfolio_evidence_history")
            if not history and history_loader is not None:
                try:
                    history=history_loader()
                except (ValueError,TypeError,KeyError,OSError):
                    history=None
            images,missing=render_weight_views(spec,views,ribbon,payload=payload,history=history)
        for view in missing:
            st.info(f"{view}: no positive saved weights with matching quote currency.")
        if not images:
            return
        if key not in cache and len(cache)>=2:
            cache.pop(next(iter(cache)))
        cache[key]={"signature":signature,"images":images,"file":pack_weight_views(images,spec.date_slug),"ribbon":ribbon}
    rendered=cache.get(key)
    if rendered:
        st.caption("Rendered views · "+", ".join(dict.fromkeys(label.split(" · ")[0] for _,_,label in rendered["images"]))+
                   " · Definitions "+("included" if rendered["ribbon"] else "omitted")+". Download matches this preview.")
        for _,data,label in rendered["images"]:
            with st.expander(label+" preview",expanded=len(rendered["images"])==1):
                st.image(data,width=636)
        name,mime,data=rendered["file"]
        st.download_button("Download rendered image" if mime=="image/png" else "Download rendered images (ZIP)",
            data=data,file_name=name,mime=mime,key=key+"_download",icon=":material/download:",on_click="ignore")
