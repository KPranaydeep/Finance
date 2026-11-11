"""On-demand images of private saved weights; no optimizer or market requests."""
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import math
from threading import RLock
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


WIDTH = 1272
ROWS_PER_PANEL = 24
_RENDER_LOCK = RLock()


@dataclass(frozen=True)
class SavedWeightsImage:
    rows: tuple  # exact listing, native currency, saved fraction, active pause
    saved_label: str
    date_slug: str
    history_label: str
    scope_label: str
    review_only: bool

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
    cutoff=str((payload.get("history") or {}).get("valid_end") or "")[:10]
    try:
        history_label=datetime.strptime(cutoff,"%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        history_label="Unavailable"
    scope=payload.get("optimization_scope") or {}
    scoped=scope.get("mode")=="unpaused-capital-v1" and bool(scope.get("excluded_tickers"))
    conditional=any((payload.get(k) or {}).get("status")=="conditional_research" for k in ("pause_allocation_research","allocation_policy"))
    review=conditional or (payload.get("allocation_validation") or {}).get("passed") is not True
    review=review or not math.isclose(sum(r[2] for r in rows),1.,abs_tol=1e-6)
    return SavedWeightsImage(tuple(rows),saved_label,slug,history_label,
        "Unpaused capital only · paused holdings excluded" if scoped else "Saved allocation · portfolio weights",review)


def render_saved_weights_page(spec, page=0):
    """Same cream/serif share-card style, with exact weights and bounded pages."""
    panels=spec.panels();pages=(len(panels)+1)//2
    if not 0<=page<pages:
        raise ValueError("Invalid allocation-image page")
    selected=panels[page*2:page*2+2]
    height=max(1200,640+max(len(p[1]) for p in selected)*52)
    paper,ink,muted,accent,rule="#f5f0e6","#29251f","#6b665e","#9f4339","#d8cfbf"
    # Deferred downloads run on worker threads. Keep Matplotlib rendering isolated.
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
        text(left,170,spec.saved_label,13,color=muted)
        line(left,right,210,color=ink)
        for x,label,value in ((left,"SECURITIES",str(len(spec.rows))),(.36,"TOTAL ALLOCATION",f"{sum(r[2] for r in spec.rows):.2%}"),(.67,"HISTORY THROUGH",spec.history_label)):
            text(x,254,label,11,color=muted,bold=True)
            text(x,295,value,20,bold=True)
        text(left,349,spec.scope_label,13,color=muted)
        panel_width=.415 if len(selected)==2 else .88
        for index,(currency,rows,continued) in enumerate(selected):
            x=left+index*.465;end=x+panel_width
            heading=(currency if currency!="UNAVAILABLE" else "CURRENCY UNAVAILABLE")+(" · continued" if continued else "")
            text(x,412,heading,15,color=accent,bold=True)
            text(x,452,"SECURITY",11,color=muted,bold=True)
            text(end,452,"ALLOCATION",11,color=muted,bold=True,align="right")
            line(x,end,476,color=ink)
            for n,(ticker,_,weight,paused) in enumerate(rows):
                y=512+n*52
                if n%2:
                    figure.patches.append(Rectangle((x,1-(y+22)/height),panel_width,44/height,
                        transform=figure.transFigure,facecolor="#f0ebe1",edgecolor="none"))
                text(x+.005,y,ticker+(" *" if paused else ""),19,bold=True,max_width=(panel_width-.14)*WIDTH)
                text(end-.005,y,"<0.01%" if weight<.0001 else f"{weight:.2%}",19,color=accent,bold=True,align="right")
        line(left,right,height-140)
        text(left,height-104,"Saved model weights · not executed trades. No quantities or prices implied.",12,color=muted)
        if any(r[3] for r in spec.rows):
            text(left,height-68,"* Paused now · saved weights retained; trading remains paused.",12,color=muted)
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


def render_saved_weights_download(payload, paused=(), *, key):
    import streamlit as st
    try:
        spec=saved_weights_image_spec(payload,paused)
    except (ValueError,TypeError,OverflowError) as exc:
        st.caption(f"Weights image unavailable: {exc}")
        return
    multiple=len(spec.panels())>2
    st.download_button("Download optimal weights image"+("s (ZIP)" if multiple else " (PNG)"),
        data=lambda: saved_weights_image_download(spec),
        file_name=f"optimal-weights-{spec.date_slug}."+("zip" if multiple else "png"),
        mime="application/zip" if multiple else "image/png",key=key,icon=":material/image:",on_click="ignore")
