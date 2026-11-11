from copy import deepcopy
from io import BytesIO
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import pytest

from saved_allocation_image import (saved_weights_image_spec,render_saved_weights_page,
                                    saved_weights_image_download,render_saved_weights_download)


def payload(count=26):
    return {"saved_at":"2026-11-10T20:00:00+00:00",
            "optimal_allocation":[{"ticker":f"SECURITY{i:02d}"+(".NS" if i%2 else ""),"target_weight":1/count} for i in range(count)],
            "current_allocation":[{"Yahoo Ticker":f"SECURITY{i:02d}"+(".NS" if i%2 else ""),"Currency":"INR" if i%2 else "USD"} for i in range(count)],
            "allocation_validation":{"passed":True},"history":{"valid_end":"2026-11-09"},
            "optimization_scope":{"mode":"unpaused-capital-v1","excluded_tickers":["PAUSED.NS"]}}


def test_export_preserves_exact_saved_weights_and_snapshot_without_mutation():
    source=payload();before=deepcopy(source)
    spec=saved_weights_image_spec(source,["SECURITY00"])
    assert source==before
    assert spec.rows[0]==("SECURITY00","USD",1/26,True)
    assert spec.saved_label=="11 Nov 2026 · 01:30 IST" and spec.date_slug=="2026-11-11"
    assert spec.history_label=="09 Nov 2026" and spec.scope_label.startswith("Unpaused")
    assert not spec.review_only
    assert spec.panels()[0][0]=="INR" and spec.panels()[1][0]=="USD"
    assert [r[2] for p in spec.panels() for r in p[1]]==[1/26]*26


@pytest.mark.parametrize("count,expected",[(1,(1272,1200)),(26,(1272,1316)),(48,(1272,1888))])
def test_exact_png_dimensions_and_no_clipped_or_overlapping_text(monkeypatch,count,expected):
    source=payload(count);spec=saved_weights_image_spec(source,["SECURITY00"])
    original=FigureCanvasAgg.print_png
    def inspect(canvas,*args,**kwargs):
        canvas.draw();renderer=canvas.get_renderer();figure=canvas.figure
        boxes=[t.get_window_extent(renderer) for t in figure.texts]
        for i,box in enumerate(boxes):
            assert figure.bbox.contains(box.x0,box.y0) and figure.bbox.contains(box.x1,box.y1)
            assert not any(box.overlaps(other) for other in boxes[i+1:])
        labels=[t.get_text() for t in figure.texts]
        assert "SAVED OPTIMAL WEIGHTS" in labels and "SAVED MODEL" in labels
        assert "Unpaused capital only · paused holdings excluded" in labels
        assert not any("PUBLIC" in t for t in labels)
        return original(canvas,*args,**kwargs)
    monkeypatch.setattr(FigureCanvasAgg,"print_png",inspect)
    with plt.rc_context({"savefig.bbox":"tight"}):
        data=saved_weights_image_download(spec)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert Image.open(BytesIO(data)).size==expected


def test_dense_export_contains_every_security_in_bounded_png_pages():
    spec=saved_weights_image_spec(payload(101))
    panels=spec.panels()
    assert sum(len(p[1]) for p in panels)==101
    assert len({r[0] for p in panels for r in p[1]})==101
    with ZipFile(BytesIO(saved_weights_image_download(spec))) as archive:
        assert len(archive.namelist())==3
        for name in archive.namelist():
            width,height=Image.open(BytesIO(archive.read(name))).size
            assert width==1272 and 1200<=height<=1888


def test_bad_weights_fail_without_fake_zeroes_and_unknown_metadata_remains_unknown():
    for weight in (None,-.1,float("nan"),float("inf"),1.1):
        source=payload(1);source["optimal_allocation"][0]["target_weight"]=weight
        with pytest.raises((ValueError,TypeError)):
            saved_weights_image_spec(source)
    source=payload(1);source["optimal_allocation"]*=2
    with pytest.raises(ValueError,match="duplicate"):
        saved_weights_image_spec(source)
    source=payload(1);source.update(saved_at=None,current_allocation=[],history={})
    spec=saved_weights_image_spec(source)
    assert spec.rows[0][1]=="UNAVAILABLE" and spec.history_label=="Unavailable"
    assert spec.saved_label=="Saved date unavailable"
    source["pause_allocation_research"]={"status":"conditional_research"}
    assert saved_weights_image_spec(source).review_only


def test_download_is_lazy_captures_snapshot_and_never_runs_an_optimizer(monkeypatch):
    import streamlit as st
    import saved_allocation_image as module
    calls=[];generated=[]
    monkeypatch.setattr(st,"download_button",lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(module,"saved_weights_image_download",lambda spec:generated.append(spec) or b"png")
    source=payload()
    render_saved_weights_download(source,key="saved-image-one")
    assert not generated and calls[0][1]["on_click"]=="ignore"
    assert calls[0][1]["mime"]=="image/png"
    source["optimal_allocation"][0]["target_weight"]=.9
    assert calls[0][1]["data"]()==b"png" and generated[0].rows[0][2]==1/26
