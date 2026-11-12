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


@pytest.mark.parametrize("count,expected,recomputed",[(1,(1272,1740),False),(26,(1272,2090),False),(48,(1272,2662),False),(48,(1272,2662),True)])
def test_exact_png_dimensions_and_no_clipped_or_overlapping_text(monkeypatch,count,expected,recomputed):
    source=payload(count);spec=saved_weights_image_spec(source,["SECURITY00"])
    if recomputed:
        from dataclasses import replace
        spec=replace(spec,stats_recomputed=True,stats_detail="Original-base allocation weights · 300 common returns")
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
            assert width==1272 and 1740<=height<=2662


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


def test_render_then_download_is_on_demand_and_preview_changes_only_after_render(monkeypatch):
    from streamlit.testing.v1 import AppTest
    import saved_allocation_image as module
    generated=[];original=module.render_saved_weights_page
    def render(spec,page=0):
        generated.append(spec);return original(spec,page)
    monkeypatch.setattr(module,"render_saved_weights_page",render)
    app=AppTest.from_file("tests/fixtures/saved_weights_image_app.py",default_timeout=20).run()
    assert not app.exception and not generated and not app.get("download_button")
    assert app.multiselect[0].value==["Global"] and app.checkbox[0].value
    app.multiselect[0].set_value(["INR"]);app.checkbox[0].set_value(False)
    app.button[0].click().run()
    assert not app.exception and len(generated)==1
    assert generated[0].view=="INR" and not generated[0].include_ribbon
    assert generated[0].stats_recomputed and '300 common returns' in generated[0].stats_detail
    assert generated[0].statistics[0][2] not in ('Unavailable','48.38%')
    assert len(app.get("download_button"))==1
    record=app.session_state["saved_weights_image_previews"]["synthetic-image"]
    assert record["file"][1]=="image/png" and record["images"][0][1]==record["file"][2]
    assert sum(r[2] for r in generated[0].rows)==pytest.approx(.5)
    app.multiselect[0].set_value(["US"]).run()
    assert len(generated)==1 and "INR" in app.caption[-1].value
    app.button[0].click().run()
    assert len(generated)==2 and generated[1].view=="US / USD quotes"
    app.multiselect[0].set_value([]);app.button[0].click().run()
    assert not app.exception and not app.get("download_button") and app.info
    app.multiselect[0].set_value(["Global"]);app.button[0].click().run()
    assert app.get("download_button")
    app.session_state["fixture_payload"]=payload(1)
    app.run()
    assert not app.exception and not app.get("download_button")
    assert "synthetic-image" not in app.session_state["saved_weights_image_previews"]


def test_filtered_views_keep_weights_and_stats_and_zip_contains_exact_preview_bytes(monkeypatch):
    import saved_allocation_image as module
    source=saved_weights_image_spec(payload())
    rendered=[]
    monkeypatch.setattr(module,"render_saved_weights_page",lambda spec,page=0:rendered.append(spec) or b"synthetic-png")
    images,missing=module.render_weight_views(source,["INR","US","Global"],False)
    assert not missing and len(images)==3 and sum(r[2] for r in rendered[0].rows)==pytest.approx(.5)
    assert all(r[1:3]==("Unavailable","Unavailable") for r in rendered[0].statistics[:-1])
    assert rendered[0].stats_recomputed and not rendered[0].include_ribbon
    assert len(rendered[2].rows)==26 and source.include_ribbon
    name,mime,data=module.pack_weight_views(images,source.date_slug)
    assert mime=="application/zip"
    with ZipFile(BytesIO(data)) as archive:
        assert archive.namelist()==[row[0] for row in images]
        assert all(archive.read(row[0])==row[1] for row in images)
    only_inr=module.replace(source,rows=tuple(r for r in source.rows if r[1]=="INR"))
    images,missing=module.render_weight_views(only_inr,["US"],True)
    assert not images and missing==("US",)


def test_rendered_multi_page_views_use_distinct_pages(monkeypatch):
    import saved_allocation_image as module
    pages=[]
    def render(spec,page=0):
        pages.append((spec.view,page))
        return str(page).encode()
    monkeypatch.setattr(module,"render_saved_weights_page",render)
    images,missing=module.render_weight_views(saved_weights_image_spec(payload(101)),["Global"])
    assert not missing and pages==[("Global",0),("Global",1),("Global",2)]
    assert [row[1] for row in images]==[b"0",b"1",b"2"]


def test_no_ribbon_layout_and_filtered_statistics_are_explicit(monkeypatch):
    import saved_allocation_image as module
    spec=module.replace(saved_weights_image_spec(payload(48)),include_ribbon=False,view="US / USD quotes")
    original=FigureCanvasAgg.print_png
    def inspect(canvas,*args,**kwargs):
        canvas.draw();renderer=canvas.get_renderer();figure=canvas.figure
        boxes=[t.get_window_extent(renderer) for t in figure.texts]
        for i,box in enumerate(boxes):
            assert figure.bbox.contains(box.x0,box.y0) and figure.bbox.contains(box.x1,box.y1)
            assert not any(box.overlaps(other) for other in boxes[i+1:])
        labels=[t.get_text() for t in figure.texts]
        assert "HOW TO READ THESE NUMBERS" not in labels and "SHOWN ALLOCATION" in labels
        assert any("Full saved-allocation statistics" in label for label in labels)
        assert not any("Concepts:" in label for label in labels)
        return original(canvas,*args,**kwargs)
    monkeypatch.setattr(FigureCanvasAgg,"print_png",inspect)
    data=module.render_saved_weights_page(spec)
    assert Image.open(BytesIO(data)).size==(1272,2222)
