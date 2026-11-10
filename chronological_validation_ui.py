"""Independent research block; no optimizer on page load or publication writes."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from chronological_pipeline import cached_research_panel, make_pipeline_fitter, research_universe, source_fingerprint
from chronological_validation import atomic_save, experiment_id, run_validation


def render_report(state):
    summary = state.get("summary") or {}
    status = state.get("status", "paused")
    total = int(state.get("settings", {}).get("folds", 3)) * 2
    completed = len(state.get("folds") or [])
    st.markdown(f"**{completed}/{total} fold/strategy evaluations complete · {status}**")
    st.progress(min(completed / max(total, 1), 1.))
    st.caption(f"Experiment cutoff {state.get('as_of', 'unknown')} · INR · joint NSE/NYSE session clock")
    coverage = state.get("coverage") or {}
    if coverage:
        st.caption(f"Cached universe coverage: {coverage.get('included', 0):,} series included · "
                   f"{coverage.get('omitted', 0):,} unavailable/unsupported · no provider downloads")
        st.caption(f"Cached prices through: {coverage.get('data_through', 'unknown')}")
    if state.get("error"):
        st.warning(state["error"] + " Earlier completed trials are retained.")
    if summary.get("strategies"):
        st.info(summary["decision"])
        columns = st.columns(2)
        for column, strategy, label in zip(columns, ("incumbent", "risk-aware"), ("Incumbent", "Risk-aware")):
            metrics = summary["strategies"][strategy]
            with column.container(border=True):
                st.markdown("**" + label + " · unseen periods**")
                st.metric("Net return after costs", f"{metrics['net_return']:+.2%}")
                st.write(f"Max drawdown **{metrics['max_drawdown']:.2%}** · observed one-session ES95 **{metrics['observed_one_session_es95']:.2%}**")
                st.caption(f"Gross turnover {metrics['gross_turnover']:.1%} · modeled costs {metrics['cost_fraction']:.3%} of starting capital")
                st.caption(f"Fit/preparation work {metrics['fit_seconds']:.1f}s · {metrics['held_folds']} folds held the prior mix after no feasible fit")
    else:
        st.caption("Run / resume continues from the saved trial checkpoints; completed optimizer fits are not repeated.")
    if state.get("folds"):
        with st.expander("Fold evidence", expanded=False):
            st.dataframe(pd.DataFrame([{
                "Fold": r["fold"], "Allocator": r["strategy"], "Training through": r["train_end"],
                "Execution close": r["execution_date"], "Test through": r["test_end"],
                "Selected X × Y": " × ".join(map(str, r["pair"])) if r.get("pair") else "Hold prior mix",
                "Net return": r["net_return"], "Gross turnover": r["gross_turnover"],
                "Fit": r["fit_status"], "Reason": r.get("reason"),
            } for r in state["folds"]]), hide_index=True, width="stretch", column_config={
                "Net return": st.column_config.NumberColumn(format="percent"),
                "Gross turnover": st.column_config.NumberColumn(format="percent"),
            })
    with st.expander("Scope and limitations", expanded=False):
        st.write("Both allocators start from the same equal-weight basket of up to 50 securities, chosen round-robin across listing clusters and by training traded value within each cluster. Each then carries its own drifting holdings. These are research portfolios, not a reconstruction of your broker account.")
        st.write("Market-cap evidence dated/imported after a training cutoff is unavailable there. Historical pauses/membership and delisted securities are not reconstructed. Today's saved winning pair and holdings are never used to start a historical search.")
        for limitation in state.get("limitations") or []:
            st.caption(limitation)
        st.caption("Only the preceding training prefix determines each selection. Trades are modeled at the next observed session close. Missing open-session prices withhold the fold rather than becoming zero returns. ES95 is empirical one-session loss; a few 30-session test blocks cannot establish a reliable 30-session 95% tail estimate.")
        st.caption("No automatic promotion, publication or live portfolio change occurs.")
    st.download_button("Download research evidence JSON", data=json.dumps(state, allow_nan=False, indent=2).encode(),
                       file_name="chronological-research.json", mime="application/json")


@st.fragment
def render_chronological_validation(owner, database_path, cache_path, search_config):
    with st.expander("Whole-process chronological validation · P2", expanded=False):
        st.caption("Rebuild the shortlist, history/momentum filters, duplicate screen, X/Y search and allocator on each training prefix. Compare the following 30 sessions after costs. Independent of your live optimization.")
        owner_key = hashlib.sha256(str(owner).encode()).hexdigest()[:20]
        folder = Path(database_path).parent / "chronological_research" / owner_key
        pointer = folder / "latest.json"
        state = None
        pointer_info = {}
        if pointer.exists():
            try:
                pointer_info = json.loads(pointer.read_text())
                run_id = pointer_info["run_id"]
                if len(run_id) != 64 or any(c not in "0123456789abcdef" for c in run_id):
                    raise ValueError("Invalid research report identifier")
                state = json.loads((folder / (run_id + ".json")).read_text())
            except (OSError, ValueError, KeyError):
                st.caption("Previous research report could not be read; existing analysis is unchanged.")
        with st.expander("Test settings", expanded=False):
            saved_settings = (state or {}).get("settings") or {}
            as_of = st.date_input("Research evidence cutoff", value=pd.Timestamp((state or {}).get("requested_as_of") or pointer_info.get("requested_as_of") or (state or {}).get("as_of") or pd.Timestamp.now(tz="Asia/Kolkata").date() - pd.Timedelta(days=1)).date(), key=f"p2_date_{owner_key}")
            folds = int(st.number_input("Non-overlapping 30-session test blocks", min_value=2, max_value=12, value=int(saved_settings.get("folds", 3)), step=1, key=f"p2_folds_{owner_key}"))
            trials = int(st.number_input("Maximum X/Y fits per allocator per fold", min_value=1, max_value=32, value=int(saved_settings.get("trials_per_fold", 8)), step=1, key=f"p2_trials_{owner_key}"))
            current_sidebar = st.checkbox("Use current sidebar search settings for a new experiment", value=False, key=f"p2_sidebar_{owner_key}")
            selected_search = search_config if current_sidebar or not saved_settings.get("search") else dict(
                saved_settings["search"], transaction_cost_bps=saved_settings["cost_bps"])
            if selected_search.get("minimum_cap"):
                st.caption(f"Research X range: {selected_search['minimum_cap']:,}–{selected_search['maximum_cap']:,}. Saved research settings survive restart; current sidebar settings are used only when selected above.")
            st.caption("120-second scheduling budget per click. An in-progress bounded solve/preparation may finish afterwards. Resume until all folds finish. Changes to evidence, code or settings create a separate experiment.")
        if st.button("Run / resume chronological test", key=f"p2_run_{owner_key}", icon=":material/science:"):
            try:
                import portfolio_optimizer_core as core_module
                from types import SimpleNamespace
                # Streamlit can retain the AST adapter from an earlier deploy.
                # Load a private source snapshot, without modifying another run's
                # functions or relying on stale setdefault-exported callables.
                core = SimpleNamespace(**core_module._load_core_namespace())
                indicator, detail = st.progress(0.), st.empty()

                def progress(message, fraction):
                    detail.info(message)
                    if fraction is not None:
                        indicator.progress(min(max(float(fraction), 0.), 1.))

                with st.spinner("Loading existing cached research inputs…"):
                    universe = research_universe(core, owner)
                    prices, volumes, metadata, coverage = cached_research_panel(
                        cache_path, universe, as_of, currency_unit=core._currency_major_and_unit_factor)
                keys = ("minimum_cap", "maximum_cap", "step", "adaptive", "max_dd", "target_volatility",
                        "drop_bottom_pct", "redundancy_corr_threshold", "exact_optimizer_asset_caps", "minimum_trading_days")
                settings = {"folds": folds, "test_sessions": 30, "minimum_train": 252,
                            "trials_per_fold": trials, "solver_seconds": 10.,
                            "cost_bps": float(selected_search["transaction_cost_bps"]),
                            "search": {k: selected_search[k] for k in keys},
                            "active_configuration": core.chronological_research_config(),
                            "source_fingerprint": source_fingerprint()}
                run_id = experiment_id(prices, volumes, metadata, settings)
                # Publish the private pointer BEFORE the long fit, so even an
                # abrupt process shutdown exposes the partial checkpoint on open.
                atomic_save(pointer, {"run_id": run_id, "requested_as_of": str(as_of)})
                state = run_validation(prices, volumes, metadata, settings,
                                       make_pipeline_fitter(core, settings), folder / (run_id + ".json"),
                                       on_progress=progress, runtime_seconds=120)
                state.update(coverage=coverage, requested_as_of=str(as_of))
                atomic_save(folder / (run_id + ".json"), state)
                atomic_save(pointer, {"run_id": run_id, "requested_as_of": str(as_of)})
                detail.empty()
                indicator.empty()
                core._app_run_log("whole_process_validation", status=state["status"],
                                  completed=len(state["folds"]), trial_count=sum(len(t) for t in state["trials"].values()),
                                  network_requests=0, live_allocation_changed=False)
            except Exception as exc:
                st.warning(f"Research test could not advance: {exc}. Saved portfolio and earlier research checkpoints are retained.")
        if state:
            render_report(state)
