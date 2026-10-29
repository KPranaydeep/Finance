"""Interactive 3D view of the two-dimensional optimizer search."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


SEARCH_COLORSCALE = [
    [0.0, "#18264a"],
    [0.35, "#265f86"],
    [0.68, "#28b7b0"],
    [1.0, "#f4d35e"],
]


def _finite_feasible_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {"Shortlist cap", "Maximum assets", "Annual Return", "Status"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    frame = frame.loc[frame["Status"].astype(str).eq("Feasible")].copy()
    for column in ("Shortlist cap", "Maximum assets", "Annual Return"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Shortlist cap", "Maximum assets", "Annual Return"]
    )


def _marker_trace(row, *, name, color, symbol):
    if not row:
        return None
    try:
        cap = int(row["Shortlist cap"])
        assets = int(row["Maximum assets"])
        annual_return = 100.0 * float(row["Annual Return"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(annual_return):
        return None
    return go.Scatter3d(
        x=[cap], y=[assets], z=[annual_return], mode="markers+text",
        name=name, text=[name], textposition="top center",
        marker={"size": 9, "color": color, "symbol": symbol,
                "line": {"color": "#f7f9ff", "width": 1.5}},
        hovertemplate=(f"<b>{name}</b><br>Shortlist cap %{{x:,}}<br>"
                       "Maximum assets %{y:,}<br>Annual return %{z:.2f}%<extra></extra>"),
    )


def build_search_surface_figure(rows, *, raw_peak=None, robust_choice=None) -> go.Figure:
    """Build a truthful observed-grid surface; missing combinations remain holes."""
    frame = _finite_feasible_frame(rows)
    figure = go.Figure()
    if frame.empty:
        return figure
    caps = sorted(frame["Shortlist cap"].astype(int).unique())
    asset_limits = sorted(frame["Maximum assets"].astype(int).unique())
    pivot = frame.pivot_table(
        index="Maximum assets", columns="Shortlist cap", values="Annual Return",
        aggfunc="max",
    ).reindex(index=asset_limits, columns=caps)
    z_values = pivot.to_numpy(dtype=float) * 100.0
    if len(caps) >= 2 and len(asset_limits) >= 2:
        figure.add_trace(go.Surface(
            x=caps, y=asset_limits, z=z_values, name="Observed return surface",
            connectgaps=False, opacity=0.78,
            colorscale=SEARCH_COLORSCALE,
            colorbar={"title": "Return (%)", "thickness": 14, "len": 0.65},
            contours={"z": {"show": True, "usecolormap": False,
                             "color": "rgba(245,247,255,0.55)", "width": 1,
                             "project_z": True}},
            hovertemplate=("Shortlist cap %{x:,}<br>Maximum assets %{y:,}<br>"
                           "Annual return %{z:.2f}%<extra></extra>"),
        ))
    figure.add_trace(go.Scatter3d(
        x=frame["Shortlist cap"].astype(int),
        y=frame["Maximum assets"].astype(int),
        z=frame["Annual Return"].astype(float) * 100.0,
        mode="markers", name="Feasible observations",
        marker={"size": 4.5, "color": "#63e6ff", "opacity": 0.9,
                "line": {"color": "#dffaff", "width": 0.5}},
        customdata=np.column_stack((
            frame.get("Trading days", pd.Series(index=frame.index, dtype=float)).fillna(0),
            frame.get("Assets", pd.Series(index=frame.index, dtype=float)).fillna(0),
        )),
        hovertemplate=("Shortlist cap %{x:,}<br>Maximum assets %{y:,}<br>"
                       "Annual return %{z:.2f}%<br>Trading sessions %{customdata[0]:,.0f}<br>"
                       "Assets solved %{customdata[1]:,.0f}<extra></extra>"),
    ))
    raw_trace = _marker_trace(raw_peak, name="Raw peak", color="#f4d35e", symbol="diamond")
    robust_trace = _marker_trace(
        robust_choice, name="Robust choice", color="#ff6b9a", symbol="square"
    )
    if raw_trace is not None:
        figure.add_trace(raw_trace)
    if robust_trace is not None:
        figure.add_trace(robust_trace)
    figure.update_layout(
        height=650, margin={"l": 0, "r": 0, "t": 60, "b": 0},
        title={"text": "Observed optimizer landscape", "x": 0.02,
               "xanchor": "left", "font": {"size": 22, "color": "#f2f5fb"}},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d7deeb", "family": "Arial, sans-serif"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "right", "x": 1.0},
        scene={"bgcolor": "#090d18",
               "xaxis": {"title": "Shortlist cap", "gridcolor": "rgba(190,205,230,0.18)"},
               "yaxis": {"title": "Maximum assets", "gridcolor": "rgba(190,205,230,0.18)"},
               "zaxis": {"title": "Annual return (%)", "gridcolor": "rgba(190,205,230,0.18)"},
               "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.05}},
               "aspectmode": "manual", "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.72}},
        hoverlabel={"bgcolor": "#111827", "font": {"color": "#f8fafc"}},
        uirevision="optimizer-search-surface-v1",
    )
    figure.add_annotation(
        text="Observed combinations only · holes are untested · drag to rotate",
        x=0.02, y=0.01, xref="paper", yref="paper", showarrow=False,
        font={"size": 12, "color": "#aab5c8"},
    )
    return figure


def build_search_heatmap_figure(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_peak: Mapping[str, object] | None = None,
    robust_choice: Mapping[str, object] | None = None,
    shortlist_caps: Sequence[int] | None = None,
    maximum_assets_values: Sequence[int] | None = None,
    best_so_far: Mapping[str, object] | None = None,
    next_proposal: Mapping[str, object] | None = None,
    live: bool = False,
) -> go.Figure:
    """Build a full-grid heatmap while leaving unobserved combinations blank."""
    frame = _finite_feasible_frame(rows)
    figure = go.Figure()
    if frame.empty and (shortlist_caps is None or maximum_assets_values is None):
        return figure
    caps = sorted(
        set(int(value) for value in shortlist_caps)
        if shortlist_caps is not None
        else set(frame["Shortlist cap"].astype(int).unique())
    )
    asset_limits = sorted(
        set(int(value) for value in maximum_assets_values)
        if maximum_assets_values is not None
        else set(frame["Maximum assets"].astype(int).unique())
    )
    if not caps or not asset_limits:
        return figure
    cap_labels = [f"{value:,}" for value in caps]
    asset_labels = [f"{value:,}" for value in asset_limits]
    pivot = (
        frame.pivot_table(
            index="Maximum assets", columns="Shortlist cap",
            values="Annual Return", aggfunc="max",
        ).reindex(index=asset_limits, columns=caps)
        if not frame.empty
        else pd.DataFrame(index=asset_limits, columns=caps, dtype=float)
    )
    figure.add_trace(go.Heatmap(
        x=cap_labels,
        y=asset_labels,
        z=pivot.to_numpy(dtype=float) * 100.0,
        colorscale=SEARCH_COLORSCALE,
        colorbar={"title": "Return (%)", "thickness": 14},
        hoverongaps=False,
        xgap=2,
        ygap=2,
        hovertemplate=(
            "Shortlist cap %{x}<br>Maximum assets %{y}<br>"
            "Annual return %{z:.2f}%<extra></extra>"
        ),
    ))

    attempts = pd.DataFrame(rows)
    if not attempts.empty and {"Shortlist cap", "Maximum assets", "Status"}.issubset(attempts.columns):
        attempts = attempts.loc[attempts["Status"].astype(str).ne("Feasible")].copy()
        for column in ("Shortlist cap", "Maximum assets"):
            attempts[column] = pd.to_numeric(attempts[column], errors="coerce")
        attempts = attempts.dropna(subset=["Shortlist cap", "Maximum assets"])
        attempts = attempts.drop_duplicates(
            subset=["Shortlist cap", "Maximum assets"], keep="last"
        )
        attempts = attempts.loc[
            attempts["Shortlist cap"].astype(int).isin(caps)
            & attempts["Maximum assets"].astype(int).isin(asset_limits)
        ]
        if not attempts.empty:
            figure.add_trace(go.Scatter(
                x=attempts["Shortlist cap"].astype(int).map(lambda value: f"{value:,}"),
                y=attempts["Maximum assets"].astype(int).map(lambda value: f"{value:,}"),
                mode="markers", name="Evaluated, no feasible result",
                marker={"size": 8, "color": "#a8afbd", "symbol": "x"},
                customdata=attempts["Status"].astype(str),
                hovertemplate=(
                    "Shortlist cap %{x}<br>Maximum assets %{y}<br>"
                    "Status %{customdata}<extra></extra>"
                ),
            ))

    def add_decision_marker(row, name, color, symbol):
        if not row:
            return
        try:
            cap = f"{int(row['Shortlist cap']):,}"
            assets = f"{int(row['Maximum assets']):,}"
        except (KeyError, TypeError, ValueError):
            return
        figure.add_trace(go.Scatter(
            x=[cap], y=[assets], mode="markers", name=name,
            marker={"size": 15, "color": color, "symbol": symbol,
                    "line": {"color": "#f7f9ff", "width": 2}},
            hovertemplate=f"<b>{name}</b><br>%{{x}} × %{{y}}<extra></extra>",
        ))

    if best_so_far:
        try:
            best_cap = f"{int(best_so_far['Shortlist cap']):,}"
            best_assets = f"{int(best_so_far['Maximum assets']):,}"
            best_return = 100.0 * float(best_so_far["Annual Return"])
        except (KeyError, TypeError, ValueError):
            best_cap = best_assets = None
            best_return = math.nan
        if best_cap is not None and math.isfinite(best_return):
            figure.add_trace(go.Scatter(
                x=[best_cap], y=[best_assets], mode="markers+text",
                name="Highest found",
                text=[f"Best {best_cap} × {best_assets}"],
                textposition="top right",
                marker={"size": 16, "color": "#f4d35e", "symbol": "star",
                        "line": {"color": "#111827", "width": 1.5}},
                hovertemplate=(
                    f"<b>Highest return found</b><br>{best_cap} × {best_assets}"
                    f"<br>Annual return {best_return:.2f}%<extra></extra>"
                ),
            ))
    else:
        add_decision_marker(raw_peak, "Raw peak", "#f4d35e", "diamond")
        add_decision_marker(robust_choice, "Robust choice", "#ff6b9a", "square")
    if live and next_proposal:
        try:
            from_x = f"{int(next_proposal['from_x']):,}"
            from_y = f"{int(next_proposal['from_y']):,}"
            to_x = f"{int(next_proposal['x']):,}"
            to_y = f"{int(next_proposal['y']):,}"
        except (KeyError, TypeError, ValueError):
            from_x = from_y = to_x = to_y = None
        try:
            predicted = float(next_proposal.get("predicted_return"))
        except (TypeError, ValueError):
            predicted = math.nan
        if (
            from_x in cap_labels and to_x in cap_labels
            and from_y in asset_labels and to_y in asset_labels
        ):
            figure.add_trace(go.Scatter(
                x=[from_x, to_x], y=[from_y, to_y],
                mode="lines+markers+text", name="LR direction / next probe",
                text=["Current best", "Next LR probe"],
                textposition="top right",
                line={"color": "#ff6b9a", "width": 3, "dash": "dash"},
                marker={"size": [8, 14], "color": ["#ff6b9a", "#63e6ff"],
                        "symbol": ["circle", "triangle-up"]},
                customdata=[
                    "Current incumbent",
                    (
                        f"LR predicted {predicted * 100.0:.2f}%"
                        if math.isfinite(predicted) else "LR prediction unavailable"
                    ),
                ],
                hovertemplate=(
                    "LR search path<br>X %{x}<br>Y %{y}"
                    "<br>%{customdata}<extra></extra>"
                ),
            ))
    def sparse_ticks(labels, maximum=12):
        stride = max(1, math.ceil(len(labels) / maximum))
        selected = list(labels[::stride])
        if labels and selected[-1] != labels[-1]:
            selected.append(labels[-1])
        return selected

    cap_ticks = sparse_ticks(cap_labels)
    asset_ticks = sparse_ticks(asset_labels)
    figure.update_layout(
        height=520,
        margin={"l": 70, "r": 20, "t": 60, "b": 55},
        title={
            "text": "Live search map · blank cells are untested" if live else "Decision heatmap",
            "x": 0.02, "xanchor": "left",
        },
        xaxis={
            "title": "Shortlist cap", "type": "category", "automargin": True,
            "categoryorder": "array", "categoryarray": cap_labels,
            "tickmode": "array", "tickvals": cap_ticks, "ticktext": cap_ticks,
        },
        yaxis={
            "title": "Maximum assets", "type": "category", "automargin": True,
            "categoryorder": "array", "categoryarray": asset_labels,
            "tickmode": "array", "tickvals": asset_ticks, "ticktext": asset_ticks,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#090d18",
        font={"color": "#d7deeb"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "right", "x": 1.0},
        uirevision=(
            "optimizer-search-heatmap-live-v1"
            if live
            else "optimizer-search-heatmap-v1"
        ),
    )
    figure.add_annotation(
        text=(
            "Dashed arrow shows the LR-directed next probe; star is best found so far"
            if live and next_proposal and best_so_far
            else "Highest-return feasible pair found so far is marked with a star"
            if best_so_far
            else "Blank cells are untested"
        ),
        x=0.01, y=-0.13, xref="paper", yref="paper", showarrow=False,
        font={"size": 12, "color": "#aab5c8"},
    )
    return figure


def build_search_slice_figure(
    rows: Sequence[Mapping[str, object]],
    shortlist_cap: int,
) -> go.Figure:
    """Show the observed Maximum-assets cross-section for one shortlist cap."""
    frame = _finite_feasible_frame(rows)
    frame = frame.loc[
        frame["Shortlist cap"].astype(int).eq(int(shortlist_cap))
    ].sort_values("Maximum assets")
    figure = go.Figure()
    if frame.empty:
        return figure
    figure.add_trace(go.Scatter(
        x=frame["Maximum assets"].astype(int),
        y=frame["Annual Return"].astype(float) * 100.0,
        mode="lines+markers",
        name=f"Cap {int(shortlist_cap):,}",
        line={"color": "#63e6ff", "width": 3},
        marker={"size": 8, "color": "#f4d35e",
                "line": {"color": "#f7f9ff", "width": 1}},
        hovertemplate=(
            "Maximum assets %{x:,}<br>Annual return %{y:.2f}%<extra></extra>"
        ),
    ))
    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        title={"text": f"Cross-section at shortlist cap {int(shortlist_cap):,}",
               "x": 0.02, "xanchor": "left"},
        xaxis={"title": "Maximum assets", "tickformat": ",d",
               "gridcolor": "rgba(190,205,230,0.16)"},
        yaxis={"title": "Annual return (%)",
               "gridcolor": "rgba(190,205,230,0.16)"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#090d18",
        font={"color": "#d7deeb"},
        showlegend=False,
        uirevision=f"optimizer-search-slice-{int(shortlist_cap)}",
    )
    return figure
