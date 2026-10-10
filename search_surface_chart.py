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
        x=[cap], y=[assets], z=[annual_return],
        mode="markers+text",
        name=name,
        text=[name],
        textposition="top center",
        marker={
            "size": 9,
            "color": color,
            "symbol": symbol,
            "line": {"color": "#f7f9ff", "width": 1.5},
        },
        hovertemplate=(
            f"<b>{name}</b><br>Shortlist cap %{{x:,}}<br>"
            "Maximum assets %{y:,}<br>Annual return %{z:.2f}%<extra></extra>"
        ),
    )


def build_search_surface_figure(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_peak: Mapping[str, object] | None = None,
    robust_choice: Mapping[str, object] | None = None,
) -> go.Figure:
    """Build a truthful observed-grid surface; missing combinations remain holes."""
    frame = _finite_feasible_frame(rows)
    figure = go.Figure()
    if frame.empty:
        return figure

    caps = sorted(frame["Shortlist cap"].astype(int).unique())
    asset_limits = sorted(frame["Maximum assets"].astype(int).unique())
    pivot = frame.pivot_table(
        index="Maximum assets",
        columns="Shortlist cap",
        values="Annual Return",
        aggfunc="max",
    ).reindex(index=asset_limits, columns=caps)
    z_values = pivot.to_numpy(dtype=float) * 100.0

    if len(caps) >= 2 and len(asset_limits) >= 2:
        figure.add_trace(go.Surface(
            x=caps,
            y=asset_limits,
            z=z_values,
            name="Observed return surface",
            connectgaps=False,
            opacity=0.78,
            colorscale=SEARCH_COLORSCALE,
            colorbar={"title": "Return (%)", "thickness": 14, "len": 0.65},
            contours={
                "z": {
                    "show": True,
                    "usecolormap": False,
                    "color": "rgba(245,247,255,0.55)",
                    "width": 1,
                    "project_z": True,
                }
            },
            hovertemplate=(
                "Shortlist cap %{x:,}<br>Maximum assets %{y:,}<br>"
                "Annual return %{z:.2f}%<extra></extra>"
            ),
        ))

    figure.add_trace(go.Scatter3d(
        x=frame["Shortlist cap"].astype(int),
        y=frame["Maximum assets"].astype(int),
        z=frame["Annual Return"].astype(float) * 100.0,
        mode="markers",
        name="Feasible observations",
        marker={
            "size": 4.5,
            "color": "#63e6ff",
            "opacity": 0.9,
            "line": {"color": "#dffaff", "width": 0.5},
        },
        customdata=np.column_stack((
            frame.get("Trading days", pd.Series(index=frame.index, dtype=float)).fillna(0),
            frame.get("Assets", pd.Series(index=frame.index, dtype=float)).fillna(0),
        )),
        hovertemplate=(
            "Shortlist cap %{x:,}<br>Maximum assets %{y:,}<br>"
            "Annual return %{z:.2f}%<br>Trading sessions %{customdata[0]:,.0f}<br>"
            "Assets solved %{customdata[1]:,.0f}<extra></extra>"
        ),
    ))

    raw_trace = _marker_trace(
        raw_peak, name="Raw peak", color="#f4d35e", symbol="diamond"
    )
    robust_trace = _marker_trace(
        robust_choice, name="Robust choice", color="#ff6b9a", symbol="square"
    )
    if raw_trace is not None:
        figure.add_trace(raw_trace)
    if robust_trace is not None:
        figure.add_trace(robust_trace)

    figure.update_layout(
        height=650,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        title={
            "text": "Observed optimizer landscape",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 22, "color": "#f2f5fb"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#d7deeb", "family": "Arial, sans-serif"},
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.01,
            "xanchor": "right", "x": 1.0,
        },
        scene={
            "bgcolor": "#090d18",
            "xaxis": {
                "title": "Shortlist cap",
                "gridcolor": "rgba(190,205,230,0.18)",
                "zerolinecolor": "rgba(190,205,230,0.25)",
            },
            "yaxis": {
                "title": "Maximum assets",
                "gridcolor": "rgba(190,205,230,0.18)",
                "zerolinecolor": "rgba(190,205,230,0.25)",
            },
            "zaxis": {
                "title": "Annual return (%)",
                "gridcolor": "rgba(190,205,230,0.18)",
                "zerolinecolor": "rgba(190,205,230,0.25)",
            },
            "camera": {"eye": {"x": 1.45, "y": 1.55, "z": 1.05}},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.35, "y": 1.0, "z": 0.72},
        },
        hoverlabel={"bgcolor": "#111827", "font": {"color": "#f8fafc"}},
        uirevision="optimizer-search-surface-v1",
    )
    figure.add_annotation(
        text="Observed combinations only · holes are untested · drag to rotate",
        x=0.02, y=0.01, xref="paper", yref="paper",
        showarrow=False, font={"size": 12, "color": "#aab5c8"},
    )
    return figure


def build_search_heatmap_figure(
    rows: Sequence[Mapping[str, object]],
    *,
    raw_peak: Mapping[str, object] | None = None,
    robust_choice: Mapping[str, object] | None = None,
) -> go.Figure:
    """Build the decision-first top view without filling unobserved grid cells."""
    frame = _finite_feasible_frame(rows)
    figure = go.Figure()
    if frame.empty:
        return figure
    caps = sorted(frame["Shortlist cap"].astype(int).unique())
    asset_limits = sorted(frame["Maximum assets"].astype(int).unique())
    pivot = frame.pivot_table(
        index="Maximum assets", columns="Shortlist cap",
        values="Annual Return", aggfunc="max",
    ).reindex(index=asset_limits, columns=caps)
    figure.add_trace(go.Heatmap(
        x=caps,
        y=asset_limits,
        z=pivot.to_numpy(dtype=float) * 100.0,
        colorscale=SEARCH_COLORSCALE,
        colorbar={"title": "Return (%)", "thickness": 14},
        hoverongaps=False,
        xgap=2,
        ygap=2,
        hovertemplate=(
            "Shortlist cap %{x:,}<br>Maximum assets %{y:,}<br>"
            "Annual return %{z:.2f}%<extra></extra>"
        ),
    ))

    def add_decision_marker(row, name, color, symbol):
        if not row:
            return
        try:
            cap = int(row["Shortlist cap"])
            assets = int(row["Maximum assets"])
        except (KeyError, TypeError, ValueError):
            return
        figure.add_trace(go.Scatter(
            x=[cap], y=[assets], mode="markers", name=name,
            marker={"size": 15, "color": color, "symbol": symbol,
                    "line": {"color": "#f7f9ff", "width": 2}},
            hovertemplate=f"<b>{name}</b><br>%{{x:,}} × %{{y:,}}<extra></extra>",
        ))

    add_decision_marker(raw_peak, "Raw peak", "#f4d35e", "diamond")
    add_decision_marker(robust_choice, "Robust choice", "#ff6b9a", "square")
    figure.update_layout(
        height=520,
        margin={"l": 20, "r": 20, "t": 60, "b": 40},
        title={"text": "Decision heatmap", "x": 0.02, "xanchor": "left"},
        xaxis={"title": "Shortlist cap", "tickformat": ",d"},
        yaxis={"title": "Maximum assets", "tickformat": ",d"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#090d18",
        font={"color": "#d7deeb"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01,
                "xanchor": "right", "x": 1.0},
        uirevision="optimizer-search-heatmap-v1",
    )
    figure.add_annotation(
        text="Blank cells are untested",
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
