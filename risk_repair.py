"""Bounded, sell-only minimum-turnover repair. No providers or trade submission."""
from datetime import datetime, timezone
import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from allocation_integrity import execution_check
from allocation_policy_config import validate_allocation_policy
from risk_aware_allocator import build_policy_limits, joint_block_scenarios
from trade_quantities import quantity_step
from fixed_weight_rebalance import holdings_fingerprint


VERSION = "sell-only-risk-repair-v1"


def verify_owned_history_fresh(prices, holdings, now):
    """Require each listing's latest prior-day session; no weekday/holiday guesses."""
    import pandas_market_calendars as mcal
    end = pd.Timestamp(now).tz_convert("Asia/Kolkata").normalize().tz_localize(None)-pd.Timedelta(days=1)
    evidence, calendars = {}, {}
    suffixes = {".NS":"NSE",".BO":"BSE",".SA":"BMF",".L":"LSE",".TO":"TSX",".HK":"HKEX",".T":"JPX",".AX":"ASX"}
    for row in pd.DataFrame(holdings).to_dict(orient="records"):
        ticker, exchange = str(row["Yahoo Ticker"]), str(row.get("Exchange", "")).upper()
        calendar = next((c for suffix,c in suffixes.items() if ticker.endswith(suffix)),None)
        if exchange == "CCC" or ticker.endswith("-USD"): calendar = "24/7"
        if calendar is None and exchange in {"NYQ","NMS","NGM","NCM","ASE","PCX","BTS","NYSE","NASDAQ","NYSEARCA"}: calendar = "NYSE"
        if calendar is None: raise ValueError("Verified listing calendar required: " + ticker)
        if calendar not in calendars:
            days = pd.DatetimeIndex([end]) if calendar == "24/7" else mcal.get_calendar(calendar).valid_days(end-pd.Timedelta(days=31),end)
            if not len(days): raise ValueError("No verified recent session: " + calendar)
            calendars[calendar] = str(days[-1].date())
        observed = pd.to_numeric(prices[ticker],errors="coerce").dropna() if ticker in prices else pd.Series(dtype=float)
        if observed.empty or str(pd.Timestamp(observed.index[-1]).date()) < calendars[calendar]:
            raise ValueError("Stale owned history: " + ticker + "; needs session " + calendars[calendar])
        evidence[ticker] = {"expected":calendars[calendar],"observed":str(pd.Timestamp(observed.index[-1]).date())}
    return evidence


def repair_context(holdings, paused, frozen, policy, settings, date_ist):
    return {"holdings": holdings_fingerprint(holdings), "paused": sorted(set(map(str, paused))),
            "frozen": sorted(set(map(str, frozen))), "policy": dict(policy),
            "max_drawdown": settings.get("internal_max_dd"),
            "target_volatility": settings.get("target_volatility_daily"),
            "cost_bps": float(((settings.get("security_scenario_forecast") or {}).get("one_way_cost_bps", 22.))),
            "date_ist": str(date_ist)}


def prepare_risk_repair(holdings, log_returns, native_prices, fx_rates, policy, *,
                        paused=(), frozen=(), cost_bps=22., max_weight=.5,
                        max_drawdown=None, target_volatility=None, covariance=None):
    """Integer sale lots minimize gross sales under position/group/two-book CVaR.

    No full-investment equality: sale proceeds are cash with zero modeled return.
    Actual fees reduce capital; stressed fees also enter each tail-loss book.
    Every returned proposal is independently replayed. A timeout is not a pass.
    """
    started = time.perf_counter()
    report = {"version": VERSION, "status": "unavailable", "orders": [], "rows": [],
              "computed_at": datetime.now(timezone.utc).isoformat(), "optimizer_search_run": False}
    try:
        policy = validate_allocation_policy(policy)
        if not policy["enabled"]:
            raise ValueError("An enabled, explicit allocation policy is required")
        held = pd.DataFrame(holdings).copy()
        held["Quantity"] = pd.to_numeric(held["Quantity"], errors="coerce")
        if not np.isfinite(held["Quantity"]).all() or held["Quantity"].lt(0).any():
            raise ValueError("Invalid current holding quantities")
        held = held.loc[held["Quantity"].gt(0)].copy()
        held["Yahoo Ticker"] = held["Yahoo Ticker"].astype(str).str.strip().str.upper()
        names = held["Yahoo Ticker"].tolist()
        if not names or len(names) != len(set(names)):
            raise ValueError("Unique current owned holdings are required")
        history = pd.DataFrame(log_returns).reindex(columns=names).copy()
        # Per-listing gaps may be holidays; use only complete common observations.
        history = history.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        held["FX to INR"] = pd.to_numeric(held["Currency"].map(fx_rates), errors="coerce")
        prices = pd.to_numeric(held["Yahoo Ticker"].map(native_prices), errors="coerce").to_numpy()
        marks = prices * held["FX to INR"].to_numpy()
        if not np.isfinite(marks).all() or (marks <= 0).any():
            raise ValueError("Verified positive prices and FX are required for every owned holding")
        quantities = held["Quantity"].to_numpy(dtype=float)
        values = marks * quantities
        base = float(values.sum())
        if not math.isfinite(base) or base <= 0 or not 0 < float(max_weight) <= 1:
            raise ValueError("Invalid portfolio base or maximum weight")
        initial = values/base
        held["Weight"] = initial
        samples, limits = build_policy_limits(history, initial, [(0.,max_weight)]*len(names), policy, cost_bps)
        report.update(history_through=str(history.index[-1])[:10], policy=policy, limits=limits,
                      marked_base_inr=base, quotes_checked_at=report["computed_at"],
                      tax_coverage="Tax lots and capital-gains taxes not modeled", cash_return_assumption=0.)
        report["history_hash"] = hashlib.sha256(pd.util.hash_pandas_object(history,index=True).values.tobytes()
                                               + json.dumps(names).encode()).hexdigest()
        report["quoted_inputs"] = [{"ticker":t,"quantity":float(quantities[i]),"native_price":float(prices[i]),
                                    "currency":str(held.iloc[i]["Currency"]),"fx_to_inr":float(held.iloc[i]["FX to INR"])}
                                   for i,t in enumerate(names)]
        blocked = {str(t).strip().upper() for t in [*paused, *frozen]}
        frozen_map = {t: float(initial[i]) for i,t in enumerate(names) if t in blocked}
        kwargs = dict(cost_bps=cost_bps, max_weight=max_weight, max_drawdown=max_drawdown,
            target_volatility=target_volatility,
            covariance=(history.cov().to_numpy() if covariance is None else covariance) if target_volatility is not None else None,
            excluded_tickers=paused, frozen_weights=frozen_map, policy_limits=limits)
        before = execution_check(held, [], native_prices, history, **kwargs)
        report["before"] = before
        report["rows"] = [{"Security":t,"Current allocation (%)":float(100*initial[i]),
            "After repair (%)":None,"Cap (%)":float(100*limits["position_caps"][t]),
            "Sell quantity":None,"Sale value (INR)":None,
            "Restriction":"Paused/frozen" if t in blocked else ""} for i,t in enumerate(names)]
        if before["passed"]:
            report.update(status="no_repair_needed", validation=before)
            return report
        report["blocked_tickers"] = sorted(blocked.intersection(names))
        steps = np.array([quantity_step(q,c) for q,c in zip(quantities,held["Currency"])])
        lot_values = marks*steps/base
        # Never assume a fractional INR remainder is tradable as a whole lot.
        upper = np.floor(quantities/steps + 1e-8)
        upper[[i for i,t in enumerate(names) if t in blocked]] = 0.
        n, m = len(names), len(samples)
        books = [samples]
        if policy["adaptive_enabled"]:
            books.append(joint_block_scenarios(history, policy))
        k = len(books)
        extra = k*(m+1)
        rate = float(cost_bps)/10000
        stressed_rate = limits["cost_rate"]
        matrices, rhs = [], []
        caps = np.array([limits["position_caps"][t] for t in names])
        position = -np.diag(lot_values) + np.outer(caps*rate, lot_values)
        matrices.append(sparse.hstack([sparse.csr_matrix(position), sparse.csr_matrix((n,extra))]))
        rhs.append(caps-initial)
        for group in limits["groups"]:
            member = np.array([t in group["tickers"] for t in names], dtype=float)
            row = -member*lot_values + group["cap"]*rate*lot_values
            matrices.append(sparse.csr_matrix(np.r_[row,np.zeros(extra)].reshape(1,-1)))
            rhs.append(np.array([group["cap"]-member@initial]))
        for b, book in enumerate(books):
            var = sparse.csr_matrix((-np.ones(m),(np.arange(m),np.full(m,b))), shape=(m,k))
            slack = sparse.hstack([sparse.csr_matrix((m,b*m)), -sparse.eye(m), sparse.csr_matrix((m,(k-b-1)*m))])
            matrices.append(sparse.hstack([sparse.csr_matrix((book+stressed_rate)*lot_values),var,slack]))
            rhs.append(book@initial)
            row = np.zeros(n+extra)
            row[n+b] = 1.
            row[n+k+b*m:n+k+(b+1)*m] = 1/((1-policy["confidence"])*m)
            matrices.append(sparse.csr_matrix(row.reshape(1,-1)))
            rhs.append(np.array([limits["tail_limit"]]))
        solution = milp(np.r_[lot_values,np.zeros(extra)],
            integrality=np.r_[np.ones(n),np.zeros(extra)],
            bounds=Bounds(np.r_[np.zeros(n),np.full(k,-np.inf),np.zeros(k*m)],
                          np.r_[upper,np.full(extra,np.inf)]),
            constraints=LinearConstraint(sparse.vstack(matrices,format="csc"), -np.inf, np.concatenate(rhs)),
            options={"time_limit": policy["solver_time_limit_seconds"], "mip_rel_gap":0.})
        report.update(solver_status=int(solution.status), solver_message=str(solution.message))
        if not solution.success:
            report.update(status="time_limit" if solution.status == 1 else "blocked" if solution.status == 2 else "unavailable",
                reason="No complete sell-only repair established within the current limits and tradable/frozen quantities.")
            return report
        units = np.rint(solution.x[:n])
        if np.max(np.abs(units-solution.x[:n])) > 1e-5:
            raise ValueError("Solver did not establish integer sale quantities")
        orders = []
        for i,ticker in enumerate(names):
            q = float(round(units[i]*steps[i],8))
            if q <= 0:
                continue
            orders.append({"Symbol":str(held.iloc[i].get("Symbol",ticker)), "Yahoo Ticker":ticker,
                "Currency":str(held.iloc[i]["Currency"]), "Action":"Sell", "Native Price":float(prices[i]),
                "FX to INR":float(held.iloc[i]["FX to INR"]), "Executable Quantity":q,
                "Executable Value":float(q*marks[i])})
        checked = execution_check(held, orders, native_prices, history, **kwargs)
        after = checked.get("final_weights") or {}
        sales = {r["Yahoo Ticker"]:r for r in orders}
        report.update(orders=orders, validation=checked,
            status="checked_draft" if checked["passed"] else "withheld",
            objective="Minimum gross sales in recorded tradable lots under position/group/joint-tail constraints; historical drawdown/volatility checked afterwards",
            cash_after_costs_inr=checked.get("cash_after_costs_inr"), estimated_cost_inr=checked.get("estimated_cost_inr"),
            gross_sales_inr=sum(r["Executable Value"] for r in orders),
            rows=[{"Security":t,"Current allocation (%)":100*initial[i],
                "After repair (%)":100*after.get(t,0.),"Cap (%)":100*caps[i],
                "Sell quantity":sales.get(t,{}).get("Executable Quantity",0.),
                "Sale value (INR)":sales.get(t,{}).get("Executable Value",0.),
                "Restriction":"Paused/frozen" if t in blocked else ""} for i,t in enumerate(names)])
        if not checked["passed"]:
            report["reason"] = "Minimum-turnover proposal failed independent final checks; not released."
        return report
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        report["reason"] = str(exc)
        return report
    finally:
        report["elapsed_seconds"] = round(time.perf_counter()-started,3)


def _schema(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS risk_repair_workspace (owner TEXT PRIMARY KEY, report_json TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS risk_repair_archive (owner TEXT, computed_at TEXT, report_json TEXT NOT NULL, PRIMARY KEY(owner,computed_at))")


def save_repair(conn, owner, report, context):
    _schema(conn)
    snapshot = dict(report, context=context)
    snapshot["evidence_hash"] = hashlib.sha256(json.dumps(snapshot,sort_keys=True,allow_nan=False).encode()).hexdigest()
    raw = json.dumps(snapshot,allow_nan=False)
    prior = conn.execute("SELECT report_json FROM risk_repair_workspace WHERE owner=?",(str(owner),)).fetchone()
    if prior and datetime.fromisoformat(json.loads(prior[0])["computed_at"]) > datetime.fromisoformat(snapshot["computed_at"]):
        raise ValueError("A newer risk-reduction attempt already exists")
    changed = conn.execute("INSERT INTO risk_repair_workspace VALUES(?,?) ON CONFLICT(owner) DO UPDATE SET report_json=excluded.report_json WHERE report_json=?",
                 (str(owner),raw,prior[0] if prior else None))
    if changed.rowcount != 1:
        conn.rollback()
        raise ValueError("Another risk-reduction attempt finished concurrently")
    conn.execute("INSERT OR IGNORE INTO risk_repair_archive VALUES(?,?,?)",(str(owner),snapshot["computed_at"],raw))
    conn.commit()


def load_repair(conn, owner):
    _schema(conn)
    row = conn.execute("SELECT report_json FROM risk_repair_workspace WHERE owner=?",(str(owner),)).fetchone()
    return json.loads(row[0]) if row else None
