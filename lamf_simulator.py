"""Indian retail-investor Loan Against Mutual Funds (LAMF) simulator.

This Streamlit app compares two ways to fund a genuine cash requirement:
A) Redeem mutual-fund units and pay applicable capital-gains tax.
B) Borrow against eligible mutual-fund units and keep the portfolio invested.

The model intentionally does NOT assume that borrowed money is reinvested in equity.
A separate disciplined-leverage mode evaluates borrowing against mutual funds to invest in stocks,
with explicit return haircuts, break-even return, cash-flow gates and drawdown stress tests.

All lender terms and tax inputs remain editable because eligibility, LTV, charges,
prepayment rules and tax treatment depend on the scheme, lender, acquisition date,
and the investor's own tax situation.
"""

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from matplotlib.ticker import FuncFormatter


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LAMF Simulator – India",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🇮🇳 Loan Against Mutual Funds (LAMF) Simulator")
st.caption("Build: 2026-08-09 LEVERAGE • yfinance Nifty 200-DMA • disciplined stock-leverage analysis")
st.caption(
    "For Indian retail investors: compare **selling mutual funds vs taking LAMF**, "
    "including tax, cash-flow strain, LTV/margin-call risk, repayment style, renewals and sensitivity."
)

st.info(
    "This is a decision-support model, not a loan recommendation. Enter the exact terms from your "
    "lender's sanction letter and your own tax position. Borrowing against market-linked assets can "
    "trigger margin calls or forced redemption when markets fall."
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def inr(value: float) -> str:
    """Indian-style compact currency formatter."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    sign = "-" if value < 0 else ""
    x = abs(float(value))
    if x >= 1_00_00_000:
        return f"{sign}₹{x / 1_00_00_000:.2f} Cr"
    if x >= 1_00_000:
        return f"{sign}₹{x / 1_00_000:.2f} L"
    return f"{sign}₹{x:,.0f}"


def monthly_rate(annual_percent: float) -> float:
    return (1 + annual_percent / 100.0) ** (1 / 12) - 1


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_nifty_200dma() -> dict[str, object]:
    """Fetch Nifty 50 daily closes and calculate its latest 200-session SMA.

    Yahoo Finance symbol ``^NSEI`` represents the Nifty 50 index. The result is
    cached for one hour to avoid repeated network calls on every Streamlit rerun.
    """
    history = yf.Ticker("^NSEI").history(
        period="2y",
        interval="1d",
        auto_adjust=False,
        actions=False,
        timeout=15,
    )
    if history.empty or "Close" not in history.columns:
        raise ValueError("Yahoo Finance returned no Nifty 50 closing-price data.")

    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(closes) < 200:
        raise ValueError(f"Only {len(closes)} valid sessions were returned; 200 are required.")

    latest_close = float(closes.iloc[-1])
    moving_average_200 = float(closes.tail(200).mean())
    if moving_average_200 <= 0:
        raise ValueError("The calculated Nifty 50 200-DMA is invalid.")

    deviation_pct = (latest_close / moving_average_200 - 1.0) * 100.0
    latest_index = closes.index[-1]
    as_of_date = latest_index.date() if hasattr(latest_index, "date") else latest_index

    return {
        "latest_close": latest_close,
        "moving_average_200": moving_average_200,
        "deviation_pct": deviation_pct,
        "as_of_date": as_of_date,
        "sessions": int(len(closes)),
    }


def fv_lump_sum(pv: float, annual_percent: float, months: int) -> float:
    return pv * (1 + monthly_rate(annual_percent)) ** months


def fv_monthly_contribution(payment: float, annual_percent: float, months: int) -> float:
    """Future value of end-of-month contributions."""
    r = monthly_rate(annual_percent)
    if months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return payment * months
    return payment * (((1 + r) ** months - 1) / r)


def estimate_embedded_gain_pct(annual_return_pct: float, holding_months: int) -> float:
    """Estimate the gain share of current value from tenure + annualised return.

    If an investment starts at cost C and compounds to value V, then
    embedded gain share = (V - C) / V = 1 - 1 / growth_factor.
    Negative/zero estimated returns are treated as 0% embedded capital gain
    for the redemption-tax approximation.
    """
    months = max(0, int(holding_months))
    if months == 0 or annual_return_pct <= 0:
        return 0.0
    growth_factor = (1 + annual_return_pct / 100.0) ** (months / 12.0)
    if growth_factor <= 0:
        return 0.0
    gain_pct = (1.0 - 1.0 / growth_factor) * 100.0
    return max(0.0, min(99.99, gain_pct))


def amortizing_emi(principal: float, annual_percent: float, months: int) -> float:
    r = annual_percent / 1200.0
    if months <= 0:
        return 0.0
    if abs(r) < 1e-12:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def amortization_schedule(principal: float, annual_percent: float, months: int) -> pd.DataFrame:
    """Standard reducing-balance amortization used as an EMI-style manual prepayment proxy."""
    emi = amortizing_emi(principal, annual_percent, months)
    r = annual_percent / 1200.0
    balance = principal
    rows = []
    for m in range(1, months + 1):
        interest = balance * r
        principal_paid = min(balance, max(0.0, emi - interest))
        payment = interest + principal_paid
        balance = max(0.0, balance - principal_paid)
        rows.append((m, payment, interest, principal_paid, balance))
    return pd.DataFrame(rows, columns=["Month", "Payment", "Interest", "Principal", "Balance"])


def required_sinking_fund_payment(target: float, annual_percent: float, months: int) -> float:
    """Monthly contribution required to accumulate target by the end of the horizon."""
    r = monthly_rate(annual_percent)
    if months <= 0:
        return target
    if abs(r) < 1e-12:
        return target / months
    factor = ((1 + r) ** months - 1) / r
    return target / factor


def tax_on_redemption(
    gross_redemption: float,
    embedded_gain_pct: float,
    tax_rate_pct: float,
    exemption_remaining: float,
    cess_pct: float,
    surcharge_pct: float,
) -> float:
    gain = gross_redemption * embedded_gain_pct / 100.0
    taxable_gain = max(0.0, gain - exemption_remaining)
    base_tax = taxable_gain * tax_rate_pct / 100.0
    surcharge = base_tax * surcharge_pct / 100.0
    tax_plus_surcharge = base_tax + surcharge
    cess = tax_plus_surcharge * cess_pct / 100.0
    return tax_plus_surcharge + cess


def gross_redemption_for_net_cash(
    net_cash_needed: float,
    embedded_gain_pct: float,
    tax_rate_pct: float,
    exemption_remaining: float,
    cess_pct: float,
    surcharge_pct: float,
) -> tuple[float, float]:
    """Solve gross redemption such that gross - tax ~= requested net cash."""
    if net_cash_needed <= 0:
        return 0.0, 0.0

    lo = net_cash_needed
    hi = net_cash_needed * 2.0 + 1_00_000

    # Expand upper bound if an extreme custom tax rate is entered.
    for _ in range(20):
        tax_hi = tax_on_redemption(
            hi, embedded_gain_pct, tax_rate_pct, exemption_remaining, cess_pct, surcharge_pct
        )
        if hi - tax_hi >= net_cash_needed:
            break
        hi *= 1.5

    for _ in range(100):
        mid = (lo + hi) / 2.0
        tax_mid = tax_on_redemption(
            mid, embedded_gain_pct, tax_rate_pct, exemption_remaining, cess_pct, surcharge_pct
        )
        net_mid = mid - tax_mid
        if net_mid < net_cash_needed:
            lo = mid
        else:
            hi = mid

    gross = hi
    tax = tax_on_redemption(
        gross, embedded_gain_pct, tax_rate_pct, exemption_remaining, cess_pct, surcharge_pct
    )
    return gross, tax


def flat_rate_loan_irr(flat_rate_pct: float, principal: float, months: int) -> float:
    """Approximate annual effective rate for a flat-rate installment loan via bisection."""
    if principal <= 0 or months <= 0:
        return 0.0
    years = months / 12.0
    total_interest = principal * flat_rate_pct / 100.0 * years
    payment = (principal + total_interest) / months

    def npv(monthly_r: float) -> float:
        return principal - sum(payment / ((1 + monthly_r) ** m) for m in range(1, months + 1))

    lo, hi = 0.0, 0.10  # up to 10% per month, intentionally wide
    for _ in range(100):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            hi = mid
        else:
            lo = mid
    monthly_irr = (lo + hi) / 2
    return ((1 + monthly_irr) ** 12 - 1) * 100


@dataclass
class ScenarioResult:
    sell_end_wealth: float
    loan_end_wealth: float
    advantage: float
    gross_redemption: float
    redemption_tax: float
    loan_monthly_debt_service: float
    residual_monthly_investment: float
    cashflow_shortfall: float
    total_interest: float
    total_fees: float
    ending_loan_balance: float
    sinking_fund_value: float
    bullet_repayment_tax: float
    renewal_count: int


def run_scenario(
    *,
    cash_required: float,
    portfolio_value: float,
    embedded_gain_pct: float,
    tax_rate_pct: float,
    exemption_remaining: float,
    cess_pct: float,
    surcharge_pct: float,
    expected_return_pct: float,
    monthly_surplus: float,
    loan_rate_pct: float,
    horizon_months: int,
    sanction_months: int,
    processing_fee_pre_gst: float,
    processing_gst_pct: float,
    other_upfront_charges: float,
    prepayment_charge_pct: float,
    repayment_style: str,
    sinking_return_pct: float,
    bullet_repayment_from_mf: bool,
    repayment_embedded_gain_pct: float,
    repayment_exemption_available: float,
) -> ScenarioResult:
    # Path A: sell enough units to receive the same usable cash after tax.
    gross_redemption, redemption_tax = gross_redemption_for_net_cash(
        cash_required,
        embedded_gain_pct,
        tax_rate_pct,
        exemption_remaining,
        cess_pct,
        surcharge_pct,
    )
    remaining_portfolio = max(0.0, portfolio_value - gross_redemption)
    sell_end_wealth = fv_lump_sum(remaining_portfolio, expected_return_pct, horizon_months)
    sell_end_wealth += fv_monthly_contribution(monthly_surplus, expected_return_pct, horizon_months)

    # Loan fees repeat on every sanction/renewal cycle. Example: 12y planned / 3y sanction = 4 fee events.
    fee_events = max(1, math.ceil(horizon_months / max(1, sanction_months)))
    renewal_count = max(0, fee_events - 1)
    processing_with_gst = processing_fee_pre_gst * (1 + processing_gst_pct / 100.0)
    total_fees = fee_events * (processing_with_gst + other_upfront_charges)

    principal = cash_required
    ending_balance = principal
    sinking_fund_value = 0.0
    bullet_repayment_tax = 0.0
    bullet_gross_redemption = principal

    if repayment_style == "EMI-style manual principal prepayment":
        schedule = amortization_schedule(principal, loan_rate_pct, horizon_months)
        monthly_debt_service = float(schedule["Payment"].iloc[0]) if not schedule.empty else 0.0
        total_interest = float(schedule["Interest"].sum())
        ending_balance = float(schedule["Balance"].iloc[-1]) if not schedule.empty else 0.0
        # Optional foreclosure/prepayment cost; user can set zero if lender waives it.
        prepayment_charge = principal * prepayment_charge_pct / 100.0
        total_fees += prepayment_charge

    elif repayment_style == "Interest-only + sinking fund":
        monthly_interest = principal * loan_rate_pct / 1200.0
        sinking_payment = required_sinking_fund_payment(principal, sinking_return_pct, horizon_months)
        monthly_debt_service = monthly_interest + sinking_payment
        total_interest = monthly_interest * horizon_months
        sinking_fund_value = fv_monthly_contribution(sinking_payment, sinking_return_pct, horizon_months)
        ending_balance = principal

    else:  # Interest-only, bullet principal at end
        monthly_interest = principal * loan_rate_pct / 1200.0
        monthly_debt_service = monthly_interest
        total_interest = monthly_interest * horizon_months
        ending_balance = principal
        if bullet_repayment_from_mf:
            bullet_gross_redemption, bullet_repayment_tax = gross_redemption_for_net_cash(
                principal,
                repayment_embedded_gain_pct,
                tax_rate_pct,
                repayment_exemption_available,
                cess_pct,
                surcharge_pct,
            )

    residual_monthly_investment = max(0.0, monthly_surplus - monthly_debt_service)
    monthly_shortfall = max(0.0, monthly_debt_service - monthly_surplus)
    cashflow_shortfall = monthly_shortfall * horizon_months

    # Path B: keep portfolio intact. Any monthly surplus left after debt service continues into investments.
    loan_end_wealth = fv_lump_sum(portfolio_value, expected_return_pct, horizon_months)
    loan_end_wealth += fv_monthly_contribution(
        residual_monthly_investment, expected_return_pct, horizon_months
    )
    loan_end_wealth += sinking_fund_value
    if repayment_style == "Interest-only, bullet principal at end" and bullet_repayment_from_mf:
        # Repaying from the portfolio may require redeeming more than the principal because tax can be triggered.
        loan_end_wealth -= bullet_gross_redemption
    else:
        loan_end_wealth -= ending_balance
    loan_end_wealth -= total_fees

    # If the stated monthly surplus cannot cover debt service, make the comparison conservative by
    # subtracting the additional outside cash that would have to be brought into the loan path.
    loan_end_wealth -= cashflow_shortfall

    return ScenarioResult(
        sell_end_wealth=sell_end_wealth,
        loan_end_wealth=loan_end_wealth,
        advantage=loan_end_wealth - sell_end_wealth,
        gross_redemption=gross_redemption,
        redemption_tax=redemption_tax,
        loan_monthly_debt_service=monthly_debt_service,
        residual_monthly_investment=residual_monthly_investment,
        cashflow_shortfall=cashflow_shortfall,
        total_interest=total_interest,
        total_fees=total_fees,
        ending_loan_balance=ending_balance,
        sinking_fund_value=sinking_fund_value,
        bullet_repayment_tax=bullet_repayment_tax,
        renewal_count=renewal_count,
    )


@dataclass
class LeverageResult:
    no_leverage_end_wealth: float
    leverage_end_wealth: float
    advantage: float
    borrowed_sleeve_value: float
    monthly_debt_service: float
    residual_monthly_investment: float
    cashflow_shortfall: float
    total_interest: float
    total_fees: float
    ending_loan_balance: float
    sinking_fund_value: float
    exit_tax: float
    exit_gross_redemption: float
    repayment_shortfall: float


def run_leverage_scenario(
    *,
    borrowed_amount: float,
    existing_stock_value: float,
    stock_return_pct: float,
    monthly_surplus: float,
    loan_rate_pct: float,
    horizon_months: int,
    sanction_months: int,
    processing_fee_pre_gst: float,
    processing_gst_pct: float,
    other_upfront_charges: float,
    prepayment_charge_pct: float,
    repayment_style: str,
    sinking_return_pct: float,
    repay_bullet_from_stock_sleeve: bool,
    exit_tax_rate_pct: float,
    exit_exemption: float,
    exit_cess_pct: float,
    exit_surcharge_pct: float,
) -> LeverageResult:
    """Compare a stock portfolio with and without an incremental LAMF-funded stock sleeve.

    The same monthly investable surplus is used in both paths. In the leverage path,
    mandatory debt service consumes part of that surplus, so the model does not assume
    that salary/investing capacity magically increases after borrowing.
    """
    months = max(1, int(horizon_months))
    sanction = max(1, int(sanction_months))
    principal = max(0.0, float(borrowed_amount))

    fee_events = max(1, math.ceil(months / sanction))
    processing_with_gst = processing_fee_pre_gst * (1 + processing_gst_pct / 100.0)
    total_fees = fee_events * (processing_with_gst + other_upfront_charges)

    ending_balance = principal
    sinking_fund_value = 0.0
    exit_tax = 0.0
    exit_gross_redemption = principal
    repayment_shortfall = 0.0

    if repayment_style == "EMI-style manual principal prepayment":
        schedule = amortization_schedule(principal, loan_rate_pct, months)
        monthly_debt_service = float(schedule["Payment"].iloc[0]) if not schedule.empty else 0.0
        total_interest = float(schedule["Interest"].sum()) if not schedule.empty else 0.0
        ending_balance = float(schedule["Balance"].iloc[-1]) if not schedule.empty else 0.0
        total_fees += principal * prepayment_charge_pct / 100.0
    elif repayment_style == "Interest-only + sinking fund":
        monthly_interest = principal * loan_rate_pct / 1200.0
        sinking_payment = required_sinking_fund_payment(principal, sinking_return_pct, months)
        monthly_debt_service = monthly_interest + sinking_payment
        total_interest = monthly_interest * months
        sinking_fund_value = fv_monthly_contribution(sinking_payment, sinking_return_pct, months)
    else:
        monthly_interest = principal * loan_rate_pct / 1200.0
        monthly_debt_service = monthly_interest
        total_interest = monthly_interest * months

    residual_monthly_investment = max(0.0, monthly_surplus - monthly_debt_service)
    monthly_shortfall = max(0.0, monthly_debt_service - monthly_surplus)
    cashflow_shortfall = monthly_shortfall * months

    no_leverage_end_wealth = fv_lump_sum(existing_stock_value, stock_return_pct, months)
    no_leverage_end_wealth += fv_monthly_contribution(monthly_surplus, stock_return_pct, months)

    borrowed_sleeve_value = fv_lump_sum(principal, stock_return_pct, months)
    leverage_end_wealth = fv_lump_sum(existing_stock_value, stock_return_pct, months)
    leverage_end_wealth += borrowed_sleeve_value
    leverage_end_wealth += fv_monthly_contribution(
        residual_monthly_investment, stock_return_pct, months
    )
    leverage_end_wealth += sinking_fund_value

    if repayment_style == "Interest-only, bullet principal at end" and repay_bullet_from_stock_sleeve:
        gain_share_pct = 0.0
        if borrowed_sleeve_value > 0 and borrowed_sleeve_value > principal:
            gain_share_pct = (borrowed_sleeve_value - principal) / borrowed_sleeve_value * 100.0
        exit_gross_redemption, exit_tax = gross_redemption_for_net_cash(
            principal,
            gain_share_pct,
            exit_tax_rate_pct,
            exit_exemption,
            exit_cess_pct,
            exit_surcharge_pct,
        )
        leverage_end_wealth -= exit_gross_redemption
        repayment_shortfall = max(0.0, exit_gross_redemption - borrowed_sleeve_value)
    else:
        leverage_end_wealth -= ending_balance

    leverage_end_wealth -= total_fees
    leverage_end_wealth -= cashflow_shortfall

    return LeverageResult(
        no_leverage_end_wealth=no_leverage_end_wealth,
        leverage_end_wealth=leverage_end_wealth,
        advantage=leverage_end_wealth - no_leverage_end_wealth,
        borrowed_sleeve_value=borrowed_sleeve_value,
        monthly_debt_service=monthly_debt_service,
        residual_monthly_investment=residual_monthly_investment,
        cashflow_shortfall=cashflow_shortfall,
        total_interest=total_interest,
        total_fees=total_fees,
        ending_loan_balance=ending_balance,
        sinking_fund_value=sinking_fund_value,
        exit_tax=exit_tax,
        exit_gross_redemption=exit_gross_redemption,
        repayment_shortfall=repayment_shortfall,
    )


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1) Cash requirement")
    use_case = st.selectbox(
        "Purpose",
        [
            "Emergency / medical / family need",
            "Used vehicle / large purchase",
            "Business working capital",
            "Debt consolidation",
            "Home down payment / token",
            "Other genuine expense",
            "Disciplined leverage into stocks",
        ],
    )
    cash_input_label = (
        "Planned LAMF draw for stock leverage (₹)"
        if use_case == "Disciplined leverage into stocks"
        else "Net cash required (₹)"
    )
    cash_input_default = 1_50_000 if use_case == "Disciplined leverage into stocks" else 5_00_000
    cash_required = st.number_input(
        cash_input_label,
        min_value=25_000,
        max_value=5_00_00_000,
        value=cash_input_default,
        step=25_000,
        help=(
            "This is the amount actually drawn and invested. Interest is charged on the draw, not merely the sanctioned limit."
            if use_case == "Disciplined leverage into stocks"
            else None
        ),
    )
    monthly_surplus = st.number_input(
        "Monthly amount available for SIP / debt service (₹)",
        min_value=0,
        max_value=10_00_000,
        value=25_000,
        step=1_000,
        help="Use the amount you can sustainably allocate each month. The comparison keeps this budget equal across both paths.",
    )

    # Stock-leverage inputs are deliberately separate from the MF collateral assumptions.
    existing_stock_value = 0.0
    historical_stock_xirr_pct = 0.0
    forward_return_haircut_pct = 0.0
    forward_stock_return_pct = 0.0
    personal_max_leverage_pct = 100.0
    max_debt_service_share_pct = 100.0
    min_break_even_buffer_pct = 0.0
    leverage_exit_tax_rate_pct = 0.0
    leverage_exit_exemption = 0.0

    if use_case == "Disciplined leverage into stocks":
        st.header("1A) Stock leverage discipline")
        existing_stock_value = st.number_input(
            "Current stock portfolio value (₹)",
            min_value=0,
            max_value=50_00_00_000,
            value=3_70_000,
            step=10_000,
            help="Existing unlevered stock capital. Default reflects the portfolio figure you provided and is editable.",
        )
        historical_stock_xirr_pct = st.number_input(
            "Historical stock XIRR (%)",
            min_value=-100.0,
            max_value=100.0,
            value=18.0,
            step=0.5,
            help="Backward-looking reference only. It is not assumed to repeat automatically.",
        )
        forward_return_haircut_pct = st.number_input(
            "Haircut to historical XIRR for forward planning (%)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
            help="Example: 18% historical XIRR with a 25% haircut gives a 13.5% forward planning return.",
        )
        forward_stock_return_pct = historical_stock_xirr_pct * (1.0 - forward_return_haircut_pct / 100.0)
        st.metric("Forward planning stock return", f"{forward_stock_return_pct:.2f}% p.a.")

        personal_max_leverage_pct = st.number_input(
            "Personal max borrowed amount (% of existing stock portfolio)",
            min_value=0.0,
            max_value=300.0,
            value=50.0,
            step=5.0,
            help="A personal position-sizing rule, separate from the lender's LTV and the Nifty 200-DMA sanction rule.",
        )
        max_debt_service_share_pct = st.number_input(
            "Max debt service (% of monthly investable surplus)",
            min_value=1.0,
            max_value=100.0,
            value=50.0,
            step=5.0,
            help="Keeps mandatory loan payments from consuming your entire monthly investing capacity.",
        )
        min_break_even_buffer_pct = st.number_input(
            "Minimum forward-return buffer above break-even (percentage points)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5,
            help="A margin of safety: forward expected return should exceed the modeled break-even return by at least this amount.",
        )
        with st.expander("Optional exit-tax assumption for leveraged stock sleeve"):
            leverage_exit_tax_rate_pct = st.number_input(
                "Tax rate on gains when leveraged stock sleeve is sold (%)",
                min_value=0.0,
                max_value=50.0,
                value=12.5,
                step=0.5,
                help="Editable. Used only when a bullet principal is assumed to be repaid by selling the leveraged stock sleeve.",
            )
            leverage_exit_exemption = st.number_input(
                "Gain exemption available at leverage exit (₹)",
                min_value=0,
                max_value=10_00_000,
                value=1_25_000,
                step=5_000,
            )

    st.header("2) Mutual-fund portfolio")
    portfolio_value = st.number_input(
        "Current MF portfolio value (₹)",
        min_value=50_000,
        max_value=50_00_00_000,
        value=20_00_000,
        step=50_000,
    )
    eligible_collateral_value = st.number_input(
        "LAMF-eligible / pledgeable value (₹)",
        min_value=0,
        max_value=50_00_00_000,
        value=min(20_00_000, portfolio_value),
        step=50_000,
        help="Only enter units actually eligible with your lender; not every scheme/unit may be accepted.",
    )
    expected_return_pct = st.number_input(
        "Expected portfolio return (p.a. %)",
        min_value=-20.0,
        max_value=30.0,
        value=10.0,
        step=0.5,
        help="Used for portfolio projection and to estimate embedded gains from the holding tenure.",
    )
    holding_period_months = st.number_input(
        "Holding period of units likely to be sold (months)",
        min_value=1,
        max_value=600,
        value=60,
        step=1,
        help="Approximate weighted holding period of units you would redeem. Use tax-lot data when available.",
    )
    embedded_gain_pct = estimate_embedded_gain_pct(
        float(expected_return_pct), int(holding_period_months)
    )
    st.metric(
        "Estimated unrealised gain embedded in units sold",
        f"{embedded_gain_pct:.2f}%",
    )
    st.caption(
        "Auto-estimated from holding tenure and expected annual return: "
        "gain share = 1 - 1 / (1 + return)^(tenure in years)."
    )

    st.header("3) Tax on redemption")
    tax_options = [
        "Equity-oriented LTCG (editable defaults)",
        "Equity-oriented STCG (editable defaults)",
        "Custom / other mutual-fund tax treatment",
    ]
    default_tax_index = 1 if int(holding_period_months) <= 12 else 0
    tax_bucket = st.selectbox(
        "Tax bucket for the units that would be sold",
        tax_options,
        index=default_tax_index,
        help="Default follows the entered holding period for equity-oriented funds; override when your fund/tax lot differs.",
    )

    if tax_bucket.startswith("Equity-oriented LTCG"):
        default_tax_rate = 12.5
        default_exemption = 1_25_000
    elif tax_bucket.startswith("Equity-oriented STCG"):
        default_tax_rate = 20.0
        default_exemption = 0
    else:
        default_tax_rate = 12.5
        default_exemption = 0

    tax_rate_pct = st.number_input(
        "Capital-gains tax rate (%)",
        min_value=0.0,
        max_value=50.0,
        value=float(default_tax_rate),
        step=0.1,
    )
    annual_exemption = st.number_input(
        "Applicable annual exemption (₹)",
        min_value=0,
        max_value=10_00_000,
        value=int(default_exemption),
        step=5_000,
    )
    exemption_used = st.number_input(
        "Exemption already used this FY (₹)",
        min_value=0,
        max_value=10_00_000,
        value=0,
        step=5_000,
    )
    exemption_remaining = max(0.0, annual_exemption - exemption_used)
    cess_pct = st.number_input("Health & education cess (%)", 0.0, 10.0, 4.0, 0.5)
    surcharge_pct = st.number_input("Surcharge on this tax, if applicable (%)", 0.0, 50.0, 0.0, 1.0)

    st.header("4) LAMF terms")
    loan_rate_pct = st.number_input(
        "LAMF interest rate (p.a. %)", 4.0, 24.0, 10.5, 0.1
    )
    horizon_months = st.number_input(
        "How long you expect to need the loan (months)", 1, 180, 36, 1
    )
    sanction_months = st.number_input(
        "Lender sanction / renewal cycle (months)", 1, 120, 36, 1,
        help="If your planned horizon is longer, the simulator repeats fees at each renewal cycle.",
    )
    lender_ltv_pct = st.number_input(
        "Lender maximum LTV (%)",
        min_value=5.0,
        max_value=95.0,
        value=50.0,
        step=1.0,
        help="Enter the exact LTV from the lender for the funds you are pledging.",
    )
    st.markdown("**Nifty 50 versus 200-DMA — automatic**")
    if st.button("↻ Refresh Nifty data", width="stretch"):
        fetch_nifty_200dma.clear()
        st.rerun()

    nifty_data_error = None
    try:
        nifty_data = fetch_nifty_200dma()
        nifty_200dma_deviation_pct = float(nifty_data["deviation_pct"])
        st.metric(
            "Nifty 50 deviation from 200-DMA",
            f"{nifty_200dma_deviation_pct:+.2f}%",
        )
        st.caption(
            f"Yahoo Finance (^NSEI), as of {nifty_data['as_of_date']}: "
            f"close **{nifty_data['latest_close']:,.2f}**, "
            f"200-session SMA **{nifty_data['moving_average_200']:,.2f}**. "
            "Data is cached for one hour."
        )
    except Exception as exc:
        nifty_data_error = str(exc)
        st.warning(
            "Live Nifty data could not be loaded from Yahoo Finance. "
            "A manual fallback is shown so the simulator remains usable."
        )
        nifty_200dma_deviation_pct = st.number_input(
            "Manual Nifty 50 deviation from 200-DMA (%)",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=0.1,
            help=f"Automatic yfinance fetch error: {nifty_data_error}",
        )

    formula_target_utilisation_pct = max(
        0.0,
        min(
            100.0,
            50.0 + 0.5 * (100.0 - 2.0 * abs(float(nifty_200dma_deviation_pct))),
        ),
    )
    st.caption(
        f"Target utilisation = 50 + 0.5 × (100 − 2 × |{nifty_200dma_deviation_pct:.2f}%|) "
        f"= **{formula_target_utilisation_pct:.2f}% of lender maximum eligible loan**."
    )
    processing_fee_pre_gst = st.number_input(
        "Processing / renewal fee before GST (₹)", 0, 2_00_000, 5_000, 500
    )
    processing_gst_pct = st.number_input("GST on processing fee (%)", 0.0, 30.0, 18.0, 1.0)
    other_upfront_charges = st.number_input(
        "Other charges per sanction cycle (₹)", 0, 2_00_000, 0, 500
    )
    prepayment_charge_pct = st.number_input(
        "Prepayment / foreclosure charge (%)", 0.0, 10.0, 0.0, 0.25,
        help="Keep at 0 only if your lender actually waives it.",
    )

    st.header("5) Repayment")
    repayment_style = st.selectbox(
        "Repayment style",
        [
            "EMI-style manual principal prepayment",
            "Interest-only + sinking fund",
            "Interest-only, bullet principal at end",
        ],
    )
    sinking_return_pct = 0.0
    bullet_repayment_from_mf = False
    repay_bullet_from_stock_sleeve = False
    repayment_embedded_gain_pct = float(embedded_gain_pct)
    repayment_exemption_available = float(annual_exemption)
    if repayment_style == "Interest-only + sinking fund":
        sinking_return_pct = st.number_input(
            "Expected sinking-fund return (p.a. %)",
            min_value=0.0,
            max_value=12.0,
            value=6.0,
            step=0.25,
            help="Use a low-risk assumption if this money is meant to repay principal on a fixed date.",
        )
    elif repayment_style == "Interest-only, bullet principal at end":
        if use_case == "Disciplined leverage into stocks":
            bullet_source = st.selectbox(
                "Expected principal repayment source",
                ["Sell leveraged stock sleeve at horizon", "Future cash outside the portfolio"],
            )
            repay_bullet_from_stock_sleeve = bullet_source == "Sell leveraged stock sleeve at horizon"
            st.caption(
                "If the leveraged stock sleeve is sold to clear principal, the leverage dashboard estimates tax on the gain portion using the optional exit-tax inputs above."
            )
        else:
            bullet_source = st.selectbox(
                "Expected principal repayment source",
                ["Redeem mutual funds at horizon", "Future cash outside the portfolio"],
            )
            bullet_repayment_from_mf = bullet_source == "Redeem mutual funds at horizon"
            if bullet_repayment_from_mf:
                repayment_holding_months = int(holding_period_months) + int(horizon_months)
                repayment_embedded_gain_pct = estimate_embedded_gain_pct(
                    float(expected_return_pct), repayment_holding_months
                )
                st.metric(
                    "Estimated gain embedded in repayment-time redemption",
                    f"{repayment_embedded_gain_pct:.2f}%",
                )
                st.caption(
                    f"Auto-estimated using {repayment_holding_months} months "
                    f"(current holding tenure + loan horizon) at {expected_return_pct:.2f}% p.a."
                )
                repayment_exemption_available = st.number_input(
                    "Expected exemption available in repayment FY (₹)",
                    min_value=0,
                    max_value=10_00_000,
                    value=int(annual_exemption),
                    step=5_000,
                    help="Future-year exemption is uncertain; enter what you expect to remain unused in that financial year.",
                )


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------
if use_case == "Disciplined leverage into stocks":
    st.warning(
        "Leverage mode: historical XIRR is used only as a reference. The forward case applies your chosen haircut, "
        "and the strategy is evaluated against break-even return, cash-flow capacity, collateral risk and personal position-size limits."
    )

if eligible_collateral_value > portfolio_value:
    st.warning("Eligible collateral cannot logically exceed the total portfolio value. Check the inputs.")

max_lender_loan = eligible_collateral_value * lender_ltv_pct / 100.0
formula_target_sanction = max_lender_loan * formula_target_utilisation_pct / 100.0
initial_ltv = cash_required / eligible_collateral_value * 100.0 if eligible_collateral_value > 0 else math.inf

result = run_scenario(
    cash_required=float(cash_required),
    portfolio_value=float(portfolio_value),
    embedded_gain_pct=float(embedded_gain_pct),
    tax_rate_pct=float(tax_rate_pct),
    exemption_remaining=float(exemption_remaining),
    cess_pct=float(cess_pct),
    surcharge_pct=float(surcharge_pct),
    expected_return_pct=float(expected_return_pct),
    monthly_surplus=float(monthly_surplus),
    loan_rate_pct=float(loan_rate_pct),
    horizon_months=int(horizon_months),
    sanction_months=int(sanction_months),
    processing_fee_pre_gst=float(processing_fee_pre_gst),
    processing_gst_pct=float(processing_gst_pct),
    other_upfront_charges=float(other_upfront_charges),
    prepayment_charge_pct=float(prepayment_charge_pct),
    repayment_style=repayment_style,
    sinking_return_pct=float(sinking_return_pct),
    bullet_repayment_from_mf=bool(bullet_repayment_from_mf),
    repayment_embedded_gain_pct=float(repayment_embedded_gain_pct),
    repayment_exemption_available=float(repayment_exemption_available),
)

# Margin-call math based on initial principal. For amortizing loans, risk generally declines as principal is prepaid.
margin_call_collateral = cash_required / (lender_ltv_pct / 100.0) if lender_ltv_pct > 0 else math.inf
crash_to_margin_call_pct = (
    max(0.0, 1 - margin_call_collateral / eligible_collateral_value) * 100.0
    if eligible_collateral_value > 0
    else -math.inf
)


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
st.subheader("Decision dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Net cash needed", inr(cash_required))
c2.metric("Current drawn LTV", f"{initial_ltv:.1f}%" if math.isfinite(initial_ltv) else "N/A")
c3.metric("Lender maximum eligible loan", inr(max_lender_loan))
c4.metric("200-DMA formula target sanction", inr(formula_target_sanction))

st.caption(
    f"Formula utilisation: {formula_target_utilisation_pct:.2f}% of lender maximum "
    f"(Nifty 50 vs 200-DMA: {nifty_200dma_deviation_pct:+.2f}%)."
)

if cash_required > max_lender_loan:
    st.error(
        f"❌ Requested loan {inr(cash_required)} exceeds the lender-LTV capacity of {inr(max_lender_loan)}. "
        "LAMF is not feasible with the entered eligible collateral."
    )
elif cash_required > formula_target_sanction:
    st.warning(
        f"⚠️ Requested draw {inr(cash_required)} is within lender eligibility but exceeds the "
        f"200-DMA formula target sanction of {inr(formula_target_sanction)}."
    )
else:
    st.success(
        f"✅ Requested draw is within the 200-DMA formula target sanction of "
        f"{inr(formula_target_sanction)} and within lender eligibility."
    )

sell_feasible = result.gross_redemption <= portfolio_value
if not sell_feasible:
    st.error(
        f"Selling the entered portfolio cannot provide {inr(cash_required)} net after estimated tax: "
        f"the model would need to redeem about {inr(result.gross_redemption)} from a {inr(portfolio_value)} portfolio."
    )

if result.cashflow_shortfall > 0:
    st.error(
        f"💸 Cash-flow stress: debt service exceeds your stated monthly surplus by about "
        f"{inr(result.cashflow_shortfall / horizon_months)} per month. "
        "A loan that wins on spreadsheet wealth but strains monthly cash flow is not practical."
    )


# -----------------------------------------------------------------------------
# Dedicated disciplined stock-leverage dashboard
# -----------------------------------------------------------------------------
if use_case == "Disciplined leverage into stocks":
    leverage_result = run_leverage_scenario(
        borrowed_amount=float(cash_required),
        existing_stock_value=float(existing_stock_value),
        stock_return_pct=float(forward_stock_return_pct),
        monthly_surplus=float(monthly_surplus),
        loan_rate_pct=float(loan_rate_pct),
        horizon_months=int(horizon_months),
        sanction_months=int(sanction_months),
        processing_fee_pre_gst=float(processing_fee_pre_gst),
        processing_gst_pct=float(processing_gst_pct),
        other_upfront_charges=float(other_upfront_charges),
        prepayment_charge_pct=float(prepayment_charge_pct),
        repayment_style=repayment_style,
        sinking_return_pct=float(sinking_return_pct),
        repay_bullet_from_stock_sleeve=bool(repay_bullet_from_stock_sleeve),
        exit_tax_rate_pct=float(leverage_exit_tax_rate_pct),
        exit_exemption=float(leverage_exit_exemption),
        exit_cess_pct=float(cess_pct),
        exit_surcharge_pct=float(surcharge_pct),
    )

    def leverage_advantage_at_return(ret: float) -> float:
        return run_leverage_scenario(
            borrowed_amount=float(cash_required),
            existing_stock_value=float(existing_stock_value),
            stock_return_pct=float(ret),
            monthly_surplus=float(monthly_surplus),
            loan_rate_pct=float(loan_rate_pct),
            horizon_months=int(horizon_months),
            sanction_months=int(sanction_months),
            processing_fee_pre_gst=float(processing_fee_pre_gst),
            processing_gst_pct=float(processing_gst_pct),
            other_upfront_charges=float(other_upfront_charges),
            prepayment_charge_pct=float(prepayment_charge_pct),
            repayment_style=repayment_style,
            sinking_return_pct=float(sinking_return_pct),
            repay_bullet_from_stock_sleeve=bool(repay_bullet_from_stock_sleeve),
            exit_tax_rate_pct=float(leverage_exit_tax_rate_pct),
            exit_exemption=float(leverage_exit_exemption),
            exit_cess_pct=float(cess_pct),
            exit_surcharge_pct=float(surcharge_pct),
        ).advantage

    # Solve the forward stock CAGR at which leverage and no-leverage end wealth are equal.
    be_lo, be_hi = -50.0, 100.0
    be_flo, be_fhi = leverage_advantage_at_return(be_lo), leverage_advantage_at_return(be_hi)
    leverage_breakeven = None
    if be_flo == 0:
        leverage_breakeven = be_lo
    elif be_fhi == 0:
        leverage_breakeven = be_hi
    elif be_flo * be_fhi < 0:
        for _ in range(100):
            be_mid = (be_lo + be_hi) / 2.0
            be_fmid = leverage_advantage_at_return(be_mid)
            if be_flo * be_fmid <= 0:
                be_hi, be_fhi = be_mid, be_fmid
            else:
                be_lo, be_flo = be_mid, be_fmid
        leverage_breakeven = (be_lo + be_hi) / 2.0

    personal_max_borrow = float(existing_stock_value) * float(personal_max_leverage_pct) / 100.0
    leverage_ratio_pct = (
        float(cash_required) / float(existing_stock_value) * 100.0
        if existing_stock_value > 0
        else math.inf
    )
    debt_service_share_pct = (
        leverage_result.monthly_debt_service / float(monthly_surplus) * 100.0
        if monthly_surplus > 0
        else math.inf
    )
    forward_buffer = (
        float(forward_stock_return_pct) - leverage_breakeven
        if leverage_breakeven is not None
        else math.nan
    )

    st.divider()
    st.header("📈 Disciplined stock-leverage dashboard")
    st.caption(
        "This compares keeping your existing stock portfolio unlevered versus adding a stock sleeve funded by the LAMF draw. "
        "The same monthly investable surplus is used in both paths, so loan payments reduce what can continue into SIPs/investing."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Historical stock XIRR", f"{historical_stock_xirr_pct:.2f}%")
    m2.metric("Forward planning return", f"{forward_stock_return_pct:.2f}%")
    m3.metric(
        "Modeled break-even stock CAGR",
        f"{leverage_breakeven:.2f}%" if leverage_breakeven is not None else "No crossing",
    )
    m4.metric(
        "Forward cushion over break-even",
        f"{forward_buffer:+.2f} pp" if math.isfinite(forward_buffer) else "N/A",
    )

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Existing stock capital", inr(existing_stock_value))
    l2.metric("LAMF-funded stock sleeve", inr(cash_required))
    l3.metric(
        "Borrowed / existing stocks",
        f"{leverage_ratio_pct:.1f}%" if math.isfinite(leverage_ratio_pct) else "N/A",
    )
    l4.metric("Personal max leverage amount", inr(personal_max_borrow))

    w1, w2, w3, w4 = st.columns(4)
    w1.metric("No-leverage ending wealth", inr(leverage_result.no_leverage_end_wealth))
    w2.metric("Leveraged ending wealth", inr(leverage_result.leverage_end_wealth))
    w3.metric(
        "Incremental leverage advantage",
        inr(leverage_result.advantage),
        delta=f"{leverage_result.advantage / max(1.0, leverage_result.no_leverage_end_wealth) * 100:.2f}%",
    )
    w4.metric("Borrowed sleeve at horizon", inr(leverage_result.borrowed_sleeve_value))

    st.subheader("Discipline gates")
    gates = []
    gates.append(("Within lender maximum LTV", cash_required <= max_lender_loan,
                  f"Draw {inr(cash_required)} vs lender max {inr(max_lender_loan)}"))
    gates.append(("Within Nifty 200-DMA target sanction", cash_required <= formula_target_sanction,
                  f"Draw {inr(cash_required)} vs formula target {inr(formula_target_sanction)}"))
    gates.append(("Within personal stock-leverage cap", cash_required <= personal_max_borrow,
                  f"{leverage_ratio_pct:.1f}% borrowed/existing vs cap {personal_max_leverage_pct:.1f}%"))
    gates.append(("Debt service within monthly-surplus cap", debt_service_share_pct <= max_debt_service_share_pct,
                  f"{debt_service_share_pct:.1f}% of surplus vs cap {max_debt_service_share_pct:.1f}%"))
    if leverage_breakeven is not None:
        gates.append(("Forward return clears break-even margin", forward_buffer >= min_break_even_buffer_pct,
                      f"Buffer {forward_buffer:+.2f} pp vs required {min_break_even_buffer_pct:.2f} pp"))
    else:
        gates.append(("Forward return clears break-even margin", False,
                      "No break-even crossing found in -50% to +100% CAGR search range"))
    gates.append(("No monthly cash-flow shortfall", leverage_result.cashflow_shortfall <= 0,
                  f"Monthly debt service {inr(leverage_result.monthly_debt_service)} vs surplus {inr(monthly_surplus)}"))

    gate_df = pd.DataFrame([
        {"Gate": name, "Status": "PASS" if ok else "FAIL", "Detail": detail}
        for name, ok, detail in gates
    ])
    st.dataframe(gate_df, hide_index=True, width="stretch")
    failed_gates = [name for name, ok, _ in gates if not ok]
    if not failed_gates:
        st.success(
            "All configured discipline gates pass under the current assumptions. This means the setup fits your rules; "
            "it does not make the future stock return certain."
        )
    else:
        st.warning("Discipline gates failing: " + "; ".join(failed_gates))

    st.subheader("Loan carrying cost and cash-flow load")
    cst1, cst2, cst3, cst4 = st.columns(4)
    cst1.metric("Monthly debt service", inr(leverage_result.monthly_debt_service))
    cst2.metric("Debt service / monthly surplus", f"{debt_service_share_pct:.1f}%" if math.isfinite(debt_service_share_pct) else "N/A")
    cst3.metric("Total modeled interest", inr(leverage_result.total_interest))
    cst4.metric("Fees across sanction cycles", inr(leverage_result.total_fees))
    if leverage_result.exit_tax > 0:
        st.caption(
            f"If the bullet is repaid by selling the leveraged stock sleeve, estimated exit tax included: "
            f"{inr(leverage_result.exit_tax)}; gross sale needed to net principal: {inr(leverage_result.exit_gross_redemption)}."
        )
    if leverage_result.repayment_shortfall > 0:
        st.error(
            f"At the forward return assumption, the leveraged stock sleeve is short by about "
            f"{inr(leverage_result.repayment_shortfall)} of the gross amount needed to clear the bullet principal after tax."
        )

    st.subheader("Return sensitivity — incremental wealth from leverage")
    leverage_return_grid = sorted(set([
        -20.0, -10.0, 0.0, 8.0, 12.0,
        round(float(forward_stock_return_pct), 2),
        round(float(historical_stock_xirr_pct), 2),
        25.0,
    ]))
    lev_rows = []
    for ret in leverage_return_grid:
        temp = run_leverage_scenario(
            borrowed_amount=float(cash_required),
            existing_stock_value=float(existing_stock_value),
            stock_return_pct=float(ret),
            monthly_surplus=float(monthly_surplus),
            loan_rate_pct=float(loan_rate_pct),
            horizon_months=int(horizon_months),
            sanction_months=int(sanction_months),
            processing_fee_pre_gst=float(processing_fee_pre_gst),
            processing_gst_pct=float(processing_gst_pct),
            other_upfront_charges=float(other_upfront_charges),
            prepayment_charge_pct=float(prepayment_charge_pct),
            repayment_style=repayment_style,
            sinking_return_pct=float(sinking_return_pct),
            repay_bullet_from_stock_sleeve=bool(repay_bullet_from_stock_sleeve),
            exit_tax_rate_pct=float(leverage_exit_tax_rate_pct),
            exit_exemption=float(leverage_exit_exemption),
            exit_cess_pct=float(cess_pct),
            exit_surcharge_pct=float(surcharge_pct),
        )
        lev_rows.append({
            "Stock CAGR": f"{ret:.2f}%",
            "Borrowed sleeve at horizon": inr(temp.borrowed_sleeve_value),
            "No-leverage wealth": inr(temp.no_leverage_end_wealth),
            "Leveraged wealth": inr(temp.leverage_end_wealth),
            "Incremental advantage": inr(temp.advantage),
            "Result": "Ahead" if temp.advantage >= 0 else "Behind",
        })
    st.dataframe(pd.DataFrame(lev_rows), hide_index=True, width="stretch")

    st.subheader("Immediate drawdown stress on leveraged stock sleeve")
    shock_rows = []
    for shock in [0, 10, 20, 30, 40, 50, 60]:
        sleeve_after_shock = cash_required * (1 - shock / 100.0)
        sleeve_loss = cash_required - sleeve_after_shock
        net_sleeve_equity_vs_principal = sleeve_after_shock - cash_required
        shock_rows.append({
            "Immediate stock fall": f"-{shock}%",
            "Leveraged sleeve value": inr(sleeve_after_shock),
            "Mark-to-market loss": inr(sleeve_loss),
            "Sleeve value minus original principal": inr(net_sleeve_equity_vs_principal),
        })
    st.dataframe(pd.DataFrame(shock_rows), hide_index=True, width="stretch")
    st.caption(
        "The stock-sleeve drawdown and the LAMF margin call are different risks: the loan is secured by the pledged MF collateral, "
        "while the borrowed stock sleeve can simultaneously be down."
    )

    st.subheader("Collateral / margin-call stress")
    if eligible_collateral_value > 0:
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Current drawn LTV", f"{initial_ltv:.1f}%")
        cc2.metric("Collateral at lender-LTV breach", inr(margin_call_collateral))
        cc3.metric("Approx. collateral fall to breach", f"{crash_to_margin_call_pct:.1f}%")
        collateral_rows = []
        for crash in [0, 10, 20, 30, 40, 50, 60]:
            stressed_collateral = eligible_collateral_value * (1 - crash / 100.0)
            stressed_ltv = cash_required / stressed_collateral * 100.0 if stressed_collateral > 0 else math.inf
            principal_repay_needed = max(0.0, cash_required - stressed_collateral * lender_ltv_pct / 100.0)
            collateral_rows.append({
                "MF collateral fall": f"-{crash}%",
                "Collateral value": inr(stressed_collateral),
                "Resulting LTV": f"{stressed_ltv:.1f}%" if math.isfinite(stressed_ltv) else "∞",
                "Principal repay needed to restore lender max": inr(principal_repay_needed),
                "Status": "BREACH" if stressed_ltv > lender_ltv_pct else "OK",
            })
        st.dataframe(pd.DataFrame(collateral_rows), hide_index=True, width="stretch")

    st.info(
        f"Your historical XIRR of {historical_stock_xirr_pct:.2f}% is useful as evidence of past execution, but the leverage decision is based on the "
        f"haircut forward assumption of {forward_stock_return_pct:.2f}% and a modeled break-even of "
        f"{leverage_breakeven:.2f}% p.a." if leverage_breakeven is not None else
        "Your historical XIRR is useful as evidence of past execution, but the model could not find a break-even return in the search range."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Sell vs borrow comparison
# -----------------------------------------------------------------------------
st.subheader("A) Sell mutual funds vs B) Take LAMF")

comparison = pd.DataFrame(
    {
        "Metric": [
            "Cash available for expense",
            "MF units redeemed today",
            "Estimated tax triggered today",
            "Monthly debt service",
            "Monthly MF investment that can continue",
            f"Estimated investment wealth after {horizon_months} months",
            "Loan balance at horizon",
            "Total loan interest",
            "Total lender fees / charges",
        ],
        "Sell MF": [
            cash_required,
            result.gross_redemption,
            result.redemption_tax,
            0.0,
            monthly_surplus,
            result.sell_end_wealth,
            0.0,
            0.0,
            0.0,
        ],
        "Take LAMF": [
            cash_required,
            0.0,
            0.0,
            result.loan_monthly_debt_service,
            result.residual_monthly_investment,
            result.loan_end_wealth,
            result.ending_loan_balance,
            result.total_interest,
            result.total_fees,
        ],
    }
)

formatted_comparison = comparison.copy()
for col in ["Sell MF", "Take LAMF"]:
    formatted_comparison[col] = formatted_comparison[col].map(inr)
st.dataframe(formatted_comparison, width='stretch', hide_index=True)

adv1, adv2, adv3 = st.columns(3)
adv1.metric("Tax avoided today by not selling", inr(result.redemption_tax))
adv2.metric("Renewals assumed", f"{result.renewal_count}")
adv3.metric(
    "LAMF wealth advantage vs selling",
    inr(result.advantage),
    delta=f"{result.advantage / max(1.0, result.sell_end_wealth) * 100:.2f}%",
)

if use_case == "Disciplined leverage into stocks":
    st.warning(
        "No positive/negative verdict is issued for leveraged investing. The comparison above assumes the loan funds a cash need, "
        "not an additional equity purchase."
    )
elif cash_required > max_lender_loan:
    st.error("Verdict: LAMF is not feasible at the entered lender LTV.")
elif not sell_feasible:
    st.info("Verdict: the 'sell MF' alternative is not feasible with the entered portfolio; compare LAMF with other funding sources instead.")
elif result.cashflow_shortfall > 0:
    st.warning("Verdict: financially stressed — fix monthly affordability before comparing expected wealth outcomes.")
elif result.advantage > 0:
    st.success(
        f"Model result: keeping the portfolio pledged is ahead by about {inr(result.advantage)} under your assumptions. "
        "This is only attractive if you can tolerate the LTV risk and repay on schedule."
    )
else:
    st.warning(
        f"Model result: selling units is ahead by about {inr(abs(result.advantage))} under your assumptions. "
        "LAMF does not compensate for its interest/fees in this scenario."
    )


# -----------------------------------------------------------------------------
# Tax details
# -----------------------------------------------------------------------------
with st.expander("Tax calculation details", expanded=False):
    estimated_gain_realised = result.gross_redemption * embedded_gain_pct / 100.0
    taxable_gain = max(0.0, estimated_gain_realised - exemption_remaining)
    tax_df = pd.DataFrame(
        {
            "Item": [
                "Net cash needed",
                "Gross MF redemption required",
                "Estimated embedded gain realised",
                "Exemption remaining",
                "Estimated taxable gain",
                "Estimated tax incl. surcharge/cess",
            ],
            "Value": [
                cash_required,
                result.gross_redemption,
                estimated_gain_realised,
                exemption_remaining,
                taxable_gain,
                result.redemption_tax,
            ],
        }
    )
    tax_df["Value"] = tax_df["Value"].map(inr)
    st.dataframe(tax_df, hide_index=True, width='stretch')
    st.caption(
        "This is an estimate. Mutual-fund taxation can depend on fund classification, acquisition/redemption dates, "
        "grandfathering/cost rules, set-off of losses and your overall tax situation."
    )


# -----------------------------------------------------------------------------
# Margin/LTV safety
# -----------------------------------------------------------------------------
st.subheader("Margin-call / collateral safety")

if eligible_collateral_value <= 0:
    st.error("Enter a positive eligible collateral value to evaluate LTV safety.")
else:
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Initial LTV", f"{initial_ltv:.1f}%")
    mc2.metric("Collateral value at lender-LTV breach", inr(margin_call_collateral))
    mc3.metric("Approx. fall to lender-LTV breach", f"{crash_to_margin_call_pct:.1f}%")

    crash_levels = [0, 10, 20, 30, 40, 50, 60]
    stress_rows = []
    for crash in crash_levels:
        stressed_collateral = eligible_collateral_value * (1 - crash / 100.0)
        stressed_ltv = cash_required / stressed_collateral * 100.0 if stressed_collateral > 0 else math.inf
        top_up_needed = max(0.0, cash_required / (lender_ltv_pct / 100.0) - stressed_collateral)
        principal_repay_needed = max(0.0, cash_required - stressed_collateral * lender_ltv_pct / 100.0)
        stress_rows.append(
            {
                "Market fall": f"-{crash}%",
                "Collateral value": inr(stressed_collateral),
                "LTV": f"{stressed_ltv:.1f}%" if math.isfinite(stressed_ltv) else "∞",
                "Extra collateral to restore max LTV": inr(top_up_needed),
                "Principal repayment to restore max LTV": inr(principal_repay_needed),
                "Status": "BREACH" if stressed_ltv > lender_ltv_pct else "OK",
            }
        )
    st.dataframe(pd.DataFrame(stress_rows), width='stretch', hide_index=True)

    st.caption(
        "The breach calculation is intentionally conservative and uses the original principal. Under EMI-style principal prepayment, "
        "your actual LTV normally improves over time as the outstanding balance falls. Lender cure periods and liquidation rules vary."
    )


# -----------------------------------------------------------------------------
# Repayment details
# -----------------------------------------------------------------------------
st.subheader("Repayment and renewal details")

rep1, rep2, rep3, rep4 = st.columns(4)
rep1.metric("Monthly debt service", inr(result.loan_monthly_debt_service))
rep2.metric("Total interest", inr(result.total_interest))
rep3.metric("Fees across sanction cycles", inr(result.total_fees))
rep4.metric("Ending principal outstanding", inr(result.ending_loan_balance))

if repayment_style == "Interest-only + sinking fund":
    st.info(
        f"Sinking fund is projected to reach about {inr(result.sinking_fund_value)} by month {horizon_months}, "
        f"against a bullet principal of {inr(result.ending_loan_balance)}."
    )
elif repayment_style == "Interest-only, bullet principal at end":
    if bullet_repayment_from_mf:
        st.warning(
            f"Bullet risk: the full principal remains due at the end. With the auto-estimated repayment-time embedded gain, "
            f"the model includes about {inr(result.bullet_repayment_tax)} of capital-gains tax in the final MF redemption. "
            "A market fall near repayment can still make the timing painful."
        )
    else:
        st.warning(
            "Bullet risk: the full principal remains due at the end and must come from future cash outside the modeled portfolio."
        )

if horizon_months > sanction_months:
    st.warning(
        f"The model assumes {result.renewal_count} renewal(s) and repeats charges. It also assumes the loan can be renewed. "
        "In real life, renewal rate, eligible schemes, LTV and rollover terms can change; the lender may require principal settlement."
    )


# -----------------------------------------------------------------------------
# Break-even return
# -----------------------------------------------------------------------------
st.subheader("Break-even portfolio return")


def advantage_at_return(ret: float) -> float:
    r = run_scenario(
        cash_required=float(cash_required),
        portfolio_value=float(portfolio_value),
        embedded_gain_pct=float(embedded_gain_pct),
        tax_rate_pct=float(tax_rate_pct),
        exemption_remaining=float(exemption_remaining),
        cess_pct=float(cess_pct),
        surcharge_pct=float(surcharge_pct),
        expected_return_pct=float(ret),
        monthly_surplus=float(monthly_surplus),
        loan_rate_pct=float(loan_rate_pct),
        horizon_months=int(horizon_months),
        sanction_months=int(sanction_months),
        processing_fee_pre_gst=float(processing_fee_pre_gst),
        processing_gst_pct=float(processing_gst_pct),
        other_upfront_charges=float(other_upfront_charges),
        prepayment_charge_pct=float(prepayment_charge_pct),
        repayment_style=repayment_style,
        sinking_return_pct=float(sinking_return_pct),
        bullet_repayment_from_mf=bool(bullet_repayment_from_mf),
        repayment_embedded_gain_pct=float(repayment_embedded_gain_pct),
        repayment_exemption_available=float(repayment_exemption_available),
    )
    return r.advantage


lo, hi = -20.0, 40.0
flo, fhi = advantage_at_return(lo), advantage_at_return(hi)
if flo == 0:
    breakeven_return = lo
elif fhi == 0:
    breakeven_return = hi
elif flo * fhi < 0:
    for _ in range(80):
        mid = (lo + hi) / 2.0
        fmid = advantage_at_return(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    breakeven_return = (lo + hi) / 2.0
else:
    breakeven_return = None

if breakeven_return is not None:
    st.metric("Return at which LAMF and selling are roughly equal", f"{breakeven_return:.2f}% p.a.")
    if expected_return_pct < breakeven_return:
        st.caption("Your entered expected return is below the modeled break-even return.")
    else:
        st.caption("Your entered expected return is above the modeled break-even return — but that return is not guaranteed.")
else:
    st.caption("No break-even crossing was found between -20% and +40% p.a. for the current inputs.")


# -----------------------------------------------------------------------------
# Sensitivity matrix
# -----------------------------------------------------------------------------
st.subheader("Sensitivity: LAMF advantage vs selling")
return_grid = sorted(set([0, 4, 6, 8, 10, 12, 15, round(float(expected_return_pct), 1)]))
rate_grid = sorted(set([8, 9, 10, 11, 12, 14, round(float(loan_rate_pct), 1)]))

matrix = []
for ret in return_grid:
    row = []
    for lr in rate_grid:
        temp = run_scenario(
            cash_required=float(cash_required),
            portfolio_value=float(portfolio_value),
            embedded_gain_pct=float(embedded_gain_pct),
            tax_rate_pct=float(tax_rate_pct),
            exemption_remaining=float(exemption_remaining),
            cess_pct=float(cess_pct),
            surcharge_pct=float(surcharge_pct),
            expected_return_pct=float(ret),
            monthly_surplus=float(monthly_surplus),
            loan_rate_pct=float(lr),
            horizon_months=int(horizon_months),
            sanction_months=int(sanction_months),
            processing_fee_pre_gst=float(processing_fee_pre_gst),
            processing_gst_pct=float(processing_gst_pct),
            other_upfront_charges=float(other_upfront_charges),
            prepayment_charge_pct=float(prepayment_charge_pct),
            repayment_style=repayment_style,
            sinking_return_pct=float(sinking_return_pct),
            bullet_repayment_from_mf=bool(bullet_repayment_from_mf),
            repayment_embedded_gain_pct=float(repayment_embedded_gain_pct),
            repayment_exemption_available=float(repayment_exemption_available),
        )
        row.append(temp.advantage)
    matrix.append(row)

sens_df = pd.DataFrame(
    matrix,
    index=[f"MF {r:g}%" for r in return_grid],
    columns=[f"Loan {r:g}%" for r in rate_grid],
)
st.dataframe(sens_df.style.format(lambda x: inr(x)), width='stretch')
st.caption("Positive = LAMF path ends with higher modeled financial wealth. Negative = selling path is ahead.")


# -----------------------------------------------------------------------------
# Wealth chart
# -----------------------------------------------------------------------------
st.subheader("Visual comparison")
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["Sell MF", "Take LAMF"]
values = [result.sell_end_wealth, result.loan_end_wealth]
bars = ax.bar(labels, values)
for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        inr(val),
        ha="center",
        va="bottom",
        fontsize=10,
    )
ax.set_ylabel("Estimated ending financial wealth (₹)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: inr(x)))
ax.set_title(f"Estimated position after {horizon_months} months")
ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
st.pyplot(fig, width='content')
plt.close(fig)


# -----------------------------------------------------------------------------
# Alternative loan comparator: flat vs reducing balance
# -----------------------------------------------------------------------------
with st.expander("Compare an alternative flat-rate vehicle / consumer loan"):
    st.write(
        "A quoted flat rate is not directly comparable with a reducing-balance annual rate because interest is charged "
        "on the original principal even while installments reduce the amount economically outstanding."
    )
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        flat_rate = st.number_input("Quoted flat rate (p.a. %)", 0.0, 40.0, 8.0, 0.25)
    with fc2:
        flat_months = st.number_input("Flat-loan tenure (months)", 1, 120, 48, 1)
    with fc3:
        flat_principal = st.number_input(
            "Principal for comparison (₹)", 10_000, 1_00_00_000, int(cash_required), 10_000
        )
    effective_rate = flat_rate_loan_irr(float(flat_rate), float(flat_principal), int(flat_months))
    st.metric("Approx. effective reducing-balance equivalent", f"{effective_rate:.2f}% p.a.")
    st.caption("Approximation assumes equal monthly installments and no fees/insurance/prepayment charges.")


# -----------------------------------------------------------------------------
# Practical checklist
# -----------------------------------------------------------------------------
st.subheader("Practical checks before using LAMF")
checks = [
    "Confirm the exact eligible schemes/units and haircut/LTV in the lender's sanction letter.",
    "Use the 200-DMA formula target as a sanction/draw discipline; lender maximum LTV remains a separate eligibility limit.",
    "Know the margin-call cure period: cash repayment, extra collateral, or forced redemption can be required.",
    "Budget monthly interest/principal from income; do not depend on market returns to make mandatory payments.",
    "If the loan horizon exceeds the sanction tenure, verify rollover/renewal mechanics instead of assuming automatic renewal.",
    "Check processing fee + GST, annual/renewal charges, documentation, foreclosure/prepayment terms and penal interest.",
    "For the sell alternative, use the tax lot / holding-period information for the units actually likely to be redeemed.",
    "Prefer LAMF for a genuine funding need when it is cheaper and manageable—not simply to increase market leverage.",
]
for item in checks:
    st.write(f"• {item}")

st.caption(
    "Tax defaults in the app are editable reference inputs. Verify the applicable tax law and lender terms at the time you transact."
)
