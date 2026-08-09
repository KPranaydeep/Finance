"""Indian retail-investor Loan Against Mutual Funds (LAMF) simulator.

This Streamlit app compares two ways to fund a genuine cash requirement:
A) Redeem mutual-fund units and pay applicable capital-gains tax.
B) Borrow against eligible mutual-fund units and keep the portfolio invested.

The model intentionally does NOT assume that borrowed money is reinvested in equity.
A separate high-risk leverage mode is shown only for education and is never presented
as the default recommendation.

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
st.caption("Build: 2026-08-09 YFINANCE • automatic Nifty 50 200-DMA target sanction")
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
            "Invest borrowed money (high-risk leverage)",
        ],
    )
    cash_required = st.number_input(
        "Net cash required (₹)",
        min_value=25_000,
        max_value=5_00_00_000,
        value=5_00_000,
        step=25_000,
    )
    monthly_surplus = st.number_input(
        "Monthly amount available for SIP / debt service (₹)",
        min_value=0,
        max_value=10_00_000,
        value=25_000,
        step=1_000,
        help="Use the amount you can sustainably allocate each month. The comparison keeps this budget equal across both paths.",
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
                "(current holding tenure + loan horizon) at {expected_return_pct:.2f}% p.a."
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
if use_case == "Invest borrowed money (high-risk leverage)":
    st.error(
        "⚠️ High-risk leverage mode selected. Loan interest is certain; market returns are not. "
        "This app will still show collateral and cash-flow risk, but it will not label leveraged investing as 'recommended'."
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

if use_case == "Invest borrowed money (high-risk leverage)":
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
