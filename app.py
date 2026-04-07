"""
Streamlit entrypoint: product-style demo for creator monetization intelligence.
Run: streamlit run app.py
"""

from __future__ import annotations
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_analysis import generate_openai_analysis
from scoring import add_creator_scores, decision_summary, rank_explanation_bullets

st.set_page_config(page_title="Creator Decision Agent", layout="wide")

# ---------- Product header ----------
st.title("Creator Decision Agent")
st.markdown("> This is not a dashboard. This is an AI decision agent.")
st.markdown("""
### AI-powered monetization intelligence for the creator economy

We build an AI decision agent that evaluates creator commercial value and predicts monetization potential.

**What this agent does:**
- Predicts creator revenue potential
- Scores commercial value
- Identifies monetization opportunities
- Explains why a creator is valuable
- Scans an entire creator niche to surface top prospects
""")

# ---------- Dataset setup ----------
DATA_DIR = Path("datasets")

NICHE_FILES = {
    "Beauty": DATA_DIR / "beauty_creators.csv",
    "Fitness": DATA_DIR / "fitness_creators.csv",
    "Lifestyle": DATA_DIR / "lifestyle_creators.csv",
}

REQUIRED_COLS = {
    "username",
    "followers",
    "avg_views",
    "avg_likes",
    "avg_comments",
    "growth_30d",
    "niche",
}

def load_niche_df(label: str) -> pd.DataFrame:
    path = NICHE_FILES[label]
    if not path.exists():
        st.error(f"Missing dataset: {path}")
        st.stop()

    df_raw = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df_raw.columns)
    if missing:
        st.error(f"{path.name} is missing columns: {sorted(missing)}")
        st.stop()

    df = add_creator_scores(df_raw)

    if "rank" not in df.columns:
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

    return df

def resolve_username(name: str, df: pd.DataFrame) -> str | None:
    """Match username from CSV (case-insensitive, strip @)."""
    q = name.strip().lstrip("@").lower()
    if not q:
        return None
    for u in df["username"]:
        if str(u).lower() == q:
            return str(u)
    return None

# ---------- Noise detection ----------
def detect_paid_noise(row: pd.Series, df_cohort: pd.DataFrame) -> str:
    """
    Simple public-signal heuristic for potential paid traffic / noise.
    """
    avg_views = row.get("avg_views", 0)
    engagement_rate = row.get("engagement_rate", 0)
    growth_30d = row.get("growth_30d", 0)
    avg_likes = row.get("avg_likes", 0)
    avg_comments = row.get("avg_comments", 0)

    view_pct = df_cohort["avg_views"].rank(pct=True).get(row.name, 0.5)
    engagement_pct = df_cohort["engagement_rate"].rank(pct=True).get(row.name, 0.5)
    growth_pct = df_cohort["growth_30d"].rank(pct=True).get(row.name, 0.5)

    signals = 0
    if avg_views > 50000 and engagement_rate < 0.01:
        signals += 1
    if view_pct > 0.9 and engagement_pct < 0.2:
        signals += 1
    if growth_pct > 0.9 and engagement_pct < 0.3:
        signals += 1
    if avg_views > 10000 and (avg_likes / max(avg_views, 1) < 0.005 or avg_comments / max(avg_views, 1) < 0.001):
        signals += 1

    if signals >= 3:
        return "Likely Paid Noise"
    if signals >= 1:
        return "Possible Noise"
    return "Normal"

def market_scan_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return top picks / watchlist / risk tables for niche-level scan.
    """
    scan_df = df.copy()

    noise_flags = []
    decisions = []
    for idx, row in scan_df.iterrows():
        ds = decision_summary(row, scan_df)
        noise = detect_paid_noise(row, scan_df)
        noise_flags.append(noise)
        decisions.append(ds)

    scan_df["noise_flag"] = noise_flags
    scan_df["monetization_verdict"] = [d["monetization_verdict"] for d in decisions]
    scan_df["traffic_monetization_gap"] = [d["traffic_monetization_gap"] for d in decisions]
    scan_df["recommended_action"] = [d["recommended_action"] for d in decisions]

    top_picks = (
        scan_df[scan_df["noise_flag"] == "Normal"]
        .sort_values(["final_score", "growth_30d"], ascending=[False, False])
        .head(5)
    )

    watchlist = (
        scan_df[
            (scan_df["noise_flag"] != "Likely Paid Noise")
            & (scan_df["growth_30d"] > scan_df["growth_30d"].median())
            & (scan_df["final_score"] >= scan_df["final_score"].median())
        ]
        .sort_values(["growth_30d", "final_score"], ascending=[False, False])
        .head(5)
    )

    risk = (
        scan_df[scan_df["noise_flag"] != "Normal"]
        .sort_values(["noise_flag", "final_score"], ascending=[True, False])
        .head(5)
    )

    return top_picks, watchlist, risk

# ---------- Session ----------
if "selected_niche" not in st.session_state:
    st.session_state.selected_niche = "Beauty"
if "analyzed_user" not in st.session_state:
    st.session_state.analyzed_user = None
if "market_scan_ran" not in st.session_state:
    st.session_state.market_scan_ran = False

# ---------- Hero ----------
st.markdown("## Creator Monetization Intelligence")
st.markdown("*Identify which creators can actually monetize, not just generate traffic*")

# ---------- Controls ----------
left, right = st.columns([1.2, 1.8])

with left:
    niche_label = st.selectbox(
        "Select niche",
        options=list(NICHE_FILES.keys()),
        index=list(NICHE_FILES.keys()).index(st.session_state.selected_niche),
    )
    st.session_state.selected_niche = niche_label

df = load_niche_df(niche_label)
demo_usernames = df["username"].astype(str).tolist()

with right:
    st.caption(f"Loaded niche dataset: **{niche_label}** · {len(df)} creators")

# ---------- Market scan ----------
st.markdown("### Market Scan")
scan_col1, scan_col2 = st.columns([1, 3])

with scan_col1:
    if st.button("Run Market Scan", type="primary", use_container_width=True):
        scan_box = st.empty()
        with scan_box.container():
            st.markdown("##### AI Agent Process")
            s1 = st.empty()
            s2 = st.empty()
            s3 = st.empty()

            s1.markdown("⏳ Running market scan...")
            time.sleep(0.5)
            s1.markdown("✅ Running market scan...")

            s2.markdown("⏳ Scoring engagement quality...")
            time.sleep(0.7)
            s2.markdown("✅ Scoring engagement quality...")

            s3.markdown("⏳ Detecting monetization gaps...")
            time.sleep(0.7)
            s3.markdown("✅ Detecting monetization gaps...")
            time.sleep(0.4)
        scan_box.empty()
        st.session_state.market_scan_ran = True

with scan_col2:
    st.caption("Scan an entire niche to surface top signing opportunities, watchlist creators, and potential risk flags.")

if st.session_state.market_scan_ran:
    top_picks, watchlist, risk = market_scan_summary(df)

    st.markdown("#### Scan output")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("##### Top creators to sign")
        st.dataframe(
            top_picks[["username", "final_score", "growth_30d", "recommended_action"]].rename(
                columns={
                    "username": "Creator",
                    "final_score": "Score",
                    "growth_30d": "Growth 30d",
                    "recommended_action": "Action",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with a2:
        st.markdown("##### Watchlist")
        st.dataframe(
            watchlist[["username", "final_score", "growth_30d", "traffic_monetization_gap"]].rename(
                columns={
                    "username": "Creator",
                    "final_score": "Score",
                    "growth_30d": "Growth 30d",
                    "traffic_monetization_gap": "Gap",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with a3:
        st.markdown("##### Risk flags")
        if len(risk) == 0:
            st.success("No major paid-noise flags detected in this niche.")
        else:
            st.dataframe(
                risk[["username", "final_score", "noise_flag"]].rename(
                    columns={
                        "username": "Creator",
                        "final_score": "Score",
                        "noise_flag": "Risk",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

# ---------- Individual analysis ----------
st.markdown("### Individual Analysis")
with st.container():
    with st.form("analyze_form", clear_on_submit=False):
        username_input = st.text_input(
            "Enter TikTok username",
            placeholder="@creator or creator handle",
            help="Matches a row in the selected niche dataset (case-insensitive).",
        )
        demo_pick = st.selectbox(
            "Or select sample creator",
            options=[""] + demo_usernames,
            format_func=lambda x: "— Pick a sample profile —" if x == "" else x,
        )
        submitted = st.form_submit_button("Analyze creator")

    if submitted:
        target = None

        if username_input.strip():
            target = resolve_username(username_input, df)
            if target is None:
                st.error(
                    f"No sample profile matches “{username_input.strip()}”. "
                    "Use a username from the selected niche or pick from the dropdown."
                )
                st.session_state.analyzed_user = None
        elif demo_pick:
            target = demo_pick
        else:
            st.warning("Enter a TikTok username or choose a sample creator.")
            st.session_state.analyzed_user = None

        if target is not None:
            thinking_box = st.empty()
            progress_box = st.empty()

            with thinking_box.container():
                st.markdown("##### AI Agent Process")
                step1 = st.empty()
                step2 = st.empty()
                step3 = st.empty()

                step1.markdown("⏳ Running decision agent...")
                time.sleep(0.6)

                step1.markdown("✅ Running decision agent...")
                step2.markdown("⏳ Scoring engagement quality...")
                time.sleep(0.8)

                step2.markdown("✅ Scoring engagement quality...")
                step3.markdown("⏳ Detecting monetization gap...")
                time.sleep(0.8)

                step3.markdown("✅ Detecting monetization gap...")
                progress_box.info("⏳ Generating monetization verdict...")
                time.sleep(0.9)

            progress_box.success("✅ Decision ready")
            st.session_state.analyzed_user = target
            time.sleep(0.4)
            thinking_box.empty()
            progress_box.empty()

# ---------- Results ----------
analyzed = st.session_state.analyzed_user
if not analyzed:
    st.info("Run a market scan or analyze an individual creator to see monetization output.")
elif analyzed not in set(df["username"].astype(str)):
    st.session_state.analyzed_user = None
    st.warning("Saved profile is no longer in the selected niche dataset; run analysis again.")
else:
    row = df[df["username"].astype(str) == analyzed].iloc[0]

    st.divider()
    st.markdown("### Results")
    st.markdown("""
### Why this matters

Most creator tools provide analytics.

This agent goes further — it **makes economic decisions**.

It helps brands, investors, and agencies answer:

> *"Which creators are actually worth signing or backing?"*
""")

    ds = decision_summary(row, df)
    ds["noise_flag"] = detect_paid_noise(row, df)

    if ds["noise_flag"] == "Likely Paid Noise":
        ds["monetization_verdict"] = "Low (Noise detected)"
        ds["traffic_monetization_gap"] = "High traffic, low genuine engagement"
        ds["recommended_action"] = "Pass / Monitor only"
    elif ds["noise_flag"] == "Possible Noise":
        ds["monetization_verdict"] += " (Possible Noise)"
        ds["recommended_action"] = "Pilot test with caution"

    st.markdown("##### Decision Summary")
    d1, d2, d3 = st.columns(3)
    _card_style = (
        "background:#f8fafc;padding:1rem 1.1rem;border-radius:10px;"
        "border:1px solid #e2e8f0;border-left:5px solid {accent};"
        "min-height:5.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.06);"
    )

    with d1:
        st.markdown(
            f"""
            <div style="{_card_style.format(accent='#0d9488')}">
                <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;">
                    Monetization Verdict
                </div>
                <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-top:0.45rem;line-height:1.3;">
                    {ds['monetization_verdict']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d2:
        st.markdown(
            f"""
            <div style="{_card_style.format(accent='#2563eb')}">
                <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;">
                    Traffic vs Monetization Gap
                </div>
                <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-top:0.45rem;line-height:1.3;">
                    {ds['traffic_monetization_gap']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with d3:
        st.markdown(
            f"""
            <div style="{_card_style.format(accent='#7c3aed')}">
                <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.04em;">
                    Recommended Action
                </div>
                <div style="font-size:1.05rem;font-weight:700;color:#0f172a;margin-top:0.45rem;line-height:1.3;">
                    {ds['recommended_action']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"#### @{row['username']}")
        st.caption(f"Niche · **{row.get('niche', '—')}**  ·  Rank **#{int(row['rank'])}** in selected niche")
    with col_b:
        st.metric("Final score", f"{row['final_score']:.2f}", help="Blend of reach, engagement, and growth (0–100).")

    st.markdown("##### Performance snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Followers", f"{row['followers']:,.0f}")
    c2.metric("Avg views", f"{row['avg_views']:,.0f}")
    c3.metric("Avg likes", f"{row['avg_likes']:,.0f}")
    c4.metric("Avg comments", f"{row['avg_comments']:,.0f}")
    c5, c6 = st.columns(2)
    c5.metric("Growth (30d)", f"{row['growth_30d']:.1%}")
    c6.metric("Engagement rate", f"{row['engagement_rate']:.2%}")

    st.markdown("##### Monetization scores (0–100)")
    st.caption("Reach · Engagement · Growth — normalized within the selected niche; final score is their average.")
    sb1, sb2, sb3 = st.columns(3)
    sb1.metric("Reach", f"{row['reach_score']:.2f}")
    sb2.metric("Engagement", f"{row['engagement_score']:.2f}")
    sb3.metric("Growth", f"{row['growth_score']:.2f}")

    rank_why = rank_explanation_bullets(row, df)
    with st.expander("Why this rank?", expanded=True):
        if rank_why:
            for line in rank_why:
                st.markdown(f"- {line}")
        else:
            st.caption("No standout rank bullets for this profile vs. the selected cohort; final score still reflects the blended model.")

    st.markdown("##### Intelligence memo")
    analysis_md, analysis_mode = generate_openai_analysis(row, df)
    st.caption("Mode: OpenAI" if analysis_mode == "openai" else "Mode: Rule-based fallback")
    st.markdown(analysis_md)

with st.expander("Sample data (selected niche)", expanded=False):
    st.caption("For demo transparency — not part of the default workflow.")
    st.dataframe(
        df[
            [
                "rank",
                "username",
                "niche",
                "final_score",
                "followers",
                "avg_views",
                "engagement_rate",
                "growth_30d",
            ]
        ].sort_values("rank"),
        use_container_width=True,
        hide_index=True,
    )

st.markdown("""
---

### Future Vision

From creator monetization → to marketing decision infrastructure

We aim to become the AI layer for:
- Creator economy
- Brand partnerships
- Marketing budget allocation
""")
