"""
Streamlit entrypoint: product-style demo for creator monetization intelligence.
Run: streamlit run app.py
"""

from __future__ import annotations
import pandas as pd
import streamlit as st
import time
from llm_analysis import generate_openai_analysis
from scoring import add_creator_scores, decision_summary, rank_explanation_bullets

st.set_page_config(page_title="Creator Decision Agent", layout="wide")

st.title("Creator Decision Agent")

st.markdown("> This is an AI decision agent.")

st.markdown("""
### AI-powered monetization intelligence for the creator economy

We build an AI decision agent that evaluates creator commercial value and predicts monetization potential.

---

**What this agent does:**
- Predicts creator revenue potential
- Scores commercial value
- Identifies monetization opportunities
- Explains why a creator is valuable

---
""")


# --- Load & score data ---
data_path = "data.csv"
try:
    df_raw = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Missing `{data_path}`. Place your CSV next to `app.py`.")
    st.stop()

required = {"username", "followers", "avg_views", "avg_likes", "avg_comments", "growth_30d"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"CSV is missing columns: {sorted(missing)}")
    st.stop()

df = add_creator_scores(df_raw)
demo_usernames = df["username"].tolist()


def resolve_username(name: str) -> str | None:
    """Match username from CSV (case-insensitive, strip @)."""
    q = name.strip().lstrip("@").lower()
    if not q:
        return None
    for u in df["username"]:
        if str(u).lower() == q:
            return str(u)
    return None


# ========== Noise detection ==========
def detect_paid_noise(row, df_cohort) -> str:
    """
    判定 Creator 是否存在投流/噪音行为
    """
    avg_views = row.get("avg_views", 0)
    engagement_rate = row.get("engagement_rate", 0)
    growth_30d = row.get("growth_30d", 0)
    avg_likes = row.get("avg_likes", 0)
    avg_comments = row.get("avg_comments", 0)

    # --- 相对指标 ---
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
    elif signals >= 1:
        return "Possible Noise"
    else:
        return "Normal"


# --- Session ---
if "analyzed_user" not in st.session_state:
    st.session_state.analyzed_user = None

# ========== Hero ==========
st.markdown("## Creator Monetization Intelligence")
st.markdown("*Identify which creators can actually monetize, not just generate traffic*")
st.markdown("")

# ========== Input ==========
with st.container():
    st.markdown("##### Run analysis")
    with st.form("analyze_form", clear_on_submit=False):
        username_input = st.text_input(
            "Enter TikTok username",
            placeholder="@creator or creator handle",
            help="Matches a row in the demo CSV (case-insensitive).",
        )
        demo_pick = st.selectbox(
            "Or select demo creator",
            options=[""] + demo_usernames,
            format_func=lambda x: "— Pick a demo profile —" if x == "" else x,
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        target = None

    if username_input.strip():
        target = resolve_username(username_input)
        if target is None:
            st.error(
                f"No demo profile matches “{username_input.strip()}”. "
                "Use a username from the demo list or pick from the dropdown."
            )
            st.session_state.analyzed_user = None

    elif demo_pick:
        target = demo_pick

    else:
        st.warning("Enter a TikTok username or choose a demo creator.")
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

# ========== Results ==========
analyzed = st.session_state.analyzed_user
if not analyzed:
    st.info("Enter a handle or pick a demo creator, then click **Analyze** to see the monetization view.")
elif analyzed not in set(df["username"].astype(str)):
    st.session_state.analyzed_user = None
    st.warning("Saved profile is no longer in the dataset; run analysis again.")
else:
    row = df[df["username"].astype(str) == analyzed].iloc[0]

    st.divider()
    st.markdown("### Results")
    st.markdown("""
    ---

    ### Why this matters

    Most creator tools provide analytics.

    This agent goes further — it **makes economic decisions**.

    It helps brands, investors, and creators answer:

    > *"How valuable is this creator, and how can they monetize better?"*
    """)
    # --- Decision summary ---
    ds = decision_summary(row, df)

    # --- Add noise detection ---
    ds["noise_flag"] = detect_paid_noise(row, df)

    # Adjust monetization verdict based on noise
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

    # --- Cards ---
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

    # --- Profile header ---
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown(f"#### @{row['username']}")
        st.caption(f"Niche · **{row.get('niche', '—')}**  ·  Rank **#{int(row['rank'])}** in demo cohort")
    with col_b:
        st.metric("Final score", f"{row['final_score']:.2f}", help="Blend of reach, engagement, and growth (0–100).")

    # --- Performance snapshot ---
    st.markdown("##### Performance snapshot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Followers", f"{row['followers']:,.0f}")
    c2.metric("Avg views", f"{row['avg_views']:,.0f}")
    c3.metric("Avg likes", f"{row['avg_likes']:,.0f}")
    c4.metric("Avg comments", f"{row['avg_comments']:,.0f}")
    c5, c6 = st.columns(2)
    c5.metric("Growth (30d)", f"{row['growth_30d']:,.0f}")
    c6.metric("Engagement rate", f"{row['engagement_rate']:.4f}")

    # --- Monetization scores ---
    st.markdown("##### Monetization scores (0–100)")
    st.caption("Reach · Engagement · Growth — normalized within this demo file; final score is their average.")
    sb1, sb2, sb3 = st.columns(3)
    sb1.metric("Reach", f"{row['reach_score']:.2f}")
    sb2.metric("Engagement", f"{row['engagement_score']:.2f}")
    sb3.metric("Growth", f"{row['growth_score']:.2f}")

    # --- Rank explanation ---
    rank_why = rank_explanation_bullets(row, df)
    with st.expander("Why this rank?", expanded=True):
        if rank_why:
            for line in rank_why:
                st.markdown(f"- {line}")
        else:
            st.caption("No standout rank bullets for this profile vs. the cohort; final score still reflects the blended model.")

    # --- AI Analysis ---
    st.markdown("##### Intelligence memo")
    analysis_md, analysis_mode = generate_openai_analysis(row, df)
    if analysis_mode == "openai":
        st.caption("Mode: OpenAI")
    else:
        st.caption("Mode: Rule-based fallback")
    st.markdown(analysis_md)

# ========== Dataset hidden by default ==========
with st.expander("Demo dataset (raw)", expanded=False):
    st.caption("For debugging or transparency — not part of the default demo flow.")
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
        ],
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