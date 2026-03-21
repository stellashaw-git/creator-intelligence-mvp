"""
Streamlit entrypoint: load creators from data.csv, score, rank, and explore.

Run: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from llm_analysis import generate_openai_analysis
from scoring import add_creator_scores, rank_explanation_bullets

# Page config (title in browser tab + layout).
st.set_page_config(page_title="AI Creator Intelligence MVP", layout="wide")

st.title("AI Creator Intelligence MVP")
st.markdown(
    "Identify high-potential TikTok creators and generate decision-ready operating insights "
    "for MCN and agency teams."
)

# --- Load data ---
data_path = "data.csv"
try:
    df_raw = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Missing `{data_path}`. Place your CSV next to `app.py`.")
    st.stop()

# Basic validation so errors are obvious in the UI.
required = {"username", "followers", "avg_views", "avg_likes", "avg_comments", "growth_30d"}
missing = required - set(df_raw.columns)
if missing:
    st.error(f"CSV is missing columns: {sorted(missing)}")
    st.stop()

# --- Score and rank ---
df = add_creator_scores(df_raw)

# --- Ranked table ---
st.subheader("Ranked creators")
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

st.divider()

# --- Creator selection (order matches rank) ---
usernames = df["username"].tolist()
selected = st.selectbox(
    "Select a creator",
    options=usernames,
    index=0,
    help="Choices follow rank order (best score first).",
)
row = df[df["username"] == selected].iloc[0]

# --- Profile ---
st.subheader("Creator profile")
st.markdown(f"### {row['username']}")
st.caption(f"Niche: **{row.get('niche', '—')}**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Followers", f"{row['followers']:,.0f}")
m2.metric("Avg views", f"{row['avg_views']:,.0f}")
m3.metric("Avg likes", f"{row['avg_likes']:,.0f}")
m4.metric("Avg comments", f"{row['avg_comments']:,.0f}")

m5, _, _, _ = st.columns(4)
m5.metric("Growth (30d)", f"{row['growth_30d']:,.0f}")

st.markdown("##### Scores (0–100)")
st.caption(
    "Reach = avg of normalized followers & avg views. Engagement = normalized engagement rate. "
    "Growth = normalized 30d growth. Final = average of the three."
)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Reach score", f"{row['reach_score']:.2f}")
s2.metric("Engagement score", f"{row['engagement_score']:.2f}")
s3.metric("Growth score", f"{row['growth_score']:.2f}")
s4.metric("Final score", f"{row['final_score']:.2f}")

# How this rank lines up with raw signals (rank itself comes from final_score).
rank_why = rank_explanation_bullets(row, df)
st.markdown(f"#### Why this creator ranks #{int(row['rank'])}")
for line in rank_why:
    st.markdown(f"- {line}")
if not rank_why:
    st.caption(
        "No bullet matched (growth not #1 in file, engagement at/below median, or reach efficiency below median). "
        "Rank still reflects the blended score above."
    )

st.divider()

# --- AI analysis: OpenAI when API key is set; otherwise rule-based memo ---
st.subheader("AI Analysis")

analysis_md, analysis_mode = generate_openai_analysis(row, df)
if analysis_mode == "openai":
    st.caption("Mode: OpenAI")
else:
    st.caption("Mode: Rule-based fallback")

st.markdown(analysis_md)
