"""
Creator scoring: component scores on 0–100, then a final score.

All scores use min–max normalization *within the loaded CSV* (same file = same scale).

Transparent formulas (n_x = min–max normalized column x in [0, 1]):
  - reach_score     = 100 * (n_followers + n_avg_views) / 2
  - engagement_score = 100 * n_engagement_rate
  - growth_score    = 100 * n_growth_30d
  - final_score     = (reach_score + engagement_score + growth_score) / 3

 engagement_rate = (avg_likes + avg_comments) / avg_views (safe for zero views).

Rank uses final_score (higher is better).
"""

from __future__ import annotations

import pandas as pd


def _min_max(series: pd.Series) -> pd.Series:
    """Scale values to 0–1; constant column becomes 0.5."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _round_scores(s: pd.Series) -> pd.Series:
    return s.round(2)


def engagement_rate(df: pd.DataFrame) -> pd.Series:
    """
    Rough engagement: (likes + comments) per view.
    Avoids divide-by-zero when avg_views is missing or zero.
    """
    views = df["avg_views"].replace(0, pd.NA).fillna(1)
    return (df["avg_likes"] + df["avg_comments"]) / views


def add_creator_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engagement_rate, reach_score, engagement_score, growth_score,
    final_score (all 0–100, 2 decimals), and rank (1 = best by final_score).

    Expects columns: followers, avg_views, avg_likes, avg_comments, growth_30d.
    """
    out = df.copy()
    out["engagement_rate"] = engagement_rate(out)

    n_followers = _min_max(out["followers"])
    n_views = _min_max(out["avg_views"])
    n_eng = _min_max(out["engagement_rate"])
    n_growth = _min_max(out["growth_30d"])

    out["reach_score"] = _round_scores(100 * (n_followers + n_views) / 2)
    out["engagement_score"] = _round_scores(100 * n_eng)
    out["growth_score"] = _round_scores(100 * n_growth)
    out["final_score"] = _round_scores(
        (out["reach_score"] + out["engagement_score"] + out["growth_score"]) / 3
    )

    out["rank"] = out["final_score"].rank(ascending=False, method="min").astype(int)

    return out.sort_values("rank", ascending=True).reset_index(drop=True)


def rank_explanation_bullets(row: pd.Series, df: pd.DataFrame) -> list[str]:
    """
    Short reasons this creator sits at their rank, using the same cohort as scoring.

    Phrases match product copy:
      - Top growth — 30d growth is highest in the file (ties allowed)
      - Above-average engagement — reaction rate above cohort median
      - Strong reach efficiency — views per follower at or above cohort median
    """
    bullets: list[str] = []
    g = float(row["growth_30d"])
    if g >= float(df["growth_30d"].max()):
        bullets.append("Top growth")

    eng = float(row["engagement_rate"])
    if eng > float(df["engagement_rate"].median()):
        bullets.append("Above-average engagement")

    ratio = float(row["avg_views"]) / max(float(row["followers"]), 1.0)
    cohort_ratio = df["avg_views"] / df["followers"].clip(lower=1.0)
    if ratio >= float(cohort_ratio.median()):
        bullets.append("Strong reach efficiency")

    return bullets


def decision_summary(row: pd.Series, df: pd.DataFrame) -> dict[str, str]:
    """
    Three-line decision outputs for the demo UI (uses engagement_rate, views/followers, growth_score).

    Returns keys: monetization_verdict, traffic_monetization_gap, recommended_action.
    """
    eng = float(row["engagement_rate"])
    ratio = float(row["avg_views"]) / max(float(row["followers"]), 1.0)
    gs = float(row["growth_score"])

    med_eng = float(df["engagement_rate"].median())
    q75_eng = float(df["engagement_rate"].quantile(0.75))
    cohort_ratio = df["avg_views"] / df["followers"].clip(lower=1.0)
    med_ratio = float(cohort_ratio.median())

    high_traffic = ratio >= med_ratio
    weak_engagement = eng < med_eng
    strong_engagement = eng > 0.1 or eng >= q75_eng

    # 1. Monetization verdict
    if high_traffic and weak_engagement:
        monetization_verdict = "Low (High traffic but weak monetization)"
    elif strong_engagement:
        monetization_verdict = "High Monetization Potential"
    else:
        monetization_verdict = "Medium"

    # 2. Traffic vs monetization gap
    if strong_engagement:
        traffic_gap = "Strong monetization (engagement supports conversion)"
    elif high_traffic and weak_engagement:
        traffic_gap = "High traffic but weak engagement (likely poor conversion)"
    else:
        traffic_gap = "Balanced"

    # 3. Recommended action (growth_score + verdict)
    if monetization_verdict.startswith("Low"):
        recommended_action = "Pass"
    elif monetization_verdict == "High Monetization Potential" and gs > 70:
        recommended_action = "Sign (proceed)"
    elif monetization_verdict == "High Monetization Potential" and gs > 40:
        recommended_action = "Pilot test"
    elif monetization_verdict == "High Monetization Potential":
        recommended_action = "Monitor"
    elif monetization_verdict == "Medium" and gs > 55:
        recommended_action = "Pilot test"
    elif monetization_verdict == "Medium":
        recommended_action = "Monitor"
    else:
        recommended_action = "Pass"

    return {
        "monetization_verdict": monetization_verdict,
        "traffic_monetization_gap": traffic_gap,
        "recommended_action": recommended_action,
    }
