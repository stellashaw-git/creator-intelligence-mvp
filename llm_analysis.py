"""
Creator analysis: OpenAI-backed memo when OPENAI_API_KEY is set; otherwise rule-based fallback.

Rule-based signals (no API):
  - growth_30d “high” → cohort-relative (≥ median or upper tier) → momentum language
  - engagement_rate > 0.1 → strong engagement
  - avg_views / followers high vs cohort → reach efficiency
  - low followers + high growth → early-stage upside
  - weak engagement vs peers or materially low rate → risk callout
  - Final Recommendation (top of memo): Strong Candidate / Watchlist / Not Recommended
    from growth_score & engagement_score (0–100): both >80 → Strong; one high + one medium → Watchlist.
  - Recommended Actions: sign vs monitor, next step, content direction, monetization type (rule-based).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import pandas as pd

# OpenAI (optional)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_ANALYSIS_SECTIONS = """Use exactly these Markdown headings in this order (### with a space after ###):
### Final Recommendation
### Why this creator is promising
### Growth signals
### Risks
### Suggested content strategy
### Monetization potential
### Recommended Actions"""

_SYSTEM_PROMPT = """You are a senior MCN / creator-economy analyst writing an internal partner memo.
Be concise, decision-oriented, and specific to the numbers provided. Output Markdown only—no preamble or sign-off."""


def _row_dict_for_prompt(row: pd.Series) -> dict[str, Any]:
    """Plain JSON-serializable dict for API prompts."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if pd.isna(v):
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float):
            out[str(k)] = round(v, 4)
        elif isinstance(v, (int, bool)):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def _fmt_int(x: float) -> str:
    return f"{x:,.0f}"


def _score_tier(score: float) -> str:
    """High = >80, medium = (40, 80], low = ≤40 (used with Final Recommendation rules)."""
    if score > 80:
        return "high"
    if score > 40:
        return "medium"
    return "low"


def _recommended_actions_section(
    verdict: str,
    niche: str,
    followers: float,
    engagement_weak: bool,
) -> str:
    """
    Short, decision-oriented bullets—no narrative fluff.
    """
    nl = niche.lower()

    # --- Sign vs monitor ---
    if verdict == "Strong Candidate":
        decision = (
            "**Sign (proceed to diligence)** — scores clear the bar; allocate partner time to validate brand safety, "
            "exclusivity, and reporting before papering."
        )
    elif verdict == "Watchlist":
        decision = (
            "**Monitor — do not sign yet** — keep on a weekly scorecard; re-run analysis after a defined window or a clear metric move."
        )
    else:
        decision = "**Do not sign** — deprioritize pipeline; no commercial commitment until scores improve materially."

    # --- Next step ---
    if verdict == "Strong Candidate":
        next_step = (
            "**Direct outreach:** book a discovery call within **10 business days**, send a one-page talent brief, "
            "and align on deliverables + exclusivity before any test spend."
        )
    elif verdict == "Watchlist":
        next_step = (
            "**Test campaign:** run a **single** low-risk pilot (one deliverable or small flat fee), hard cap spend, "
            "with success = engagement + completion—not vanity views."
        )
    else:
        next_step = "**Ignore for now** — no outreach or pilots; only revisit if growth/engagement scores step-change."

    # --- Content direction (one actionable line) ---
    if engagement_weak:
        content_hint = (
            "Force **comment-driven formats** (prompts, polls, reply bait) twice weekly; freeze new niches until reaction rate recovers."
        )
    elif "fitness" in nl or "health" in nl:
        content_hint = (
            "Lock a **repeatable series** (e.g. 30-day challenge + weekly check-in); lead with transformation or proof in frame one."
        )
    elif "beauty" in nl or "makeup" in nl:
        content_hint = (
            "Double down on **GRWM + before/after**; one hero format per week, same hook structure so the algo can retarget fans."
        )
    elif "lifestyle" in nl:
        content_hint = (
            "Anchor **day-in-the-life + routine** arcs; batch film, ship on a fixed cadence to protect completion rate."
        )
    else:
        content_hint = (
            f"Own **{niche}** with a named weekly series + ruthless first-2s hook tests; kill formats that don’t lift saves or comments."
        )

    # --- Monetization type ---
    if verdict == "Not Recommended":
        mon = "None for now—**no** guaranteed CPM packages; if you must engage, gifting-only or rev-share micro-tests only."
    elif verdict == "Watchlist":
        mon = "**Affiliate + gifting + optional micro flat-fee pilot**—no long-term exclusivity until Watchlist clears."
    elif followers >= 100_000:
        mon = "**Managed sponsorships + integrated campaigns** (+ performance bonuses if engagement stays strong)."
    elif followers >= 10_000:
        mon = "**Micro-brand pilots + affiliate stack**; add structured sponsorships once repeat viewers and demographics are proven."
    else:
        mon = "**Gifting + rev-share / light affiliate** until audience crosses ~10k; avoid impression guarantees."

    return (
        f"{decision}\n\n"
        f"{next_step}\n\n"
        f"**Content direction:** {content_hint}\n\n"
        f"**Monetization type:** {mon}"
    )


def generate_rule_based_analysis(row: pd.Series, df: pd.DataFrame) -> str:
    """
    Return full markdown (### headings + body): Final Recommendation first, then analysis sections,
    then Recommended Actions.

    `df` must include scored columns: followers, avg_views, growth_30d,
    engagement_rate, growth_score, engagement_score, final_score, niche, username.
    """
    name = str(row.get("username", "This creator"))
    niche = str(row.get("niche", "general")).strip() or "general"

    followers = float(row["followers"])
    avg_views = float(row["avg_views"])
    growth = float(row["growth_30d"])
    eng = float(row["engagement_rate"])
    final_score = float(row["final_score"])

    med_followers = float(df["followers"].median())
    med_growth = float(df["growth_30d"].median())
    med_eng = float(df["engagement_rate"].median())
    med_final = float(df["final_score"].median())

    g75 = float(df["growth_30d"].quantile(0.75))
    # “High” growth: strong cohort signal (top quartile or at/above median if tiny n)
    growth_high = growth >= max(g75, med_growth)
    growth_above_med = growth >= med_growth

    gs = float(row["growth_score"])
    es = float(row["engagement_score"])
    g_tier = _score_tier(gs)
    e_tier = _score_tier(es)

    # Final Recommendation (transparent score rules)
    if gs > 80 and es > 80:
        verdict = "Strong Candidate"
        verdict_body = (
            f"**growth_score** ({gs:.2f}) and **engagement_score** ({es:.2f}) both exceed **80**—the dual signal we use for "
            "priority outreach and partner investment, subject to niche fit and diligence."
        )
    elif (g_tier == "high" and e_tier == "medium") or (g_tier == "medium" and e_tier == "high"):
        verdict = "Watchlist"
        verdict_body = (
            f"**One high, one medium:** growth_score **{gs:.2f}** ({g_tier}) vs engagement_score **{es:.2f}** ({e_tier}). "
            "Enough to stay on a light watchlist and re-check after the next posting cycle—not a full green light."
        )
    else:
        verdict = "Not Recommended"
        verdict_body = (
            f"Does not meet **Strong Candidate** (both scores >80) or **Watchlist** (exactly one high >80 and one medium 41–80). "
            f"Current pair: growth_score **{gs:.2f}**, engagement_score **{es:.2f}**—default pass unless strategy shifts materially."
        )

    g_label = g_tier
    e_label = e_tier

    strong_engagement = eng > 0.1
    engagement_weak = eng < med_eng or eng < 0.05

    # Reach efficiency: views per follower (typical post reach vs audience)
    denom = max(followers, 1.0)
    reach_ratio = avg_views / denom
    cohort_ratios = df["avg_views"] / df["followers"].clip(lower=1.0)
    med_ratio = float(cohort_ratios.median())
    q75_ratio = float(cohort_ratios.quantile(0.75))
    reach_efficient = reach_ratio >= med_ratio and reach_ratio >= q75_ratio * 0.85
    # User rule: “high” → also flag clearly strong absolute efficiency
    if reach_ratio >= max(med_ratio, 0.15):  # 15%+ of follower base seeing a typical post is a useful bar when cohort is small
        reach_strong = True
    else:
        reach_strong = reach_efficient

    early_stage_high_potential = followers < med_followers and growth_above_med
    beat_pct = (df["final_score"] < final_score).mean() * 100

    # --- Section bodies (analyst voice) ---
    thesis: list[str] = []
    if final_score >= med_final:
        thesis.append(
            f"{name} clears the bar on composite score ({final_score:.2f} vs ~{med_final:.1f} median)—outperforming about "
            f"{beat_pct:.0f}% of creators in this sample on that metric, which is enough to justify a diligence pass if the niche fits your mandate."
        )
    else:
        thesis.append(
            f"Composite score ({final_score:.2f}) is below cohort median (~{med_final:.1f}), so the case is more "
            "turnaround or niche-bet than core momentum—unless strategy or distribution is about to step-change."
        )
    if early_stage_high_potential:
        thesis.append(
            f"**Early-stage skew:** follower base is still modest ({_fmt_int(followers)} vs ~{_fmt_int(med_followers)} median) "
            f"but growth in the window ({_fmt_int(growth)}) punches above that weight—classic high-beta profile if content-market fit holds."
        )
    if strong_engagement:
        thesis.append(
            f"Engagement rate is **strong** at {eng:.3f} (likes + comments per view)—above the 0.10 bar we use for “audience is reacting,” "
            "which de-risks monetization tests sooner than raw reach alone."
        )
    elif not engagement_weak:
        thesis.append(
            f"Engagement is respectable ({eng:.3f} per view) though not yet in the “outlier” band—watch whether hooks improve before sizing deals."
        )
    if reach_strong:
        thesis.append(
            f"**Reach efficiency** looks solid: typical views (~{_fmt_int(avg_views)}) vs {_fmt_int(followers)} followers "
            f"(~{reach_ratio:.2%} of base per post vs cohort norms)—algorithm isn’t hoarding this account in the basement."
        )
    if growth_high:
        thesis.append(
            f"**Momentum:** 30d net adds ({_fmt_int(growth)}) sit in the **strong** band for this set—suggests the feed is still awarding distribution."
        )
    elif growth_above_med:
        thesis.append(
            f"Growth is at least in line with peers ({_fmt_int(growth)} vs ~{_fmt_int(med_growth)} median)—not overheating, but not cold either."
        )

    if not thesis:
        thesis.append(
            f"We’d want one clearer proof point (growth, engagement, or reach) before prioritizing {name} over other {niche} bets in the file."
        )

    growth_lines: list[str] = []
    if growth_high:
        growth_lines.append(
            f"Net adds over 30d (~{_fmt_int(growth)}) imply **strong momentum** relative to this sample—"
            "the kind of slope MCN talent teams flag when allocating short-form resource."
        )
    elif growth_above_med:
        growth_lines.append(
            f"Growth is positive vs median ({_fmt_int(growth)} vs ~{_fmt_int(med_growth)}), so the account isn’t decaying—"
            "next question is whether you can compound it with format iteration."
        )
    else:
        growth_lines.append(
            f"Growth trails the cohort ({_fmt_int(growth)} vs ~{_fmt_int(med_growth)} median). "
            "If you’re underwriting scale, you’ll want a credible plan on posting cadence, hooks, or cross-posting—not just hope."
        )
    if avg_views >= float(df["avg_views"].median()):
        growth_lines.append(
            f"Avg views (~{_fmt_int(avg_views)}) are holding vs peers, which supports the idea that distribution isn’t fully broken."
        )
    else:
        growth_lines.append(
            f"Avg views (~{_fmt_int(avg_views)}) lag the median here—momentum in followers and momentum in impressions need to line up before we call it a clean growth story."
        )

    risks: list[str] = []
    if engagement_weak:
        risks.append(
            f"**Engagement risk:** at {eng:.3f} per view (peer median ~{med_eng:.3f}), interaction is **under-indexing**—"
            "sponsorship CPMs and organic reach can both compress if this doesn’t improve. Worth a hard look at comment depth and repeat viewers, not just vanity metrics."
        )
    if not growth_above_med and growth < med_growth * 0.7:
        risks.append(
            "Growth is soft vs peers; if this is a share-gain narrative, you need a catalyst (series, collab, or trend tie-in) or you’re paying for flat optionality."
        )
    if reach_ratio < med_ratio * 0.8:
        risks.append(
            f"Reach efficiency is weak (views/followers below cohort norm)—the account may be follower-heavy but **under-delivering impressions per post**, which hurts campaign guarantees."
        )
    if not risks:
        risks.append(
            "No single metric screams distress in this coarse pass—still validate retention, audience geography, and brand-safety the way you would before a signed deal."
        )

    # Strategy
    strat_bits = [
        f"Anchor a **{niche}** content thesis: double down on formats that already spike comments/saves, "
        "then run disciplined experiments (one variable at a time—hook, length, posting time).",
    ]
    if engagement_weak:
        strat_bits.append(
            "Near-term: prioritize **reply velocity** and prompts that force comments; cheap tests beat another generic upload schedule."
        )
    if not reach_strong:
        strat_bits.append(
            "If reach efficiency is the gap, lean into **first-frame clarity** and repeatable series titles so the algo can learn what to push."
        )
    if growth_high:
        strat_bits.append(
            "While momentum is hot, capture **proof assets** (best clips, case-style posts) for outbound brand conversations—strike before the curve flattens."
        )

    # Monetization
    if followers >= 100_000:
        mon_tier = (
            "Audience size supports **standard managed-service deals**: structured sponsorships, integrated campaigns, and performance uplifts "
            "if reporting is clean."
        )
    elif followers >= 10_000:
        mon_tier = (
            "**Micro / rising tier:** affiliate, gifting, and pilot CPMs are realistic; package engagement proof "
            "(your strong rate helps) to justify test budgets."
        )
    else:
        mon_tier = (
            "**Early monetization** is mostly tests and rev-share—fine if you’re buying option value, but price deals against engagement and growth, not follower count alone."
        )

    if strong_engagement and not engagement_weak:
        mon_extra = (
            " Strong engagement gives you leverage for performance-tied or bonus-heavy structures—brands pay for outcomes, not labels."
        )
    elif engagement_weak:
        mon_extra = (
            " Until engagement firms up, we’d keep **deal sizes small** and favor inventory that doesn’t require guaranteed impressions."
        )
    else:
        mon_extra = " Position any pitch around **efficiency and audience quality**, not raw reach."

    final_recommendation = (
        f"**{verdict}**\n\n"
        f"{verdict_body}\n\n"
        f"*Scores: growth **{gs:.2f}** ({g_label}) · engagement **{es:.2f}** ({e_label}). "
        f"Strong Candidate = both >80; Watchlist = one score >80 and the other medium (41–80); else Not Recommended.*"
    )

    recommended_actions = _recommended_actions_section(
        verdict=verdict,
        niche=niche,
        followers=followers,
        engagement_weak=engagement_weak,
    )

    blocks = [
        ("### Final Recommendation", final_recommendation),
        ("### Why this creator is promising", "\n\n".join(thesis)),
        ("### Growth signals", "\n\n".join(growth_lines)),
        ("### Risks", " ".join(risks)),
        ("### Suggested content strategy", " ".join(strat_bits)),
        ("### Monetization potential", mon_tier + mon_extra),
        ("### Recommended Actions", recommended_actions),
    ]

    out: list[str] = []
    for heading, body in blocks:
        out.append(f"{heading}\n\n{body}")
    return "\n\n".join(out)


def generate_openai_analysis(
    row: pd.Series, df: pd.DataFrame
) -> tuple[str, Literal["openai", "rule_based"]]:
    """
    Return `(markdown, mode)` with the same seven-section memo as `generate_rule_based_analysis`.

    If `OPENAI_API_KEY` is set, call OpenAI and return mode `"openai"` on success; on missing key,
    empty response, or any API error, fall back to `generate_rule_based_analysis` and mode `"rule_based"`.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return generate_rule_based_analysis(row, df), "rule_based"

    creator = _row_dict_for_prompt(row)
    # Compact cohort context so the model can compare (not just the selected row).
    cohort_cols = [
        c
        for c in (
            "username",
            "niche",
            "followers",
            "avg_views",
            "avg_likes",
            "avg_comments",
            "growth_30d",
            "engagement_rate",
            "reach_score",
            "engagement_score",
            "growth_score",
            "final_score",
            "rank",
        )
        if c in df.columns
    ]
    cohort_json = json.dumps(df[cohort_cols].to_dict(orient="records"), indent=2, default=str)

    user_prompt = f"""Analyze this creator for an MCN/agency partner.

**Selected creator (JSON):**
```json
{json.dumps(creator, indent=2)}
```

**Full cohort in this file (JSON array, for peer comparison):**
```json
{cohort_json}
```

{_ANALYSIS_SECTIONS}

Under **Final Recommendation**, state Strong Candidate / Watchlist / Not Recommended using the same logic as:
both growth_score and engagement_score > 80 → Strong Candidate; exactly one of those > 80 and the other in (40, 80] → Watchlist; else Not Recommended.

Keep each section tight (roughly 2–6 sentences total per section unless Risks needs an extra bullet)."""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=2500,
        )
        text = resp.choices[0].message.content
        if not text or not str(text).strip():
            return generate_rule_based_analysis(row, df), "rule_based"
        return str(text).strip(), "openai"
    except Exception:
        return generate_rule_based_analysis(row, df), "rule_based"
