"""Verdict on one explained forecast and the exam poster (materials "Why XAI is vital": SHAP/LIME verdict posters).

Verdict = how far we can trust the EXPLANATION of a forecast (not whether the forecast was right):
  «Искажена»  — the explanations disagree (SHAP vs LIME FLAGGED) or the forecast sits at the decision threshold,
                where a small change flips the decision → the explanation is fragile;
  «Смещена»   — one feature carries ≥ 30 % of the whole SHAP contribution: the model "looks only at the sun"
                (the course's parable for a biased model). The Random Forest spreads its contributions (the main
                feature carries a median 22 % on the test days), so 30 % ≈ the 96th percentile (notebook 11);
  «Устойчива» — several features contribute, SHAP and LIME agree, the forecast is not at the threshold.
"""
from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd

VERDICTS = ["Устойчива", "Смещена", "Искажена"]
VERDICT_COLORS = {"Устойчива": "#2E8B57", "Смещена": "#E08E0B", "Искажена": "#C0392B"}
DOMINANCE, NEAR = 0.30, 0.05


def suggest_verdict(p: float, thr: float, shap_vec, lime_vec, corr_thr: float = 0.5, diff_thr: float = 0.15):
    """Returns (verdict, reasons, numbers). Vectors are the raw contributions of the 26 features."""
    s, l = np.asarray(shap_vec, float), np.asarray(lime_vec, float)
    sn, ln = s / max(np.abs(s).sum(), 1e-12), l / max(np.abs(l).sum(), 1e-12)
    corr = float(np.corrcoef(sn, ln)[0, 1]) if sn.std() > 0 and ln.std() > 0 else float("nan")
    diff = float(np.abs(sn - ln).max())
    dominance = float(np.abs(s).max() / max(np.abs(s).sum(), 1e-12))
    near = abs(p - thr) < NEAR
    reasons = []
    if (not np.isnan(corr) and corr < corr_thr) or diff > diff_thr:
        reasons.append(f"SHAP и LIME расходятся (корреляция {corr:.2f}, наибольшая разница {diff:.2f})")
    if near:
        reasons.append(f"прогноз у порога: |P − {thr:.2f}| = {abs(p - thr):.3f} < {NEAR}")
    if reasons:
        verdict = "Искажена"
    elif dominance >= DOMINANCE:
        verdict = "Смещена"
        reasons.append(f"один признак даёт {dominance:.0%} всего вклада SHAP")
    else:
        verdict = "Устойчива"
        reasons.append(f"вклад распределён (главный признак — {dominance:.0%}), SHAP и LIME согласны (корреляция {corr:.2f})")
    return verdict, reasons, {"corr": corr, "diff": diff, "dominance": dominance, "near_threshold": near}


def make_poster(title: str, subtitle: str, shap_s: pd.Series, lime_s: pd.Series, verdict: str, reasons: list[str],
                reflection: str, footer: str = "ȘAD · тема №23 «Погода как знак и алгоритм»") -> bytes:
    """A4-landscape PNG: forecast, SHAP and LIME bars side by side, the verdict and the student's reflection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16.5, 11.7), facecolor="#FFFDF5")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.55, 2.4, 1.35], hspace=0.35, wspace=0.55, left=0.2, right=0.97, top=0.95, bottom=0.05)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    ax.text(0, 0.75, title, fontsize=24, weight="bold", transform=ax.transAxes)
    ax.text(0, 0.15, subtitle, fontsize=14, color="#333", transform=ax.transAxes)
    for k, (s, name) in enumerate([(shap_s, "SHAP — точный вклад в P(дождь)"), (lime_s, "LIME — локальная линейная модель")]):
        a = fig.add_subplot(gs[1, k])
        top = s.abs().sort_values().index[-10:]
        a.barh(top, s[top], color=np.where(s[top] > 0, "#C0392B", "#2E8B57"))
        a.axvline(0, color="black", lw=0.8)
        a.set_title(name, fontsize=14)
        a.tick_params(axis="y", labelsize=11)
        a.set_xlabel("вклад: красный — к дождю, зелёный — к сухой погоде", fontsize=10)
    a = fig.add_subplot(gs[2, :]); a.axis("off")
    col = VERDICT_COLORS.get(verdict, "#333")
    a.add_patch(matplotlib.patches.FancyBboxPatch((0.0, 0.05), 0.24, 0.9, boxstyle="round,pad=0.02", fc=col, ec="none",
                                                  transform=a.transAxes))
    a.text(0.12, 0.62, "Вердикт", color="white", fontsize=16, ha="center", transform=a.transAxes)
    a.text(0.12, 0.32, verdict.upper(), color="white", fontsize=26, weight="bold", ha="center", transform=a.transAxes)
    why = "\n".join("• " + textwrap.fill(r, 95) for r in reasons)
    a.text(0.27, 0.95, "Почему:", fontsize=13, weight="bold", va="top", transform=a.transAxes)
    a.text(0.27, 0.78, why, fontsize=11.5, va="top", transform=a.transAxes)
    a.text(0.27, 0.42, "Рефлексия:", fontsize=13, weight="bold", va="top", transform=a.transAxes)
    a.text(0.27, 0.27, textwrap.fill(reflection or "—", 120), fontsize=11.5, va="top", style="italic", transform=a.transAxes)
    fig.text(0.97, 0.012, footer, ha="right", fontsize=10, color="#777")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
