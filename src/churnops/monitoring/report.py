"""Shareable drift report: one HTML file with charts and per-feature verdicts."""

from __future__ import annotations

import base64
import io
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from churnops.config import REPORTS_DIR, settings  # noqa: E402
from churnops.data.features import NUMERIC_FEATURES  # noqa: E402
from churnops.data.validate import clean_types  # noqa: E402
from churnops.logging_utils import get_logger  # noqa: E402
from churnops.monitoring.drift import detect_drift  # noqa: E402

logger = get_logger(__name__)

COLORS = {"stable": "#2f6f4e", "warn": "#b7791f", "alert": "#b42318"}


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def _psi_chart(features: dict) -> str:
    names = list(features.keys())
    values = [features[name]["psi"] for name in names]
    severities = [features[name]["severity"] for name in names]
    order = sorted(range(len(values)), key=lambda i: values[i])
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    severities = [severities[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.30 * len(names) + 1.1)))
    bars = ax.barh(names, values, height=0.68, color=[COLORS.get(s, "#555") for s in severities])
    ax.axvline(settings.psi_warn, color="#b7791f", linestyle="--", linewidth=1)
    ax.axvline(settings.psi_alert, color="#b42318", linestyle="--", linewidth=1)
    headroom = max(max(values) if values else 0.0, settings.psi_alert)
    label_x = headroom * 1.10
    ax.set_xlim(0, headroom * 1.30)
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#3f4a56",
        )
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Population Stability Index by feature", fontsize=11, loc="left", pad=18)
    ax.text(
        0.0,
        1.02,
        f"dashed lines: warning {settings.psi_warn:.2f} / alert {settings.psi_alert:.2f}",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#6b7280",
    )
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _distribution_chart(
    current: pd.DataFrame, reference: pd.DataFrame | None, feature: str
) -> str | None:
    if reference is None or feature not in reference.columns or feature not in current.columns:
        return None
    current_values = pd.to_numeric(current[feature], errors="coerce").dropna()
    reference_values = pd.to_numeric(reference[feature], errors="coerce").dropna()
    if current_values.empty or reference_values.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 3))
    bins = 30
    ax.hist(reference_values, bins=bins, alpha=0.55, label="reference (training)", density=True)
    ax.hist(current_values, bins=bins, alpha=0.55, label="current batch", density=True)
    ax.set_title(f"{feature}: reference vs current")
    ax.set_xlabel(feature)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _score_chart(scores: pd.Series | None) -> str | None:
    if scores is None or len(scores) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(scores, bins=30, color="#355d9c", alpha=0.85)
    ax.set_title("Predicted churn probability - current batch")
    ax.set_xlabel("score")
    ax.set_ylabel("customers")
    fig.tight_layout()
    return _fig_to_base64(fig)


def generate_drift_report(
    current: pd.DataFrame,
    scores: pd.Series | None = None,
    output_path: Path | None = None,
    reference_sample: pd.DataFrame | None = None,
    reference_stats: dict | None = None,
) -> tuple[Path, dict]:
    """Build the HTML report and return (path, drift_report)."""
    output_path = output_path or REPORTS_DIR / "drift_report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = detect_drift(
        current, reference_stats=reference_stats, reference_sample=reference_sample
    )
    features = report["features"]

    psi_png = _psi_chart(features)
    numeric_charts = []
    for feature in NUMERIC_FEATURES:
        info = features.get(feature, {})
        if info.get("severity") in {"warn", "alert"}:
            chart = _distribution_chart(clean_types(current), reference_sample, feature)
            if chart:
                numeric_charts.append((feature, chart))
    score_png = _score_chart(scores)

    rows = []
    for name, info in sorted(features.items(), key=lambda item: item[1]["psi"], reverse=True):
        ks = info.get("ks_statistic")
        ks_text = f"{ks:.3f}" if isinstance(ks, float) else "-"
        rows.append(
            f"<tr><td>{name}</td><td>{info['type']}</td><td>{info['psi']:.4f}</td>"
            f"<td>{ks_text}</td>"
            f"<td><span class='badge {info['severity']}'>{info['severity']}</span></td></tr>"
        )

    status = report["status"]
    colour = {"stable": "#2f6f4e", "warning": "#b7791f", "significant": "#b42318"}[status]
    detail_blocks = "".join(
        f"<div class='card'><h3>{feature}</h3><img src='data:image/png;base64,{png}'/></div>"
        for feature, png in numeric_charts
    )
    score_block = (
        f"<div class='card'><h3>Score distribution</h3>"
        f"<img src='data:image/png;base64,{score_png}'/></div>"
        if score_png
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>ChurnOps drift report - {report["checked_at"][:10]}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; background:#f6f7f9; color:#1f2933; }}
 .wrap {{ max-width: 1040px; margin: 0 auto; padding: 32px 24px 64px; }}
 h1 {{ font-size: 26px; margin: 0 0 4px; }}
 .sub {{ color:#6b7280; font-size: 14px; margin-bottom: 24px; }}
 .hero {{ background:white; border-radius: 14px; padding: 22px 24px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px; border-left: 6px solid {colour}; }}
 .pill {{ display:inline-block; padding: 4px 12px; border-radius: 999px; color:white; background:{colour}; font-weight:600; font-size: 13px; text-transform: uppercase; letter-spacing:.04em; }}
 .kpis {{ display:flex; gap: 28px; margin-top: 16px; flex-wrap: wrap; }}
 .kpi .v {{ font-size: 22px; font-weight: 700; }}
 .kpi .l {{ font-size: 12px; color:#6b7280; text-transform: uppercase; letter-spacing:.06em; }}
 .card {{ background:white; border-radius: 14px; padding: 18px 20px; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px; }}
 table {{ width:100%; border-collapse: collapse; font-size: 14px; }}
 th {{ text-align:left; padding: 10px 8px; border-bottom: 2px solid #e5e7eb; color:#374151; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
 td {{ padding: 9px 8px; border-bottom: 1px solid #f1f3f5; }}
 .badge {{ padding: 3px 9px; border-radius: 999px; color:white; font-size: 12px; }}
 .badge.stable {{ background:#2f6f4e; }} .badge.warn {{ background:#b7791f; }} .badge.alert {{ background:#b42318; }}
 img {{ max-width: 100%; }}
 .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
</style></head>
<body><div class="wrap">
 <h1>ChurnOps - data &amp; model drift report</h1>
 <div class="sub">Generated {report["checked_at"]} &middot; {report["rows"]:,} scored records &middot; baseline: training reference sample</div>

 <div class="hero">
   <span class="pill">{status}</span>
   <div class="kpis">
     <div class="kpi"><div class="v">{report["drift_share"]:.0%}</div><div class="l">features in alert</div></div>
     <div class="kpi"><div class="v">{len(report["alerting_features"])}</div><div class="l">alerting features</div></div>
     <div class="kpi"><div class="v">{len(report["warning_features"])}</div><div class="l">warning features</div></div>
     <div class="kpi"><div class="v">{len(features)}</div><div class="l">features monitored</div></div>
   </div>
 </div>

 <div class="card"><h3 style="margin-top:0">PSI overview</h3><img src="data:image/png;base64,{psi_png}"/></div>

 <div class="card">
   <h3 style="margin-top:0">Per-feature verdicts</h3>
   <table><thead><tr><th>Feature</th><th>Type</th><th>PSI</th><th>KS statistic</th><th>Severity</th></tr></thead>
   <tbody>{"".join(rows)}</tbody></table>
 </div>

 <div class="grid">{score_block}{detail_blocks}</div>

 <div class="card" style="color:#6b7280; font-size:13px">
   PSI thresholds: &lt; 0.10 stable, 0.10-0.20 warning, &gt; 0.20 alert.
   A batch is <strong>significant</strong> when at least {settings.drift_share_alert:.0%} of features alert.
 </div>
</div></body></html>"""

    output_path.write_text(html)
    (REPORTS_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    logger.info("drift report written", extra={"path": str(output_path), "status": status})
    return output_path, report


def notify(report: dict) -> bool:
    """Send a Slack alert when the webhook is configured."""
    if report["status"] == "stable":
        return False
    if not settings.slack_webhook_url:
        logger.info(
            "drift alert suppressed: no webhook configured", extra={"status": report["status"]}
        )
        return False

    text = (
        f":warning: ChurnOps drift {report['status'].upper()} - "
        f"{len(report['alerting_features'])} features in alert "
        f"({report['drift_share']:.0%} of monitored features). "
        f"Top: {', '.join(report['alerting_features'][:5]) or 'n/a'}"
    )
    payload = json.dumps({"text": text}).encode()
    request = urllib.request.Request(
        settings.slack_webhook_url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            logger.info("slack alert sent", extra={"status": response.status})
            return True
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("slack alert failed", extra={"error": str(exc)})
        return False


def report_timestamp() -> str:
    return datetime.now(UTC).isoformat()
