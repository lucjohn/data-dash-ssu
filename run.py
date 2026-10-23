"""Run the whole pipeline. From this folder:

    python run.py validate   test the models on the last twelve months of sales
    python run.py report     score those predictions and draw a figure
    python run.py report "what changed"
                             the same, and also log the scores as a new experiment in report/
    python run.py predict    predict the test homes and write the two submission files
    python run.py check      check the submission files before you hand them in

On Windows, type py instead of python if python is not found.
"""
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import data
import models
from check_submission import check_all

ROOT = Path(__file__).parent
TRACKS = ["off_market", "list_aware"]
# Walk-forward validation, the same way the real test works: the test trains on the
# sales up to May 2026 and predicts the homes sold in the 4 months after (June to
# September). So each fold trains on the sales before a cutoff and predicts the homes
# sold in the HORIZON months from it, 1 to 4 months ahead.
HORIZON = 4
VALIDATION_CUTOFFS = pd.period_range("2025-06", "2026-02", freq="M")[::HORIZON]   # 2025-06, 2025-10, 2026-02
VALIDATION_MONTHS = pd.period_range(VALIDATION_CUTOFFS[0], periods=HORIZON * len(VALIDATION_CUTOFFS), freq="M")
# How validation works, in words. Logged with every experiment: scores from different
# schemes are not comparable, so update this whenever you change validate().
VALIDATION_SCHEME = (f"{HORIZON}-month-ahead walk-forward, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}: "
                     f"{len(VALIDATION_CUTOFFS)} folds, each predicting {HORIZON} months from the sales before them")


def validate():
    listings = data.load("listings")
    results = []
    for cutoff in VALIDATION_CUTOFFS:
        train = listings[listings["date_sold"] < cutoff.start_time]
        sold_month = listings["date_sold"].dt.to_period("M")
        homes = listings[(sold_month >= cutoff) & (sold_month < cutoff + HORIZON)]
        month = homes["date_sold"].dt.to_period("M")
        # 1 = sold in the first month after the cutoff, as June is for the hidden test.
        months_ahead = 12 * (month.dt.year - cutoff.year) + month.dt.month - cutoff.month + 1
        for track in TRACKS:
            model = models.make_model(track).fit(train)
            predicted = model.predict(data.hide_answers(homes, track))
            results.append(pd.DataFrame({
                "ml_num": homes["ml_num"], "track": track, "month": month.astype(str),
                "months_ahead": months_ahead,
                "house_type_name": homes["house_type_name"], "sold_price": homes["sold_price"],
                "predicted_price": np.round(predicted)}))
        print(f"{cutoff} to {cutoff + HORIZON - 1}: trained on {len(train):,} earlier sales, "
              f"predicted {len(homes):,} homes")
    path = ROOT / "outputs" / "validation_predictions.csv"
    path.parent.mkdir(exist_ok=True)
    pd.concat(results).to_csv(path, index=False)
    # Which code made these predictions, for the experiment log (see log_experiment).
    run_info = {"validated_on": date.today().isoformat(), "code_version": code_version(),
                "validation": VALIDATION_SCHEME}
    (ROOT / "outputs" / "validation_run.json").write_text(json.dumps(run_info, indent=2))
    print(f"Wrote {path.relative_to(ROOT)}")


def code_version():
    """The git commit the code is at, plus any Python files changed since, so a logged
    result can be traced back to the code that produced it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout
    try:
        commit = git("rev-parse", "--short", "HEAD").strip()
        changed = [line[3:] for line in git("status", "--porcelain", "--", "*.py").splitlines()]
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return commit + (" + edits to " + ", ".join(changed) if changed else "")


def bootstrap_ci(values, statistic, repeats=1000):
    """95% interval: recompute the statistic on 1,000 resamples of the homes
    (drawn with replacement) and keep the middle 95% of the results."""
    rng = np.random.default_rng(0)
    results = [statistic(rng.choice(values, size=len(values))) for _ in range(repeats)]
    return [round(float(x), 2) for x in np.percentile(results, [2.5, 97.5])]


def metrics(predictions):
    """Scores for one track. Errors are in percent of the sold price."""
    signed = 100 * (predictions["predicted_price"] - predictions["sold_price"]) / predictions["sold_price"]
    absolute = signed.abs().to_numpy()
    return {
        "n": len(predictions),
        "mape": round(float(np.mean(absolute)), 2),          # mean absolute % error
        "mape_95ci": bootstrap_ci(absolute, np.mean),
        "mdape": round(float(np.median(absolute)), 2),       # median absolute % error
        "mdape_95ci": bootstrap_ci(absolute, np.median),
        "median_signed_error": round(float(np.median(signed)), 2),  # above 0: predicts too high
    }


def report(label=None):
    path = ROOT / "outputs" / "validation_predictions.csv"
    if not path.exists():
        sys.exit(f"{path} does not exist yet: run `python run.py validate` first")
    predictions = pd.read_csv(path)
    scores = {track: metrics(predictions[predictions["track"] == track]) for track in TRACKS}

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "validation_report.json").write_text(json.dumps(scores, indent=2))
    lines = [
        "# Validation report", "",
        f"Walk-forward validation, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}: models trained only on "
        f"the sales before {', '.join(map(str, VALIDATION_CUTOFFS))} predicted the {HORIZON} months after each.",
        "Errors are (predicted - sold) / sold. Brackets: bootstrap 95% interval.", "",
        "| Track | Homes | MAPE | MdAPE | Median signed error |", "|---|---|---|---|---|"]
    for track, s in scores.items():
        lines.append(f"| {track} | {s['n']:,} | {s['mape']:.2f}% ({s['mape_95ci'][0]:.2f} to {s['mape_95ci'][1]:.2f}) "
                     f"| {s['mdape']:.2f}% ({s['mdape_95ci'][0]:.2f} to {s['mdape_95ci'][1]:.2f}) "
                     f"| {s['median_signed_error']:+.2f}% |")
    lines += [
        "", "MAPE: mean absolute percentage error. MdAPE: median absolute percentage error.",
        "A median signed error above 0 means the model tends to predict too high.",
        "The intervals treat homes as independent. Homes sold in the same month share the same",
        "market swings, so the real uncertainty is somewhat wider.", "",
        f"MdAPE by months ahead of the training cutoff (the hidden test is 1 to {HORIZON} months ahead):", "",
        "| Months ahead | " + " | ".join(TRACKS) + " |", "|---" * (len(TRACKS) + 1) + "|"]
    absolute = 100 * (predictions["predicted_price"] - predictions["sold_price"]).abs() / predictions["sold_price"]
    by_ahead = absolute.groupby([predictions["months_ahead"], predictions["track"]]).median().unstack()
    for ahead, row in by_ahead.iterrows():
        lines.append(f"| {ahead} | " + " | ".join(f"{row[track]:.2f}%" for track in TRACKS) + " |")
    lines += ["", "![Error by price range](../figures/error_by_price_range.png)", ""]
    (ROOT / "reports" / "validation_report.md").write_text("\n".join(lines))
    write_latex_table(scores, ROOT / "reports" / "results_table.tex")
    error_by_price_range(predictions)
    print("\n".join(lines))
    print("Wrote reports/validation_report.md, reports/validation_report.json, reports/results_table.tex "
          "and figures/error_by_price_range.png")
    if label:
        log_experiment(scores, label)


def write_latex_table(scores, path):
    """The same scores as a LaTeX table, for report/report.tex to include."""
    rows = []
    for track, s in scores.items():
        name = track.replace("_", "-")
        rows.append(f"{name} & {s['n']:,} & {s['mdape']:.2f} ({s['mdape_95ci'][0]:.2f}--{s['mdape_95ci'][1]:.2f}) "
                    f"& {s['mape']:.2f} ({s['mape_95ci'][0]:.2f}--{s['mape_95ci'][1]:.2f}) "
                    f"& {s['median_signed_error']:+.2f} \\\\")
    table = "\n".join([
        "% Written by `python run.py report`. Do not edit by hand: rerun the command instead.",
        "\\begin{tabular}{lrccc}", "\\hline",
        "Track & Homes & MdAPE \\% (95\\% CI) & MAPE \\% (95\\% CI) & Median signed error \\% \\\\", "\\hline",
        *rows, "\\hline", "\\end{tabular}"])
    path.write_text(table + "\n")


def log_experiment(scores, label):
    """Keep this validation run for the report: add a row to report/experiments.csv, keep a
    copy of its results table and figure in report/experiments/, and rewrite
    report/experiments_table.tex from the whole log.

    Every run is kept, the bad ones too: judges want to see how many things you tried
    before the score you report.
    """
    run_info_path = ROOT / "outputs" / "validation_run.json"
    run_info = (json.loads(run_info_path.read_text()) if run_info_path.exists()
                else {"validated_on": "unknown", "code_version": "unknown", "validation": "unknown"})
    log_path = ROOT / "report" / "experiments.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    number = len(log)
    row = {"experiment": number, "validated_on": run_info["validated_on"], "label": label}
    for track, s in scores.items():
        for metric in ["mdape", "mape", "median_signed_error"]:
            row[f"{track}_{metric}"] = s[metric]
    row["code_version"] = run_info["code_version"]
    row["validation"] = run_info["validation"]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)

    folder = ROOT / "report" / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    log.to_csv(log_path, index=False)
    write_latex_table(scores, folder / f"experiment_{number:02d}_results.tex")
    shutil.copy(ROOT / "figures" / "error_by_price_range.png",
                folder / f"experiment_{number:02d}_error_by_price_range.png")
    write_experiments_table(log)
    print(f"Logged experiment {number} in report/experiments.csv and report/experiments/")


def latex_escape(text):
    """Text that LaTeX prints as is, even with characters such as _ % & $ in it."""
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(special.get(c, c) for c in str(text))


def write_experiments_table(log):
    """Every logged experiment as one LaTeX table, for the lab notebook and the report's
    appendix. A new heading row marks each change of validation scheme."""
    rows, scheme = [], None
    for _, e in log.iterrows():
        if e["validation"] != scheme:
            scheme = e["validation"]
            rows.append(f"\\multicolumn{{7}}{{@{{}}p{{14cm}}}}{{\\textit{{Validation: {latex_escape(scheme)}}}}} \\\\")
        rows.append(f"{e['experiment']} & {e['validated_on']} & {latex_escape(e['label'])} "
                    f"& {e['off_market_mdape']:.2f} & {e['off_market_mape']:.2f} "
                    f"& {e['list_aware_mdape']:.2f} & {e['list_aware_mape']:.2f} \\\\")
    table = "\n".join([
        "% Written by `python run.py report \"what changed\"`. Do not edit by hand: the log is report/experiments.csv.",
        "\\begin{tabular}{@{}rlp{6.5cm}cccc@{}}", "\\toprule",
        " & & & \\multicolumn{2}{c}{Off-market} & \\multicolumn{2}{c}{List-aware} \\\\",
        "\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
        "\\# & Date & What changed & MdAPE \\% & MAPE \\% & MdAPE \\% & MAPE \\% \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}"])
    (ROOT / "report" / "experiments_table.tex").write_text(table + "\n")


def error_by_price_range(predictions):
    """Figure: how big the errors are, from the cheapest tenth of homes to the dearest.

    Want more figures? Copy this function and group the errors by something else:
    house_type_name, month, or community (add it to the predictions file in validate()).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"off_market": "#2a78d6", "list_aware": "#eb6834"}
    for track in TRACKS:
        rows = predictions[predictions["track"] == track]
        error = 100 * (rows["predicted_price"] - rows["sold_price"]).abs() / rows["sold_price"]
        decile = pd.qcut(rows["sold_price"], 10, labels=False) + 1   # 1 = cheapest tenth
        by_decile = error.groupby(decile)
        middle = by_decile.median()
        ax.plot(middle.index, middle.values, marker="o", linewidth=2, color=colours[track], label=track)
        ax.fill_between(middle.index, by_decile.quantile(0.25), by_decile.quantile(0.75),
                        color=colours[track], alpha=0.15, linewidth=0)
    prices = rows["sold_price"].groupby(decile).median()
    ax.set_xticks(prices.index, [f"{d}\n${p / 1e6:.2f}M" for d, p in prices.items()])
    ax.set_xlabel("Sold-price decile (1 = cheapest tenth of homes), with its median price")
    ax.set_ylabel("Absolute % error\n(line: median, band: middle 50%)")
    ax.set_title(f"Validation error by price range, {VALIDATION_MONTHS[0]} to {VALIDATION_MONTHS[-1]}")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "error_by_price_range.png", dpi=150)
    plt.close(fig)


def predict():
    listings = data.load("listings")
    (ROOT / "submissions").mkdir(exist_ok=True)
    for track in TRACKS:
        homes = data.load(track)
        model = models.make_model(track).fit(listings)
        predicted = model.predict(homes)
        path = ROOT / "submissions" / f"submission_{track}.csv"
        pd.DataFrame({"ml_num": homes["ml_num"], "predicted_price": np.round(predicted)}).to_csv(path, index=False)
        print(f"Wrote {path.relative_to(ROOT)} ({len(homes):,} homes)")


def check():
    return check_all(ROOT / "submissions")


COMMANDS = {"validate": validate, "report": report, "predict": predict, "check": check}

if __name__ == "__main__":
    args = sys.argv[1:]
    # Only report takes an extra argument: the experiment's label.
    if not args or args[0] not in COMMANDS or len(args) > (2 if args[0] == "report" else 1):
        sys.exit(__doc__)
    result = COMMANDS[args[0]](*args[1:])
    sys.exit(1 if result is False else 0)
