#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import io
import json
import math
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SEED = 2401
N_PERM = 10_000
N_BOOT = 10_000
BARRIERS = (0.25, 0.50, 1.00)
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_date(url: str, new_date: str) -> str:
    replaced = re.sub(r"\d{4}-\d{2}-\d{2}(?=\.zip$)", new_date, url)
    if replaced == url:
        raise ValueError(f"Cannot replace date in URL: {url}")
    return replaced


def fetch_blob(task: dict[str, str]) -> tuple[dict[str, str], bytes | None, str]:
    session = requests.Session()
    session.headers["User-Agent"] = "SBC-GANN-R24A-research/0.1"
    last = ""
    for attempt in range(6):
        try:
            response = session.get(task["url"], timeout=90)
            if response.status_code == 200:
                return task, response.content, ""
            last = f"HTTP {response.status_code}"
        except Exception as exc:  # pragma: no cover - network dependent
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(min(15, 2 ** attempt))
    return task, None, last


def parse_kline(blob: bytes, source: str, day: str) -> tuple[pd.DataFrame, str, int]:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise RuntimeError(f"{source} {day}: archive members={members}")
        raw = archive.read(members[0])
    frame = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLS)
    frame["open_time"] = pd.to_numeric(frame["open_time"], errors="coerce")
    frame = frame[frame["open_time"].notna()].copy()
    for column in ["open", "high", "low", "close", "volume", "taker_buy_base"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    unit = "microseconds" if float(frame["open_time"].median()) > 1e15 else "milliseconds"
    if unit == "microseconds":
        frame["open_time"] = (frame["open_time"] // 1000).astype("int64")
    else:
        frame["open_time"] = frame["open_time"].astype("int64")
    duplicate_rows = int(frame["open_time"].duplicated(keep=False).sum())
    frame = frame.sort_values("open_time").drop_duplicates("open_time", keep=False)
    if duplicate_rows:
        raise RuntimeError(f"{source} {day}: duplicate timestamp rows={duplicate_rows}")
    return frame, unit, len(raw)


def expected_times(start_ms: int) -> np.ndarray:
    return start_ms + np.arange(144, dtype=np.int64) * 300_000


def topology_label(t_aligned: float, t_counter: float) -> str:
    aligned = np.isfinite(t_aligned)
    counter = np.isfinite(t_counter)
    if not aligned and not counter:
        return "UNRESOLVED"
    if aligned and counter and t_aligned == t_counter:
        return "SIMULTANEOUS"
    if aligned and not counter:
        return "ALIGNED_ONLY"
    if counter and not aligned:
        return "COUNTER_ONLY"
    if t_aligned < t_counter:
        return "ALIGNED_THEN_COUNTER"
    return "COUNTER_THEN_ALIGNED"


def first_passage(values: np.ndarray, barrier: float) -> float:
    hits = np.flatnonzero(values >= barrier)
    return float(5 * (hits[0] + 1)) if len(hits) else float("nan")


def entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(float)
    return float(-(probabilities * np.log(probabilities)).sum()) if len(probabilities) else float("nan")


def normalized_mutual_information(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y).to_numpy(float)
    n = table.sum()
    if n <= 0:
        return float("nan")
    pxy = table / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = pxy > 0
    mi = float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx <= 0 or hy <= 0:
        return 0.0
    return float(mi / math.sqrt(hx * hy))


def permute_routes_within_year(frame: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    routes = frame["route"].to_numpy(object).copy()
    for indices in frame.groupby("year").groups.values():
        idx = np.asarray(list(indices), dtype=int)
        routes[idx] = rng.permutation(routes[idx])
    return pd.Series(routes, index=frame.index)


def bootstrap_calendar_blocks(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year, group in frame.groupby("year", sort=True):
        months = sorted(group["cluster_month"].unique())
        chosen = rng.choice(months, size=len(months), replace=True)
        for replicate_index, month in enumerate(chosen):
            block = group[group["cluster_month"] == month].copy()
            block["_boot_block"] = f"{year}-{replicate_index}"
            parts.append(block)
    return pd.concat(parts, ignore_index=True)


def safe_median(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="r24a/input_events_202.csv")
    parser.add_argument("--output", default="r24a/output")
    parser.add_argument("--prereg", default="r24a/00_PREREGISTRATION.json")
    parser.add_argument("--scope", default="r24a/00_SCOPE.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(input_path)
    if len(events) != 202 or not events["event_id"].is_unique:
        raise RuntimeError("R24A frozen input must contain 202 unique events")
    events["path_start"] = pd.to_datetime(events["path_start"], utc=True)

    # Build unique archive tasks. Event-day hashes are frozen in the input; next-day hashes are recorded after download.
    task_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in events.itertuples(index=False):
        event_day = row.path_start.strftime("%Y-%m-%d")
        next_day = (row.path_start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for source in ("spot", "perp"):
            event_url = getattr(row, f"{source}_url")
            event_hash = str(getattr(row, f"{source}_sha256")).lower()
            event_task = {"source": source, "day": event_day, "url": event_url, "expected_sha256": event_hash}
            key = (source, event_day)
            if key in task_map and task_map[key] != event_task:
                raise RuntimeError(f"Conflicting event-day task {key}")
            task_map[key] = event_task
            next_task = {"source": source, "day": next_day, "url": replace_date(event_url, next_day), "expected_sha256": ""}
            next_key = (source, next_day)
            if next_key in task_map and task_map[next_key]["url"] != next_task["url"]:
                raise RuntimeError(f"Conflicting next-day task {next_key}")
            task_map.setdefault(next_key, next_task)

    archive_frames: dict[tuple[str, str], pd.DataFrame] = {}
    archive_records: list[dict[str, object]] = []
    with futures.ThreadPoolExecutor(max_workers=12) as executor:
        jobs = [executor.submit(fetch_blob, task) for task in task_map.values()]
        for job in futures.as_completed(jobs):
            task, blob, fetch_error = job.result()
            record: dict[str, object] = {
                **task,
                "download_ok": blob is not None,
                "parse_ok": False,
                "actual_sha256": "",
                "hash_match": "",
                "rows": 0,
                "timestamp_unit": "",
                "error": fetch_error,
            }
            if blob is not None:
                try:
                    actual_hash = sha256_bytes(blob)
                    expected_hash = task["expected_sha256"]
                    record["actual_sha256"] = actual_hash
                    record["hash_match"] = actual_hash.lower() == expected_hash.lower() if expected_hash else "NOT_PREDECLARED"
                    if expected_hash and record["hash_match"] is not True:
                        raise RuntimeError("event-day SHA-256 mismatch")
                    frame, unit, _raw_bytes = parse_kline(blob, task["source"], task["day"])
                    record["parse_ok"] = True
                    record["rows"] = int(len(frame))
                    record["timestamp_unit"] = unit
                    archive_frames[(task["source"], task["day"])] = frame
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
            archive_records.append(record)

    archive_audit = pd.DataFrame(archive_records).sort_values(["source", "day"])
    archive_audit.to_csv(output_dir / "01_ARCHIVE_INTEGRITY.csv", index=False)
    fatal_archives = archive_audit[(~archive_audit["download_ok"]) | (~archive_audit["parse_ok"])]
    if len(fatal_archives):
        result = {
            "protocol_id": "R24A_DYNAMIC_MULTIPROCESS_TRANSITION_ATLAS_v0_1",
            "verdict": "MULTISOURCE_PATH_DATA_INSUFFICIENT",
            "archive_failures": int(len(fatal_archives)),
            "statistics_executed": False,
            "multiprocess_guardrail": "No process is validated or falsified by a data-availability failure."
        }
        (output_dir / "10_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(f"Archive failures: {len(fatal_archives)}")

    panel_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    eligibility_records: list[dict[str, object]] = []

    for row in events.itertuples(index=False):
        event_day = row.path_start.strftime("%Y-%m-%d")
        next_day = (row.path_start + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_ms = int(row.path_start.timestamp() * 1000)
        expected = expected_times(start_ms)
        paths: dict[str, pd.DataFrame] = {}
        source_missing: dict[str, list[int]] = {}
        for source in ("spot", "perp"):
            combined = (
                pd.concat([archive_frames[(source, event_day)], archive_frames[(source, next_day)]], ignore_index=True)
                .sort_values("open_time")
                .drop_duplicates("open_time", keep=False)
            )
            path = combined[combined["open_time"].isin(expected)].sort_values("open_time")
            got = path["open_time"].to_numpy(np.int64)
            missing = sorted(set(int(v) for v in expected) - set(int(v) for v in got))
            source_missing[source] = missing
            paths[source] = path
        eligible = all(len(source_missing[source]) == 0 and len(paths[source]) == 144 for source in ("spot", "perp"))
        eligibility_records.append({
            "event_id": row.event_id,
            "year": int(row.year),
            "eligible": eligible,
            "spot_rows": int(len(paths["spot"])),
            "perp_rows": int(len(paths["perp"])),
            "spot_missing_count": int(len(source_missing["spot"])),
            "perp_missing_count": int(len(source_missing["perp"])),
            "spot_first_missing_utc": pd.Timestamp(source_missing["spot"][0], unit="ms", tz="UTC").isoformat() if source_missing["spot"] else "",
            "perp_first_missing_utc": pd.Timestamp(source_missing["perp"][0], unit="ms", tz="UTC").isoformat() if source_missing["perp"] else "",
        })
        if not eligible:
            continue

        spot = paths["spot"].reset_index(drop=True)
        perp = paths["perp"].reset_index(drop=True)
        if not np.array_equal(spot["open_time"].to_numpy(np.int64), perp["open_time"].to_numpy(np.int64)):
            raise RuntimeError(f"Timestamp mismatch spot/perp for {row.event_id}")

        sign = float(row.event_sign)
        scale = float(row.event_scale)
        if scale <= 0:
            raise RuntimeError(f"Invalid event scale for {row.event_id}")
        spot0 = float(spot.iloc[0]["open"])
        perp0 = float(perp.iloc[0]["open"])
        spot_close_log = np.log(spot["close"].to_numpy(float) / spot0)
        perp_close_log = np.log(perp["close"].to_numpy(float) / perp0)
        if sign > 0:
            aligned_exc = np.maximum(0.0, np.log(spot["high"].to_numpy(float) / spot0) / scale)
            counter_exc = np.maximum(0.0, -np.log(spot["low"].to_numpy(float) / spot0) / scale)
        else:
            aligned_exc = np.maximum(0.0, -np.log(spot["low"].to_numpy(float) / spot0) / scale)
            counter_exc = np.maximum(0.0, np.log(spot["high"].to_numpy(float) / spot0) / scale)
        d_spot = sign * spot_close_log / scale
        d_perp = sign * perp_close_log / scale

        spot_volume = spot["volume"].to_numpy(float)
        perp_volume = perp["volume"].to_numpy(float)
        spot_delta = 2.0 * spot["taker_buy_base"].to_numpy(float) - spot_volume
        perp_delta = 2.0 * perp["taker_buy_base"].to_numpy(float) - perp_volume
        cum_spot_flow = sign * np.cumsum(spot_delta) / (np.cumsum(spot_volume) + 1e-12)
        cum_perp_flow = sign * np.cumsum(perp_delta) / (np.cumsum(perp_volume) + 1e-12)
        flow_lead = cum_spot_flow - cum_perp_flow
        price_divergence = d_perp - d_spot
        basis0 = math.log(perp0 / spot0)
        basis_delta = np.log(perp["close"].to_numpy(float) / spot["close"].to_numpy(float)) - basis0

        for index in range(144):
            panel_records.append({
                "event_id": row.event_id,
                "year": int(row.year),
                "event_sign": sign,
                "route": row.route,
                "cluster_month": row.cluster_month,
                "tau_min": 5 * (index + 1),
                "open_time": pd.Timestamp(int(spot.iloc[index]["open_time"]), unit="ms", tz="UTC").isoformat(),
                "d_spot": float(d_spot[index]),
                "d_perp": float(d_perp[index]),
                "aligned_excursion": float(aligned_exc[index]),
                "counter_excursion": float(counter_exc[index]),
                "cum_spot_flow": float(cum_spot_flow[index]),
                "cum_perp_flow": float(cum_perp_flow[index]),
                "flow_lead_spot_minus_perp": float(flow_lead[index]),
                "price_divergence_perp_minus_spot": float(price_divergence[index]),
                "basis_delta": float(basis_delta[index]),
            })

        transition: dict[str, object] = {
            "event_id": row.event_id,
            "year": int(row.year),
            "event_sign": sign,
            "route": row.route,
            "cluster_month": row.cluster_month,
            "event_scale": scale,
        }
        for barrier in BARRIERS:
            suffix = f"{barrier:.2f}"
            t_aligned = first_passage(aligned_exc, barrier)
            t_counter = first_passage(counter_exc, barrier)
            label = topology_label(t_aligned, t_counter)
            transition[f"t_aligned_{suffix}"] = t_aligned
            transition[f"t_counter_{suffix}"] = t_counter
            transition[f"topology_{suffix}"] = label

        t_a = transition["t_aligned_0.25"]
        t_c = transition["t_counter_0.25"]
        finite_times = [value for value in (t_a, t_c) if np.isfinite(value)]
        decision_time = int(min(finite_times)) if finite_times else 720
        decision_index = max(0, min(143, decision_time // 5 - 1))
        transition.update({
            "decision_time_0.25_min": decision_time,
            "decision_d_spot": float(d_spot[decision_index]),
            "decision_d_perp": float(d_perp[decision_index]),
            "decision_cum_spot_flow": float(cum_spot_flow[decision_index]),
            "decision_cum_perp_flow": float(cum_perp_flow[decision_index]),
            "decision_flow_lead": float(flow_lead[decision_index]),
            "decision_price_divergence": float(price_divergence[decision_index]),
            "decision_basis_delta": float(basis_delta[decision_index]),
            "terminal_d_spot_h1": float(d_spot[11]),
            "terminal_d_spot_h3": float(d_spot[35]),
            "terminal_d_spot_h6": float(d_spot[71]),
            "terminal_d_spot_h12": float(d_spot[143]),
        })
        transition_records.append(transition)

    eligibility = pd.DataFrame(eligibility_records).sort_values("event_id")
    eligibility.to_csv(output_dir / "02_EVENT_ELIGIBILITY.csv", index=False)
    panel = pd.DataFrame(panel_records)
    transitions = pd.DataFrame(transition_records).sort_values("event_id").reset_index(drop=True)
    panel.to_csv(output_dir / "03_DYNAMIC_STATE_PANEL.csv", index=False, float_format="%.17g")
    transitions.to_csv(output_dir / "04_TRANSITION_LEDGER.csv", index=False, float_format="%.17g")

    if len(transitions) == 0:
        raise RuntimeError("No eligible multisource events")

    primary_topology = "topology_0.25"
    route_matrix = pd.crosstab(transitions[primary_topology], transitions["route"], margins=True)
    route_matrix.to_csv(output_dir / "05_TOPOLOGY_ROUTE_MATRIX.csv")

    rng = np.random.default_rng(SEED)
    observed_nmi = normalized_mutual_information(transitions[primary_topology], transitions["route"])
    permuted = np.empty(N_PERM)
    for index in range(N_PERM):
        permuted[index] = normalized_mutual_information(primary_topology and transitions[primary_topology], permute_routes_within_year(transitions, rng))
    permutation_p = float((1 + np.sum(permuted >= observed_nmi)) / (N_PERM + 1))

    bootstrapped: list[float] = []
    for _ in range(N_BOOT):
        sample = bootstrap_calendar_blocks(transitions, rng)
        value = normalized_mutual_information(sample[primary_topology], sample["route"])
        if np.isfinite(value):
            bootstrapped.append(value)
    ci_low, ci_high = np.quantile(bootstrapped, [0.025, 0.975])

    topology_counts = transitions[primary_topology].value_counts()
    class_audit_rows = []
    for topology, count in topology_counts.items():
        subset = transitions[transitions[primary_topology] == topology]
        years_present = int(subset["year"].nunique())
        polarities_present = int(subset["event_sign"].nunique())
        qualifies = bool(count >= 15 and years_present >= 4 and polarities_present == 2)
        class_audit_rows.append({
            "topology": topology,
            "n": int(count),
            "years_present": years_present,
            "polarities_present": polarities_present,
            "nontrivial_stability_gate": qualifies,
        })
    class_audit = pd.DataFrame(class_audit_rows).sort_values("n", ascending=False)
    class_audit.to_csv(output_dir / "06_TOPOLOGY_STABILITY_AUDIT.csv", index=False)
    stable_classes = int(class_audit["nontrivial_stability_gate"].sum())
    structural_gate = bool(permutation_p < 0.05 and ci_low > 0 and stable_classes >= 3)

    # Route entropy and concentration within topology.
    route_profiles = []
    for topology, group in transitions.groupby(primary_topology):
        counts = group["route"].value_counts()
        route_profiles.append({
            "topology": topology,
            "n": int(len(group)),
            "route_entropy_nats": entropy(group["route"]),
            "dominant_route": counts.index[0],
            "dominant_route_share": float(counts.iloc[0] / len(group)),
        })
    pd.DataFrame(route_profiles).sort_values("n", ascending=False).to_csv(output_dir / "07_ROUTE_PROFILE_BY_TOPOLOGY.csv", index=False)

    # Year/polarity distributions.
    yearly = transitions.groupby(["year", primary_topology]).size().rename("n").reset_index()
    yearly.to_csv(output_dir / "08_YEARLY_TOPOLOGY_COUNTS.csv", index=False)
    polarity = transitions.groupby(["event_sign", primary_topology]).size().rename("n").reset_index()
    polarity.to_csv(output_dir / "09_POLARITY_TOPOLOGY_COUNTS.csv", index=False)

    signature_columns = [
        "decision_time_0.25_min", "decision_d_spot", "decision_d_perp",
        "decision_cum_spot_flow", "decision_cum_perp_flow", "decision_flow_lead",
        "decision_price_divergence", "decision_basis_delta",
        "terminal_d_spot_h1", "terminal_d_spot_h3", "terminal_d_spot_h6", "terminal_d_spot_h12",
    ]
    signatures = []
    for topology, group in transitions.groupby(primary_topology):
        record = {"topology": topology, "n": int(len(group))}
        for column in signature_columns:
            record[f"median_{column}"] = safe_median(group[column])
        signatures.append(record)
    pd.DataFrame(signatures).sort_values("n", ascending=False).to_csv(output_dir / "10_PROCESS_SIGNATURES.csv", index=False, float_format="%.12g")

    first_passage_rows = []
    for barrier in BARRIERS:
        suffix = f"{barrier:.2f}"
        for topology, group in transitions.groupby(f"topology_{suffix}"):
            first_passage_rows.append({
                "barrier": barrier,
                "topology": topology,
                "n": int(len(group)),
                "median_t_aligned_min": safe_median(group[f"t_aligned_{suffix}"]),
                "median_t_counter_min": safe_median(group[f"t_counter_{suffix}"]),
            })
    pd.DataFrame(first_passage_rows).to_csv(output_dir / "11_FIRST_PASSAGE_ATLAS.csv", index=False, float_format="%.12g")

    result = {
        "protocol_id": "R24A_DYNAMIC_MULTIPROCESS_TRANSITION_ATLAS_v0_1",
        "starting_events": 202,
        "eligible_events": int(len(transitions)),
        "ineligible_events": int(202 - len(transitions)),
        "primary_barrier": 0.25,
        "observed_nmi": observed_nmi,
        "bootstrap_ci95": [float(ci_low), float(ci_high)],
        "permutation_p_one_sided": permutation_p,
        "stable_nontrivial_topology_classes": stable_classes,
        "structural_gate_pass": structural_gate,
        "verdict": "DYNAMIC_TOPOLOGY_STRUCTURALLY_ASSOCIATED_WITH_ROUTE" if structural_gate else "DYNAMIC_TOPOLOGY_DESCRIPTIVE_ONLY",
        "prediction_authorized": False,
        "trading_authorized": False,
        "multiprocess_guardrail": "Topology is a phenomenological coordinate, not a dominant route or a causal proof."
    }
    (output_dir / "12_RESULT.json").write_text(json.dumps(result, indent=2) + "\n")

    report = f"""# R24A — Atlas dynamique multi-processus\n\n**Verdict :** `{result['verdict']}`  \n**Événements admissibles :** {len(transitions)}/202  \n**NMI topologie–route :** {observed_nmi:.4f}  \n**IC bootstrap 95 % :** [{ci_low:.4f}, {ci_high:.4f}]  \n**p permutation intra-annuelle :** {permutation_p:.6f}  \n**Classes topologiques non triviales et stables :** {stable_classes}\n\n## Interprétation\n\nR24A mesure si les issues R20 recouvrent une morphologie dynamique de première-passage. Même en cas de gate positif, cette association ne transforme pas une topologie en cause, en route dominante ou en règle directionnelle.\n\nLes signatures de flux spot–perp et de basis au premier passage servent à formuler les prochains mécanismes séparément : engagement, réponse contraire, absorption potentielle, alternance ou non-résolution.\n\n`prediction_authorized = false` et `trading_authorized = false`.\n"""
    (output_dir / "13_MASTER_REPORT.md").write_text(report)

    for source, name in [(Path(args.prereg), "14_PREREGISTRATION.json"), (Path(args.scope), "15_SCOPE.md"), (input_path, "16_FROZEN_INPUT_202.csv"), (Path(__file__), "17_EXECUTION_SCRIPT.py")]:
        (output_dir / name).write_bytes(source.read_bytes())
    checksums = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
