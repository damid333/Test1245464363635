#!/usr/bin/env python3
"""
Аудит удалённых тасок: вытаскивает из raw.csv всё по списку task_id и раскладывает
по статусам, чтобы понять, что именно снесли.

    python audit_deleted_tasks.py --raw raw.csv --tasks cleaned.csv -o ./audit

На выходе в папке -o:
    rows.csv        — все строки raw.csv по этим таскам (полный дамп, как есть)
    by_task.csv     — одна строка на таску: кадры, боксы, стадии, флаги
    by_job.csv      — одна строка на (task_id, job_stage, job_state) с диапазоном frame_id
    missing.csv     — task_id из списка, которых в raw.csv нет
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

STAGE_ORDER = ["annotation", "validation", "acceptance"]


def read_task_ids(path: Path) -> list[int]:
    df = pd.read_csv(path, low_memory=False)
    col = next((c for c in ("task_id", "id", "task") if c in df.columns), None)
    if col is None:
        raise SystemExit(f"В {path} нет колонки task_id (есть: {list(df.columns)})")
    return sorted({int(t) for t in pd.to_numeric(df[col], errors="coerce").dropna()})


def main() -> int:
    p = argparse.ArgumentParser(description="Аудит удалённых тасок CVAT по raw.csv")
    p.add_argument("--raw", type=Path, required=True, help="raw")
    p.add_argument("--tasks", type=Path, required=True, help="CSV со списком task_id (cleaned.csv)")
    p.add_argument("--deleted", type=Path, default=None, help="deleted.csv, если хочешь учесть удалённые кадры")
    p.add_argument("-o", "--out", type=Path, default=Path("audit"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw, low_memory=False)
    if args.deleted and args.deleted.exists():
        raw = pd.concat([raw, pd.read_csv(args.deleted, low_memory=False)], ignore_index=True).drop_duplicates()

    raw["task_id"] = pd.to_numeric(raw["task_id"], errors="coerce").astype("Int64")
    raw["frame_id"] = pd.to_numeric(raw["frame_id"], errors="coerce").astype("Int64")
    for c in ("job_stage", "job_state"):
        if c not in raw.columns:
            raw[c] = ""
        raw[c] = raw[c].fillna("").astype(str)

    task_ids = read_task_ids(args.tasks)
    present = set(raw["task_id"].dropna().astype(int))

    missing = sorted(set(task_ids) - present)
    pd.DataFrame({"task_id": missing}).to_csv(args.out / "missing.csv", index=False)

    rows = raw[raw["task_id"].isin(task_ids)].copy()
    rows["is_box"] = rows["instance_shape"] == "box"
    rows.to_csv(args.out / "rows.csv", index=False)

    # --- разбивка по jobs: ради чего всё и затевалось ------------------------
    by_job = (
        rows.groupby(["task_id", "task_name", "job_stage", "job_state"], dropna=False)
        .agg(
            frames=("image_name", "nunique"),
            boxes=("is_box", "sum"),
            frame_min=("frame_id", "min"),
            frame_max=("frame_id", "max"),
        )
        .reset_index()
        .sort_values(["task_id", "frame_min"])
    )
    by_job.to_csv(args.out / "by_job.csv", index=False)

    # --- сводка по таскам ----------------------------------------------------
    agg = {
        "frames": ("image_name", "nunique"),
        "boxes": ("is_box", "sum"),
        "labels": ("instance_label", "nunique"),
        "n_stages": ("job_stage", "nunique"),
        "n_states": ("job_state", "nunique"),
        "stages": ("job_stage", lambda s: "|".join(sorted(set(s)))),
        "states": ("job_state", lambda s: "|".join(sorted(set(s)))),
        "frame_max": ("frame_id", "max"),
    }
    if "task_completed" in rows.columns:
        agg["task_completed"] = ("task_completed", "first")
    if "task_updated_date" in rows.columns:
        agg["task_updated_date"] = ("task_updated_date", "max")

    by_task = rows.groupby(["task_id", "task_name"], dropna=False).agg(**agg).reset_index()

    acc = (
        rows[rows["job_stage"] == "acceptance"]
        .groupby("task_id")
        .agg(acceptance_frames=("image_name", "nunique"), acceptance_boxes=("is_box", "sum"))
        .reset_index()
    )
    by_task = by_task.merge(acc, on="task_id", how="left").fillna({"acceptance_frames": 0, "acceptance_boxes": 0})

    by_task["mixed_jobs"] = by_task["n_stages"] > 1          # таска неоднородна по стадиям
    by_task["had_annotations"] = by_task["boxes"] > 0        # была разметка
    by_task["had_accepted"] = by_task["acceptance_frames"] > 0
    by_task = by_task.sort_values("boxes", ascending=False)
    by_task.to_csv(args.out / "by_task.csv", index=False)

    # --- на экран ------------------------------------------------------------
    print(f"\nТасок в списке: {len(task_ids)} | найдено в raw.csv: {len(task_ids) - len(missing)}"
          f" | отсутствует: {len(missing)}")
    print(f"Кадров: {rows['image_name'].nunique()} | боксов: {int(rows['is_box'].sum())}")

    print("\n=== СТАТУСЫ: кадры и боксы по парам stage/state ===")
    pivot = (
        rows.groupby(["job_stage", "job_state"])
        .agg(frames=("image_name", "nunique"), boxes=("is_box", "sum"), tasks=("task_id", "nunique"))
        .reset_index()
        .sort_values("boxes", ascending=False)
    )
    print(pivot.to_string(index=False))

    print("\n=== ФЛАГИ ===")
    print(f"тасок с разметкой:               {int(by_task['had_annotations'].sum())}")
    print(f"тасок с кадрами в acceptance:    {int(by_task['had_accepted'].sum())}")
    print(f"тасок с разными стадиями jobs:   {int(by_task['mixed_jobs'].sum())}")
    print(f"боксов в acceptance:             {int(by_task['acceptance_boxes'].sum())}")
    if "task_completed" in by_task.columns:
        print(f"task_completed == True:          {int((by_task['task_completed'] == True).sum())}")
        print(f"task_completed пустой:           {int(by_task['task_completed'].isna().sum())}")
    else:
        print("колонки task_completed в CSV нет (выгрузка старого формата)")

    mixed = by_task[by_task["mixed_jobs"]]
    if len(mixed):
        print(f"\n=== ТОП НЕОДНОРОДНЫХ ТАСОК (стадии jobs разошлись) ===")
        cols = ["task_id", "task_name", "frames", "boxes", "stages", "states", "acceptance_boxes"]
        print(mixed[cols].head(20).to_string(index=False))

    print(f"\nФайлы: {args.out}/rows.csv, by_task.csv, by_job.csv, missing.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
