#!/usr/bin/env python3
"""
Восстановление удалённых тасок CVAT из выгрузки cveta2 (raw.csv).

Что делает:
  1. Берёт список task_id, которые были удалены (cvat_tasks_to_clean.csv или --task-ids).
  2. Для каждой таски вытаскивает из raw.csv ВСЕ строки (включая instance_shape='none',
     чтобы не потерять кадры без разметки), сортирует по frame_id.
  3. Создаёт таску заново через cveta2 и заливает в неё боксы.
  4. Восстанавливает job_stage / job_state по каждому job, сопоставляя кадры по имени файла.
  5. Пишет чекпоинт, чтобы можно было прервать и продолжить.

Чего он НЕ делает (принципиально невосстановимо из CSV):
  - исходные task_id, владельца, дату создания, ассайни, историю событий;
  - оригинальное разбиение на jobs (segment_size в CSV нет — задаётся флагом);
  - issues как треды (см. --issues-out: выгружает их в CSV для ручной обработки);
  - любые фигуры кроме rectangle — cveta2 их при fetch вообще не выгружал.

Порядок запуска:
    python restore_cvat_tasks.py ... --dry-run          # preflight, ничего не создаёт
    python restore_cvat_tasks.py ... --limit 1          # одна таска, глазами проверить в UI
    python restore_cvat_tasks.py ...                    # всё остальное
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Порядок "продвинутости" стадий. Если внутри одного job кадры с разными
# парами (stage, state) — берём наименее продвинутую, как это делает сам cveta2.
# ---------------------------------------------------------------------------
STAGE_RANK = {"annotation": 0, "validation": 1, "acceptance": 2}
STATE_RANK = {"rejected": 0, "new": 1, "in progress": 2, "completed": 3}

CHECKPOINT_COLUMNS = ["old_task_id", "old_task_name", "new_task_id", "new_task_name", "status", "note"]


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
def load_raw(raw_path: Path, deleted_path: Path | None) -> pd.DataFrame:
    """Читает raw.csv (+ deleted.csv, если есть) в один датафрейм."""
    logger.info(f"Читаю {raw_path}")
    raw = pd.read_csv(raw_path, low_memory=False)

    if deleted_path and deleted_path.exists():
        logger.info(f"Читаю {deleted_path}")
        deleted = pd.read_csv(deleted_path, low_memory=False)
        # deleted.csv может дублировать строки, которые уже есть в raw.csv
        raw = pd.concat([raw, deleted], ignore_index=True).drop_duplicates()

    required = {"task_id", "task_name", "frame_id", "instance_shape", "s3_image_path", "image_name"}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"В CSV нет обязательных колонок: {sorted(missing)}")

    raw["task_id"] = pd.to_numeric(raw["task_id"], errors="coerce").astype("Int64")
    raw["frame_id"] = pd.to_numeric(raw["frame_id"], errors="coerce").astype("Int64")
    return raw


def load_task_ids(args) -> list[int]:
    """Список task_id для восстановления: из файла или из --task-ids."""
    if args.task_ids:
        return sorted({int(t) for t in args.task_ids})

    df = pd.read_csv(args.tasks_csv)
    if "task_id" not in df.columns:
        raise SystemExit(f"В {args.tasks_csv} нет колонки task_id")
    return sorted({int(t) for t in df["task_id"].dropna().unique()})


# ---------------------------------------------------------------------------
# Preflight: что вообще восстановимо
# ---------------------------------------------------------------------------
def preflight(raw: pd.DataFrame, task_ids: list[int]) -> pd.DataFrame:
    """Строит отчёт по каждой таске до того, как что-то создавать."""
    rows = []
    present = set(raw["task_id"].dropna().astype(int).unique())

    for tid in task_ids:
        if tid not in present:
            rows.append({"task_id": tid, "status": "НЕТ В RAW.CSV"})
            continue

        d = raw[raw["task_id"] == tid]
        boxes = d[d["instance_shape"] == "box"]
        frames = d[d["instance_shape"] != "deleted"]
        no_path = frames["s3_image_path"].isna().sum()

        rows.append(
            {
                "task_id": tid,
                "task_name": d["task_name"].dropna().iloc[0] if d["task_name"].notna().any() else "",
                "frames": frames["image_name"].nunique(),
                "boxes": len(boxes),
                "labels": boxes["instance_label"].nunique() if len(boxes) else 0,
                "deleted_frames": (d["instance_shape"] == "deleted").sum(),
                "rows_without_s3_path": int(no_path),
                "issues": int(d.get("issue_text", pd.Series(dtype=object)).notna().sum()),
                "stage_state_pairs": d[["job_stage", "job_state"]].drop_duplicates().shape[0]
                if {"job_stage", "job_state"} <= set(d.columns)
                else 0,
                "status": "ok" if len(boxes) or frames["image_name"].nunique() else "ПУСТО",
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Подготовка содержимого одной таски
# ---------------------------------------------------------------------------
def build_task_payload(d: pd.DataFrame, include_deleted_frames: bool):
    """
    Возвращает (content, annotations, name_to_stage_state, skipped_deleted).

    content        — список s3-путей в порядке исходного frame_id;
    annotations    — датафрейм только с боксами (его ждёт cveta2);
    name_to_stage_state — {image_name: (stage, state)} для восстановления статусов jobs.
    """
    d = d.sort_values("frame_id", kind="stable")

    deleted_mask = d["instance_shape"] == "deleted"
    skipped_deleted = sorted(d.loc[deleted_mask, "image_name"].dropna().unique().tolist())

    frames = d if include_deleted_frames else d[~deleted_mask]

    # unique() у pandas сохраняет порядок первого появления — то есть порядок frame_id
    content = frames["s3_image_path"].dropna().unique().tolist()

    annotations = frames[frames["instance_shape"] == "box"].copy()

    name_to_stage_state: dict[str, tuple[str, str]] = {}
    if {"job_stage", "job_state"} <= set(frames.columns):
        for name, grp in frames.groupby("image_name"):
            stage = str(grp["job_stage"].dropna().iloc[0]) if grp["job_stage"].notna().any() else ""
            state = str(grp["job_state"].dropna().iloc[0]) if grp["job_state"].notna().any() else ""
            if stage or state:
                name_to_stage_state[str(name)] = (stage, state)

    return content, annotations, name_to_stage_state, skipped_deleted


# ---------------------------------------------------------------------------
# Восстановление stage/state по jobs новой таски
# ---------------------------------------------------------------------------
def restore_job_states(client, new_task_id: int, name_to_stage_state: dict[str, tuple[str, str]]) -> str:
    """
    Сопоставляет кадры новой таски с исходными (по basename) и выставляет
    каждому job наименее продвинутую пару (stage, state) среди его кадров.
    """
    from cvat_sdk.api_client import models

    task = client.tasks.retrieve(new_task_id)
    frames_info = task.get_frames_info()
    frame_names = [Path(str(f.name)).name for f in frames_info]

    notes = []
    for job in task.get_jobs():
        start, stop = int(job.start_frame), int(job.stop_frame)
        pairs = [
            name_to_stage_state[n]
            for n in frame_names[start : stop + 1]
            if n in name_to_stage_state
        ]
        if not pairs:
            continue

        # наименее продвинутая пара — как в конвенции cveta2
        stage, state = min(
            pairs, key=lambda p: (STAGE_RANK.get(p[0], 0), STATE_RANK.get(p[1], 1))
        )
        if not stage and not state:
            continue

        payload = {}
        if stage:
            payload["stage"] = stage
        if state:
            payload["state"] = state

        try:
            client.api_client.jobs_api.partial_update(
                job.id, patched_job_write_request=models.PatchedJobWriteRequest(**payload)
            )
            notes.append(f"job {job.id}: {stage}/{state}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"  job {job.id}: не удалось выставить {payload}: {e}")
            notes.append(f"job {job.id}: FAILED")

    return "; ".join(notes)


# ---------------------------------------------------------------------------
# Чекпоинт
# ---------------------------------------------------------------------------
def load_checkpoint(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=CHECKPOINT_COLUMNS)


def append_checkpoint(path: Path, record: dict) -> None:
    df = pd.DataFrame([record], columns=CHECKPOINT_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Восстановление удалённых тасок CVAT из raw.csv")
    p.add_argument("--raw", type=Path, required=True, help="путь к raw.csv из cveta2 fetch --raw")
    p.add_argument("--deleted", type=Path, default=None, help="путь к deleted.csv (опционально)")
    p.add_argument("--tasks-csv", type=Path, default=None, help="CSV со списком удалённых task_id")
    p.add_argument("--task-ids", nargs="*", type=int, default=None, help="task_id через пробел")
    p.add_argument("--project-name", required=True, help="имя проекта CVAT, напр. vera-ctm-nta4")
    p.add_argument("--cvat-name", default="sip", help="имя конфига подключения для cveta2 CVAT()")
    p.add_argument("--segment-size", type=int, default=100, help="размер job у создаваемых тасок")
    p.add_argument("--image-quality", type=int, default=100)
    p.add_argument("--name-suffix", default="_restored", help="суффикс к исходному имени таски")
    p.add_argument("--include-deleted-frames", action="store_true",
                   help="включать в таску кадры, помеченные в CVAT как удалённые")
    p.add_argument("--no-job-states", action="store_true", help="не восстанавливать stage/state")
    p.add_argument("--checkpoint", type=Path, default=Path("restore_checkpoint.csv"))
    p.add_argument("--preflight-out", type=Path, default=Path("restore_preflight.csv"))
    p.add_argument("--issues-out", type=Path, default=Path("issues_to_recreate.csv"))
    p.add_argument("--limit", type=int, default=None, help="обработать не больше N тасок")
    p.add_argument("--dry-run", action="store_true", help="только preflight, ничего не создавать")
    args = p.parse_args()

    if not args.tasks_csv and not args.task_ids:
        raise SystemExit("нужен либо --tasks-csv, либо --task-ids")

    raw = load_raw(args.raw, args.deleted)
    task_ids = load_task_ids(args)
    logger.info(f"Тасок к восстановлению: {len(task_ids)}")

    # --- preflight -----------------------------------------------------------
    report = preflight(raw, task_ids)
    report.to_csv(args.preflight_out, index=False)
    logger.info(f"Preflight сохранён: {args.preflight_out}")

    bad = report[report["status"] != "ok"]
    if len(bad):
        logger.warning(f"Тасок без данных в raw.csv: {len(bad)} — они восстановлены не будут")
        print(bad.to_string(index=False))

    total_boxes = int(report.get("boxes", pd.Series(dtype=int)).fillna(0).sum())
    total_frames = int(report.get("frames", pd.Series(dtype=int)).fillna(0).sum())
    logger.info(f"Итого к восстановлению: {total_frames} кадров, {total_boxes} боксов")

    # issues выгружаем отдельно — upload их в исходном виде не вернёт
    if "issue_text" in raw.columns:
        iss = raw[(raw["task_id"].isin(task_ids)) & raw["issue_text"].notna() & (raw["issue_text"] != "")]
        if len(iss):
            cols = [c for c in ["task_id", "task_name", "image_name", "frame_id", "issue_text",
                                "issue_state", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
                    if c in iss.columns]
            iss[cols].to_csv(args.issues_out, index=False)
            logger.warning(f"Issues ({len(iss)} строк) выгружены в {args.issues_out} — "
                           "восстанавливать вручную, автоматом статусы/авторы не вернутся")

    if args.dry_run:
        logger.info("--dry-run: остановился после preflight")
        return 0

    # --- собственно восстановление ------------------------------------------
    from cveta.cvat.cvat_tools import CVAT
    from cvat_sdk import make_client

    done = load_checkpoint(args.checkpoint)
    already = set(done.loc[done["status"] == "ok", "old_task_id"].astype(int)) if len(done) else set()
    if already:
        logger.info(f"В чекпоинте уже успешно восстановлено: {len(already)} — пропускаю их")

    todo = [t for t in task_ids if t not in already and t in set(report.loc[report["status"] == "ok", "task_id"])]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(f"К обработке в этом запуске: {len(todo)}")

    cvat = CVAT(cvat_name=args.cvat_name)

    client = None
    if not args.no_job_states:
        host, user, password = os.getenv("CVAT_HOST"), os.getenv("CVAT_USER"), os.getenv("CVAT_PASS")
        if not all([host, user, password]):
            logger.warning("CVAT_HOST/CVAT_USER/CVAT_PASS не заданы — stage/state восстанавливаться не будут")
        else:
            client = make_client(host=host, credentials=(user, password))
            client.__enter__()

    try:
        for i, tid in enumerate(todo, 1):
            d = raw[raw["task_id"] == tid]
            old_name = str(d["task_name"].dropna().iloc[0])
            new_name = f"{old_name}{args.name_suffix}"

            content, annotations, name_map, skipped = build_task_payload(d, args.include_deleted_frames)
            logger.info(
                f"[{i}/{len(todo)}] {old_name} -> {new_name}: "
                f"{len(content)} кадров, {len(annotations)} боксов"
                + (f", пропущено удалённых кадров: {len(skipped)}" if skipped else "")
            )

            if not content:
                append_checkpoint(args.checkpoint, {
                    "old_task_id": tid, "old_task_name": old_name, "new_task_id": "",
                    "new_task_name": new_name, "status": "skip", "note": "нет s3-путей"})
                continue

            try:
                cvat.create_task(
                    name=new_name,
                    labels=None,
                    content=content,
                    annotations=annotations,
                    assignee=None,
                    image_quality=args.image_quality,
                    project_id=None,
                    project_name=args.project_name,
                    segment_size=args.segment_size,
                    annotation_xml_path=None,
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"  не удалось создать таску: {e}")
                append_checkpoint(args.checkpoint, {
                    "old_task_id": tid, "old_task_name": old_name, "new_task_id": "",
                    "new_task_name": new_name, "status": "failed", "note": str(e)[:300]})
                continue

            # create_task не возвращает id — находим свежесозданную таску по имени
            new_id, note = "", ""
            if client is not None:
                try:
                    found = client.tasks.list(filter=f'{{"and":[{{"==":[{{"var":"name"}},"{new_name}"]}}]}}')
                    if found:
                        new_id = max(t.id for t in found)
                        if not args.no_job_states and name_map:
                            note = restore_job_states(client, new_id, name_map)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"  не удалось доработать статусы: {e}")
                    note = f"stage/state FAILED: {e}"[:300]

            append_checkpoint(args.checkpoint, {
                "old_task_id": tid, "old_task_name": old_name, "new_task_id": new_id,
                "new_task_name": new_name, "status": "ok", "note": note[:300]})
    finally:
        if client is not None:
            client.__exit__(None, None, None)

    logger.success(f"Готово. Итоги в {args.checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
