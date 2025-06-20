import argparse
from typing import Set

import cv2
import numpy as np
from tqdm import tqdm

from detection import RebarTracker
from row_manager import RowManager
from rebar_validator import RebarValidator
from visualizer import visualize
from auto_row_detector import detect_marker_rows


# ---------------------------------------------------------------------- args --

def parse_args():
    p = argparse.ArgumentParser("Rebar counter")
    p.add_argument("--video", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--rebar_id", type=int)
    p.add_argument("--rows", type=int,
                   help="сколько рядов маркеров (по 20 ID каждый)")
    p.add_argument("--stable_frames", type=int, default=10,
                   help="сколько подряд кадров трек должен быть внутри ряда, "
                        "чтобы считаться арматуриной")
    p.add_argument("--frame_step", type=int, default=1,
                   help="обрабатывать каждый N‑й кадр (по умолчанию 1)")
    p.add_argument("--output", default="counted.mp4")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------- main --

def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(3)), int(cap.get(4))

    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # ---- прогресс-бар ----
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    pbar = tqdm(total=total_frames, unit="frame")
    frame_idx = 0

    detector = RebarTracker(args.weights, rebar_id=args.rebar_id)

    if args.rows is None:
        rows, _ = detect_marker_rows(args.video)
    else:
        rows = args.rows

    print(f"Rows: {rows}")

    row_mgr = RowManager(rows=rows)
    validator = RebarValidator(required=args.stable_frames)

    cur_row_ids: Set[int] = set()
    total_ids: Set[int] = set()
    prev_row_idx = row_mgr.current_row_index

    message = None
    msg_frames_left = 0      # показываем 60 кадров ≈ 2 сек

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # обновляем прогресс-бар и проверяем необходимость обработки кадра
        pbar.update(1)
        frame_idx += 1
        if (frame_idx - 1) % args.frame_step:
            writer.write(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        poly, corners, ids = row_mgr.polygon_for_frame(gray)

        # валидность многоугольника (должен содержать ≥ 3 вершин)
        valid_poly = poly is not None and len(poly) >= 3
        poly32 = np.ascontiguousarray(poly, dtype=np.float32) if valid_poly else None

        # ---------- переход на новый ряд ----------
        if row_mgr.current_row_index != prev_row_idx:
            message = f"Row {prev_row_idx} done: {len(cur_row_ids)} rebars"
            msg_frames_left = 60
            prev_row_idx = row_mgr.current_row_index
            cur_row_ids.clear()

        # ---------- трекинг ----------
        tracks = detector.track(frame)  # [(tid, bbox, conf), ...]
        active_ids = []
        filtered = []
        for tid, bbox, conf in tracks:
            active_ids.append(tid)
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            inside = (
                cv2.pointPolygonTest(poly32, (cx, cy), False) >= 0
                if valid_poly else False
            )

            # ---------- RebarValidator ----------
            if validator.update(tid, inside):
                filtered.append((tid, bbox, conf))
                if tid not in cur_row_ids:
                    cur_row_ids.add(tid)
                    total_ids.add(tid)

        # очистка счётчиков пропавших треков
        validator.purge_missing(active_ids)

        # ---------- визуализация ----------
        vis = visualize(
            frame, filtered, poly32 if valid_poly else None,
            row_mgr.current_row_index, corners, ids,
            len(cur_row_ids), len(total_ids),
            message if msg_frames_left else None,
        )
        msg_frames_left = max(0, msg_frames_left - 1)

        writer.write(vis)
        if args.show:
            cv2.imshow("Rebars", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    pbar.close()
    print("Total rebars counted:", len(total_ids))
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
