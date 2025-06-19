from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

__all__ = ["detect_marker_rows"]

# ----------------------------- config ---------------------------------------
DEFAULT_DICT = cv2.aruco.DICT_4X4_100   # словарь маркеров
IDS_PER_ROW = 20                        # диапазон id для одного ряда
CONFIRM_FRAMES = 3                      # сколько кадров подряд нужен маркер
SAMPLE_EVERY = 6                        # обрабатываем каждый N‑й кадр
MAX_FRAMES = 900                        # защитное ограничение по кадрам
SCHEME_FILE = "marker_scheme.png"       # имя выходной схемы
# ---------------------------------------------------------------------------


def _detect(gray: np.ndarray, detector):
    """Единый вызов ArUco‑детектора для разных версий OpenCV."""
    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, detector[0], parameters=detector[1])
    ids = ids.flatten() if ids is not None and len(ids) else np.empty((0,), dtype=np.int32)
    return corners, ids


def _center_of_marker(corners: np.ndarray) -> Tuple[int, int]:
    """Возвращает целочисленный центр маркера по corner[4,2]."""
    pts = corners.reshape(-1, 2)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    return int(cx), int(cy)


# ---------------------------------------------------------------------------

def detect_marker_rows(
    video_path: str,
    dict_id: int = DEFAULT_DICT,
    ids_per_row: int = IDS_PER_ROW,
    confirm: int = CONFIRM_FRAMES,
    sample_every: int = SAMPLE_EVERY,
    max_frames: int = MAX_FRAMES,
    scheme_file: str = SCHEME_FILE,
):
    """Прогоняет первые *max_frames* ролика, подтверждает маркеры (≥ `confirm`
    кадров подряд) и оценивает количество рядов.

    Параметры
    ---------
    video_path : str
        Путь к файлу видео.

    Возвращает
    ----------
    rows : int
        Количество обнаруженных рядов (≥1).
    distribution : dict[int, list[int]]
        row_index → список id в порядке слева→право (по x‑координате).
    """

    # ---- подготовка детектора ----
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = (
        cv2.aruco.ArucoDetector(aruco_dict, params)
        if hasattr(cv2.aruco, "ArucoDetector")
        else (aruco_dict, params)
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    streaks: Dict[int, int] = defaultdict(int)          # id → длина текущей серии
    accepted_row: Dict[int, int] = {}                   # id → row
    accepted_center: Dict[int, Tuple[int, int]] = {}    # id → (cx, cy)

    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break
        frame_idx += 1
        if frame_idx % sample_every:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = _detect(gray, detector)
        current_set = {int(i) for i in ids}

        # --- streak обновление ---
        for mid in current_set:
            if mid in accepted_row:
                continue
            streaks[mid] += 1
            if streaks[mid] >= confirm:
                row_idx = mid // ids_per_row
                accepted_row[mid] = row_idx
                # найдём corners и запомним центр
                idx = np.where(ids == mid)[0][0]
                accepted_center[mid] = _center_of_marker(corners[idx])

        # сброс пропавших кандидатов
        for mid in list(streaks.keys()):
            if mid not in current_set and mid not in accepted_row:
                streaks[mid] = 0

    cap.release()

    if not accepted_row:
        raise RuntimeError("Не найдено ни одного подтверждённого маркера – не могу определить ряды.")

    # ---------------- схема ----------------
    # сгруппируем по рядам и отсортируем слева→направо (по x центра)
    row_to_ids: Dict[int, List[int]] = defaultdict(list)
    for mid, row in accepted_row.items():
        row_to_ids[row].append(mid)
    for row, ids in row_to_ids.items():
        ids.sort(key=lambda m: accepted_center[m][0])

    rows_detected = max(row_to_ids.keys()) + 1

    # --- рисуем схему ---
    max_per_row = max(len(v) for v in row_to_ids.values())
    cell_w, cell_h = 120, 120
    margin_x, margin_y = 60, 60
    scheme_w = margin_x * 2 + cell_w * max_per_row
    scheme_h = margin_y * 2 + cell_h * rows_detected
    scheme = 255 * np.ones((scheme_h, scheme_w, 3), dtype=np.uint8)

    for row in range(rows_detected):
        ids_in_row = row_to_ids.get(row, [])
        y = margin_y + row * cell_h
        cv2.line(scheme, (margin_x, y), (scheme_w - margin_x, y), (200, 200, 200), 1)
        for j, mid in enumerate(ids_in_row):
            x = margin_x + j * cell_w
            cv2.circle(scheme, (x, y), 15, (0, 0, 0), 2)
            cv2.putText(
                scheme,
                str(mid),
                (x - 10, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(scheme_file, scheme)
    print(f"[auto_row_detector] Схема сохранена: {scheme_file}")
    print(f"[auto_row_detector] Обнаружено рядов: {rows_detected}")
    print(f"[auto_row_detector] Карта рядов: {dict(row_to_ids)}")

    return rows_detected, dict(row_to_ids)
