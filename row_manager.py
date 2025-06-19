from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Set

__all__ = ["RowManager"]

# ------------------------------------------------------------------ helpers ---
_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_4X4_100,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_APRILTAG_36h11,
]


# ---------------------------------------------------------------- RowManager ---
class RowManager:

    # --------------------------------------------------- ctor ------
    def __init__(self, rows: int, ids_per_row: int = 20, confirm: int = 3):
        self.marker_rows: int = rows              # «поясов» тегов (M)
        self.rebar_rows: int = rows + 1           # слоёв арматуры  (R = M + 1)
        self.ids_per_row: int = ids_per_row

        # ---- анти‑шум ----
        if confirm < 1:
            raise ValueError("confirm must be ≥ 1")
        self._confirm: int = confirm
        self._streaks: Dict[int, int] = {}        # id → текущая длина серии
        self._accepted: Set[int] = set()          # подтверждённые маркеры

        # ---- кэш детектора ----
        self._det = None
        self._frame_shape: Optional[Tuple[int, int]] = None  # (h,w)

        # ---- динамические структуры ----
        self._row_points: List[List[Tuple[float, float]]] = [
            [] for _ in range(self.marker_rows)
        ]
        self._boundaries: List[Optional[np.ndarray]] = [None] * self.marker_rows
        self._row_polys: List[Optional[np.ndarray]] = [None] * self.rebar_rows

        self._current_rebar_row: int = 0

    # --------------------------------------------------- utils -----
    @staticmethod
    def _bottom_two(corners: np.ndarray) -> np.ndarray:
        """Возвращает BR, BL из corners[4,2]."""
        pts = corners.reshape(-1, 2)
        return pts[[2, 3]]

    def _marker_row(self, marker_id: int) -> int:
        return min(marker_id // self.ids_per_row, self.marker_rows - 1)

    # ------------------------ ArUco detection ---------------------
    def _det_once(self, gray, det):
        if hasattr(det, "detectMarkers"):
            return det.detectMarkers(gray)[:2]
        dic, p = det
        return cv2.aruco.detectMarkers(gray, dic, parameters=p)[:2]

    def _detect_markers(self, gray):
        if self._det is not None:
            return self._det_once(gray, self._det)

        for d in _DICTS:
            dic = cv2.aruco.getPredefinedDictionary(d)
            p = cv2.aruco.DetectorParameters()
            det = (
                cv2.aruco.ArucoDetector(dic, p)
                if hasattr(cv2.aruco, "ArucoDetector")
                else (dic, p)
            )
            c, ids = self._det_once(gray, det)
            if ids is not None and len(ids):
                self._det = det
                return c, ids
        return [], None

    # ------------------- build boundaries & polys -----------------
    def _update_boundaries_and_polys(self):
        if self._frame_shape is None:
            return
        h, w = self._frame_shape

        # ---- ломаные ----
        for r in range(self.marker_rows):
            pts = self._row_points[r]
            if len(pts) < 2:
                self._boundaries[r] = None
                continue
            pts_np = np.array(sorted(pts, key=lambda p: p[0]), dtype=np.float32)
            yL, yR = pts_np[0, 1], pts_np[-1, 1]
            self._boundaries[r] = np.vstack([[0, yL], pts_np, [w - 1, yR]])

        # ---- полигоны ----
        self._row_polys = [None] * self.rebar_rows

        # rebar_row 0
        if self._boundaries[0] is not None:
            bot = np.array([[0, h - 1], [w - 1, h - 1]], dtype=np.float32)
            up = self._boundaries[0]
            self._row_polys[0] = np.vstack([bot, up[::-1]])

        # промежуточные
        for r in range(1, self.marker_rows):
            low, up = self._boundaries[r - 1], self._boundaries[r]
            if low is None or up is None:
                continue
            self._row_polys[r] = np.vstack([low, up[::-1]])

        # rebar_row R‑1
        if self._boundaries[self.marker_rows - 1] is not None:
            low = self._boundaries[self.marker_rows - 1]
            top = np.array([[0, 0], [w - 1, 0]], dtype=np.float32)
            self._row_polys[self.rebar_rows - 1] = np.vstack([low, top[::-1]])

    # --------------------------- API ------------------------------
    @property
    def current_row_index(self) -> int:
        return self._current_rebar_row

    def polygon_for_frame(self, gray):
        """Главный метод: возвращает `(poly, corners, ids)` для текущего кадра.
        Строго сохраняет исходное поведение, добавляя лишь фильтр «3 кадра подряд».
        """
        if self._frame_shape is None:
            self._frame_shape = gray.shape[:2]

        corners, ids = self._detect_markers(gray)
        if ids is None or len(ids) == 0:
            return None, corners, ids

        # ---- анти‑шум подтверждение ----
        detected_ids = ids.flatten().tolist()
        for mid in detected_ids:
            mid = int(mid)
            if mid in self._accepted:
                continue
            self._streaks[mid] = self._streaks.get(mid, 0) + 1
            if self._streaks[mid] >= self._confirm:
                self._accepted.add(mid)
        # сбрасываем streak тем, кто пропал
        for mid in list(self._streaks.keys()):
            if mid not in detected_ids and mid not in self._accepted:
                self._streaks[mid] = 0

        # --- сброс точек и сбор новых только от подтверждённых ---
        self._row_points = [[] for _ in range(self.marker_rows)]
        rows_visible = set()
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if mid not in self._accepted:
                continue  # маркер ещё не прошёл проверку
            mr = self._marker_row(mid)
            rows_visible.add(mr)
            self._row_points[mr].extend([tuple(p) for p in self._bottom_two(c)])

        # --- границы/полигоны ---
        self._update_boundaries_and_polys()

        # --- выбор слоя арматуры ---
        if rows_visible == {self.marker_rows - 1}:  # один, и это верхний пояс
            self._current_rebar_row = self.rebar_rows - 1
        elif rows_visible:
            self._current_rebar_row = max(rows_visible)
        # else: оставляем прежнее значение (ни одного подтверждённого маркера)

        return (
            self._row_polys[self._current_rebar_row],
            corners,
            ids,
        )

    # --------------------------------------------------------------
    def point_in_row(self, pt: Tuple[float, float]) -> bool:
        poly = self._row_polys[self._current_rebar_row]
        return bool(poly is not None and cv2.pointPolygonTest(poly, pt, False) >= 0)
