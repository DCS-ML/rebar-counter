
from typing import List, Tuple, Optional
import cv2, numpy as np

def _draw_markers(img, corners, ids):
    if ids is None:
        return
    for c, mid in zip(corners, ids.flatten()):
        pts = c.reshape(-1, 2).astype(int)
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
        cv2.putText(img, str(mid), (pts[0][0], pts[0][1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)


def visualize(frame: np.ndarray,
              tracks: List[Tuple[int, np.ndarray, float]],
              poly: Optional[np.ndarray],
              row_idx: int,
              corners, ids,
              row_cnt: int, tot_cnt: int,
              row_message: Optional[str],
              alpha: float = 0.25):

    out = frame.copy()

    if poly is not None and len(poly) >= 3:
        # mask: внутри полигона 1, снаружи 0
        mask = np.zeros(out.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(int)], 255)

        # затемняем снаружи
        dark = (out * alpha).astype(np.uint8)
        out = np.where(mask[:, :, None] == 0, dark, out)

        # обводим границу
        cv2.polylines(out, [poly.astype(int)], True, (255, 255, 0), 2)

    # --- треки ---
    for tid, bb, _ in tracks:
        x1, y1, x2, y2 = bb.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, str(tid), (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)

    _draw_markers(out, corners, ids)

    n_mark = 0 if ids is None else len(ids)
    cv2.putText(out, f"Row: {row_idx}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(out, f"Markers: {n_mark}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
    cv2.putText(out, f"Rebars in row: {row_cnt}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
    cv2.putText(out, f"Total rebars: {tot_cnt}", (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)

    if row_message:
        cv2.putText(out, row_message, (15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 255), 2)

    return out
