
from __future__ import annotations

from typing import Dict, Iterable

__all__ = ["RebarValidator"]


class RebarValidator:
    """Подтверждение детекций по критерию N подряд кадров."""

    def __init__(self, required: int = 3):
        """required – сколько подряд кадров должна длиться детекция."""
        if required < 1:
            raise ValueError("required must be ≥ 1")
        self.required = required
        self._counts: Dict[int, int] = {}  # track_id → consecutive counter

    # --------------------------------------------------------------
    def update(self, track_id: int, present: bool) -> bool:
        if not present:
            # сброс
            self._counts.pop(track_id, None)
            return False

        cnt = self._counts.get(track_id, 0) + 1
        self._counts[track_id] = cnt
        return cnt >= self.required

    # --------------------------------------------------------------
    def purge_missing(self, active_ids: Iterable[int]):
        """Удалить треки, которых больше нет в текущем кадре.

        Вызывать **после** того, как обработали все объекты кадра, передавая
        список id, полученных от трекера.  Это защищает от утечек памяти.
        """
        active_ids = set(active_ids)
        for tid in list(self._counts):
            if tid not in active_ids:
                self._counts.pop(tid, None)