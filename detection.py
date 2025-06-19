from typing import List, Dict, Optional, Tuple
import cv2, numpy as np
from ultralytics import YOLO


class RebarTracker:
    """
    YOLOv8 + ByteTrack с CLAHE-препроцессингом.
    """

    def __init__(self, weights: str,
                 rebar_name="rebar", rebar_id: Optional[int]=None,
                 conf=.4, iou=.5, tracker_cfg="bytetrack.yaml"):
        self.model = YOLO(weights)
        self.names: Dict[int,str] = self.model.names
        self.rebar_cls = rebar_id if rebar_id is not None else self._cls(rebar_name)
        self.conf, self.iou, self.tr_cfg = conf, iou, tracker_cfg
        self.clahe = cv2.createCLAHE(4.0, (8,8))

    def _cls(self,name):
        for i,n in self.names.items():
            if str(n).lower()==name.lower(): return i
        if len(self.names)==1: return next(iter(self.names))
        raise ValueError(f"'{name}' not in {self.names}")

    def _prep(self,f):
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        g=self.clahe.apply(g)
        return cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)

    def track(self,f)->List[Tuple[int,np.ndarray,float]]:
        r=self.model.track(self._prep(f),conf=self.conf,iou=self.iou,
                           tracker=self.tr_cfg,persist=True,verbose=False)[0]
        if r.boxes.id is None: return []
        out=[]
        for bb,tid,cls,cf in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.id.cpu().numpy(),
                                 r.boxes.cls.cpu().numpy(),
                                 r.boxes.conf.cpu().numpy()):
            if int(cls)==self.rebar_cls: out.append((int(tid),bb,float(cf)))
        return out

    def reset(self):
        if hasattr(self.model,"tracker") and self.model.tracker:
            self.model.tracker.reset()
