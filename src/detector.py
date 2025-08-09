from ultralytics import YOLO
import supervision as sv
from typing import Tuple, Dict, Any

class YOLODetector:
    def __init__(self, model_path: str, frame_rate: int = 30, config: object = None):
        self.model_path = model_path
        self.model = YOLO(model_path, verbose=True, task="detect")
        self.config = config
        self.frame_rate = frame_rate
        self.tracker = self._create_tracker()

    def _create_tracker(self) -> sv.ByteTrack:
        lost_track_buffer_seconds = getattr(self.config, 'LOST_TRACK_BUFFER_SECONDS', 5.0)
        minimum_matching_threshold = getattr(self.config, 'MINIMUM_MATCHING_THRESHOLD', 0.3)
        minimum_consecutive_frames = getattr(self.config, 'MINIMUM_CONSECUTIVE_FRAMES', 1)
        
        calculated_lost_track_buffer = int(self.frame_rate * lost_track_buffer_seconds)
        
        return sv.ByteTrack(
            lost_track_buffer=calculated_lost_track_buffer, 
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            frame_rate=int(self.frame_rate)
        )

    def predict(self, frame: any) -> Tuple[sv.Detections, Dict[int, str]]:
        results = self.model.predict(frame, verbose=False, imgsz=480 , conf=getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.5), device='cpu')
        
        detections = sv.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.int().cpu().numpy(),
        )
        
        display_confidence_threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.5)
        valid_mask = detections.confidence >= display_confidence_threshold
        valid_detections = detections[valid_mask]
        
        try:
            tracked_detections = self.tracker.update_with_detections(detections=valid_detections)
        except AttributeError:
            # Fallback for older supervision versions
            tracked_detections = self.tracker.update(detections=valid_detections)
        
        return tracked_detections, results[0].names

    def reset_tracker(self):
        self.tracker = self._create_tracker()
        #print("ByteTrack tracker reseteado.")
