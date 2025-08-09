import cv2
from time import time

class FrameProcessor:
    def __init__(self, detector, zone_manager, cash_register_monitor, transaction_manager, annotator, config):
        self.detector = detector
        self.zone_manager = zone_manager
        self.cash_register_monitor = cash_register_monitor
        self.transaction_manager = transaction_manager
        self.annotator = annotator
        self.config = config
        self.last_capture_time = 0

    def process_frame(self, frame, camera):
        height, width, _ = frame.shape
        
        detections, _ = self.detector.predict(frame)
        
        annotated_frame = self.annotator.draw_detections(frame, detections, self.detector.model.names, self.config.CONFIDENCE_THRESHOLD)

        annotated_frame, num_clientes, client_track_ids = self.zone_manager.get_client_detections(width, height, detections, self.config.CONFIDENCE_THRESHOLD, annotated_frame)
        annotated_frame, num_cajeros, cashier_track_ids = self.zone_manager.get_cashier_detections(width, height, detections, self.config.CONFIDENCE_THRESHOLD, annotated_frame)

        annotated_frame, _, is_cash_register_open = self.cash_register_monitor.update_and_get_status(
            annotated_frame, detections, self.config.CONFIDENCE_THRESHOLD
        )

        self.transaction_manager.process_frame(
            detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, self.config.CONFIDENCE_THRESHOLD
        )

        current_time = time()
        if num_clientes >= 1 and num_cajeros >= 1:
            if (current_time - self.last_capture_time) >= self.config.CAPTURE_INTERVAL:
                self.last_capture_time = current_time

        if self.config.USE_VIDEO_FILE_FOR_CAMERA:
            video_name = camera.get_current_video_name()
            if video_name:
                text_size = cv2.getTextSize(video_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = width - text_size[0] - 10
                text_y = height - 10
                annotated_frame = self.annotator.draw_text(annotated_frame, video_name, (text_x, text_y), font_scale=0.7, color=(255, 255, 255), thickness=2)
        
        # Add FPS information and capture status
        camera_fps = camera.get_fps() if hasattr(camera, 'get_fps') else None
        capture_status = (hasattr(self.transaction_manager, 'capture_frames') and 
                         self.transaction_manager.capture_frames and
                         hasattr(self.config, 'ENABLE_CAPTURE') and 
                         self.config.ENABLE_CAPTURE)
        
        annotated_frame = self.annotator.draw_fps_info(annotated_frame, camera_fps, capture_status)
        
        return annotated_frame
