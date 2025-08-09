import cv2
from datetime import datetime
import os
import supervision as sv


class FrameAnnotator:
    def __init__(self, config):
        self.config = config
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        self.text_color = (255, 255, 255)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4, text_position=sv.Position.BOTTOM_RIGHT)
        
        # FPS tracking
        import time
        self.last_time = time.time()
        self.frame_count = 0
        self.fps_display = 0.0
        self.fps_update_interval = 30  # Update FPS display every 30 frames

    def draw_detections(self, frame, detections, model_names, confidence_threshold):
        annotated_frame = frame.copy()
        
        if not self.config.SHOW_ANNOTATIONS:
            return annotated_frame

        for i in range(len(detections)):
            box = detections.xyxy[i]
            cls_id = detections.class_id[i]
            conf = detections.confidence[i]

            if conf > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                labelName = model_names[cls_id]
                label = f"{labelName}"
                
                if labelName != "person":
                    annotated_frame = self.label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections[i:i+1],
                        labels=[label]
                    )
                    
        return annotated_frame

    def draw_transparent_box(self, frame, x, y, width, height, color=(0, 0, 0), alpha=0.85):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def show_frame(self, window_name, frame):
        cv2.imshow(window_name, frame)

    def draw_text(self, frame, text, org, font_scale=0.8, thickness=2, color=(255, 255, 255)):
        return cv2.putText(frame, text, org, self.font, font_scale, color, thickness, cv2.LINE_AA)

    def save_full_frame(self, frame, captures_dir):
        os.makedirs(captures_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{captures_dir}/personas_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Guardado en: {filename}")

    def draw_fps_info(self, frame, camera_fps=None, capture_status=False):
        """Draw FPS information on the frame"""
        import time
        
        # Calculate actual FPS
        current_time = time.time()
        self.frame_count += 1
        
        if self.frame_count % self.fps_update_interval == 0:
            time_diff = current_time - self.last_time
            if time_diff > 0:
                self.fps_display = self.fps_update_interval / time_diff
            self.last_time = current_time
        
        # Draw FPS information
        y_offset = 30
        
        # Actual FPS (calculated)
        fps_text = f"FPS Real: {self.fps_display:.1f}"
        self.draw_text(frame, fps_text, (10, y_offset), font_scale=0.6, color=(0, 255, 0))
        y_offset += 25
        
        # Camera/Video FPS (from source)
        if camera_fps:
            camera_fps_text = f"FPS Fuente: {camera_fps:.1f}"
            self.draw_text(frame, camera_fps_text, (10, y_offset), font_scale=0.6, color=(255, 255, 0))
            y_offset += 25
        
        # Capture status
        if capture_status:
            capture_text = "CAPTURANDO FRAMES"
            self.draw_text(frame, capture_text, (10, y_offset), font_scale=0.6, color=(0, 0, 255))
        
        return frame
