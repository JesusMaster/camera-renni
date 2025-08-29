import cv2
import numpy as np
import yaml
import os
import torch
from typing import Tuple, Dict, Any
import supervision as sv

class ProductionYOLODetector:
    """
    Enterprise-grade YOLO detector optimized for edge devices.
    Prioritizes NCNN models for fast CPU inference.
    Target: <50ms inference on RPi4, <300MB RAM usage.
    """
    def __init__(self, model_path: str, config: object = None):
        self.config = config
        self.model_path = model_path
        self.model, self.model_names = self._load_optimized_model(model_path)
        self.tracker = self._create_tracker()

    def _load_optimized_model(self, model_path: str):
        """
        Loads the most optimized model available.
        Priority: NCNN > TorchScript.
        """
        print(f"DEBUG: Original model_path = {model_path}")
        
        # --- Attempt 1: Load NCNN model ---
        ncnn_model_dir = model_path.replace('.pt', '_ncnn_model')
        if os.path.isdir(ncnn_model_dir):
            print(f"🔥 Attempting to load NCNN model from: {ncnn_model_dir}")
            try:
                if not hasattr(cv2.dnn, 'readNetFromNCNN'):
                    raise AttributeError("`readNetFromNCNN` not found in cv2.dnn. Your OpenCV build may be incompatible.")
                
                param_file = os.path.join(ncnn_model_dir, 'model.ncnn.param')
                bin_file = os.path.join(ncnn_model_dir, 'model.ncnn.bin')
                metadata_file = os.path.join(ncnn_model_dir, 'metadata.yaml')

                if not (os.path.exists(param_file) and os.path.exists(bin_file)):
                    raise FileNotFoundError("NCNN .param or .bin file not found.")

                net = cv2.dnn.readNetFromNCNN(param_file, bin_file)
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                with open(metadata_file, 'r') as f:
                    model_names = yaml.safe_load(f)['names']
                
                print("✅ Successfully loaded NCNN model using OpenCV DNN.")
                self.engine = 'ncnn'
                return net, model_names
            except (AttributeError, cv2.error, FileNotFoundError) as e:
                print(f"🚨 WARNING: Failed to load NCNN model: {e}. Falling back to TorchScript.")

        # --- Attempt 2: Fallback to TorchScript model ---
        # Construct the path from the original model path, not the NCNN path
        torchscript_path = model_path.replace('.pt', '.torchscript')
        print(f"DEBUG: TorchScript path = {torchscript_path}")
        if os.path.exists(torchscript_path):
            print(f"🔥 Attempting to load TorchScript model from: {torchscript_path}")
            try:
                model = torch.jit.load(torchscript_path, map_location=torch.device('cpu'))
                model.eval()
                print("✅ Successfully loaded TorchScript model.")
                self.engine = 'torchscript'
                # For TorchScript, we need to get names from the model object itself
                model_names = model.names if hasattr(model, 'names') else {i: f'class_{i}' for i in range(80)}
                return model, model_names
            except Exception as e:
                print(f"🚨 ERROR: Failed to load TorchScript model: {e}.")

        raise RuntimeError("Failed to load any optimized model (NCNN or TorchScript).")

    def _create_tracker(self) -> sv.ByteTrack:
        frame_rate = getattr(self.config, 'CAMERA_FPS', 30) # Use a config value
        lost_track_buffer_seconds = getattr(self.config, 'LOST_TRACK_BUFFER_SECONDS', 5.0)
        minimum_matching_threshold = getattr(self.config, 'MINIMUM_MATCHING_THRESHOLD', 0.3)
        minimum_consecutive_frames = getattr(self.config, 'MINIMUM_CONSECUTIVE_FRAMES', 1)
        
        calculated_lost_track_buffer = int(frame_rate * lost_track_buffer_seconds)
        
        return sv.ByteTrack(
            lost_track_buffer=calculated_lost_track_buffer, 
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            frame_rate=int(frame_rate)
        )

    def predict(self, frame: np.ndarray) -> Tuple[sv.Detections, Dict[int, str]]:
        if self.engine == 'ncnn':
            return self._predict_ncnn(frame)
        elif self.engine == 'torchscript':
            return self._predict_torchscript(frame)
        else:
            raise RuntimeError("No valid inference engine configured.")

    def _predict_torchscript(self, frame: np.ndarray) -> Tuple[sv.Detections, Dict[int, str]]:
        """Inference using the TorchScript model."""
        # Convert NumPy array to PyTorch Tensor
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # HWC -> CHW and normalize
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            results = self.model(input_tensor)[0]  # Ultralytics TorchScript model
        
        detections = sv.Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.int().cpu().numpy(),
        )
        
        mask = detections.confidence >= self.config.CONFIDENCE_THRESHOLD
        tracked_detections = self.tracker.update_with_detections(detections=detections[mask])
        return tracked_detections, self.model_names

    def _predict_ncnn(self, frame: np.ndarray) -> Tuple[sv.Detections, Dict[int, str]]:
        """Inference using the NCNN model."""
        img_h, img_w, _ = frame.shape
        input_size = (480, 480) # Should be from metadata

        blob = cv2.dnn.blobFromImage(frame, 1/255.0, input_size, swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward()

        outputs = np.squeeze(outputs).T
        
        boxes, class_ids, confidences = [], [], []
        for row in outputs:
            x, y, w, h = row[:4]
            class_scores = row[4:]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            if confidence >= self.config.CONFIDENCE_THRESHOLD:
                x1 = int((x - w/2) * img_w / input_size[0])
                y1 = int((y - h/2) * img_h / input_size[1])
                x2 = int((x + w/2) * img_w / input_size[0])
                y2 = int((y + h/2) * img_h / input_size[1])
                
                boxes.append([x1, y1, x2, y2])
                class_ids.append(class_id)
                confidences.append(float(confidence))

        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
        )
        
        tracked_detections = self.tracker.update_with_detections(detections=detections)
        return tracked_detections, self.model_names

    def reset_tracker(self):
        self.tracker = self._create_tracker()
