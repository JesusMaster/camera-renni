import cv2
import numpy as np
import supervision as sv

class ZoneManager:
    def __init__(self, model_names, config):
        self.model_names = model_names
        self.config = config
        self.person_class_id = [key for key, value in model_names.items() if value == "person"][0]
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator_client = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_position=sv.Position.BOTTOM_RIGHT)
        self.label_annotator_cashier = sv.LabelAnnotator(text_thickness=1, text_scale=0.5, text_position=sv.Position.TOP_LEFT)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        self.text_color = (255, 255, 255)

    def get_client_detections(self, frame_width, frame_height, detections, confidence_threshold, frame):
        overlay = frame.copy()
        polygon = np.array([
            [350, 0],
            [frame_width - 350, 0],
            [frame_width - 350, 250],
            [350, 250]
        ])

        zone_clientes = sv.PolygonZone(polygon=polygon)
        mask_cliente = zone_clientes.trigger(detections)
        
        detections_clientes = detections[(detections.class_id == self.person_class_id) & (detections.confidence > confidence_threshold) & mask_cliente]
        
        annotated_frame = overlay
        if self.config.SHOW_ANNOTATIONS:
            annotated_frame = cv2.polylines(overlay, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
            labels_clientes = [
                f"Cliente ID:{int(tracker_id)} Conf:{conf:.2f}"
                for tracker_id, conf
                in zip(detections_clientes.tracker_id, detections_clientes.confidence)
            ]
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections_clientes)
            annotated_frame = self.label_annotator_client.annotate(
                scene=annotated_frame, detections=detections_clientes, labels=labels_clientes
            )

        num_clientes = len(detections_clientes)
        return annotated_frame, num_clientes, detections_clientes.tracker_id

    def get_cashier_detections(self, frame_width, frame_height, detections, confidence_threshold, frame):
        overlay = frame.copy()
        polygon = np.array([
            [350, 220],
            [frame_width - 350, 220],
            [frame_width - 350, frame_height],
            [350, frame_height]
        ])

        zone_cajero = sv.PolygonZone(polygon=polygon)
        mask_cajero = zone_cajero.trigger(detections)
        
        detections_cajero = detections[(detections.class_id == self.person_class_id) & (detections.confidence > confidence_threshold) & mask_cajero]
        
        annotated_frame = overlay
        if self.config.SHOW_ANNOTATIONS:
            annotated_frame = cv2.polylines(overlay, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
            labels_cajeros = [
                f"Cajero ID:{int(tracker_id)} Conf:{conf:.2f}"
                for tracker_id, conf
                in zip(detections_cajero.tracker_id, detections_cajero.confidence)
            ]
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections_cajero)
            annotated_frame = self.label_annotator_cashier.annotate(
                scene=annotated_frame, detections=detections_cajero, labels=labels_cajeros
            )

        num_cajeros = len(detections_cajero)
        return annotated_frame, num_cajeros, detections_cajero.tracker_id
