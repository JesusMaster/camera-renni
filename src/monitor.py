import cv2
from time import time

class CashRegisterMonitor:
    def __init__(self, model_names, cash_register_class_name="caja_abierta"):
        self.model_names = model_names
        self.cash_register_class_name = cash_register_class_name
        self.cash_register_open_start_time = None
        self.is_cash_register_open = False
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.thickness = 2
        self.text_color = (255, 255, 255) # Blanco

        try:
            self.cash_register_open_class_id = [key for key, value in model_names.items() if value == self.cash_register_class_name][0]
        except IndexError:
            print(f"Advertencia: La clase '{self.cash_register_class_name}' no se encontró en el modelo. Asegúrate de que el nombre de la clase sea correcto.")
            self.cash_register_open_class_id = None

    def update_and_get_status(self, frame, detections, confidence_threshold):
        annotated_frame = frame.copy()
        
        if self.cash_register_open_class_id is None:
            counter_text = "Clase 'caja_abierta' no encontrada"
            org_counter = (10, 150)
            annotated_frame = cv2.putText(annotated_frame, counter_text, org_counter, self.font, self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)
            return annotated_frame, self.cash_register_open_start_time, self.is_cash_register_open

        detections_cash_register_open = detections[(detections.class_id == self.cash_register_open_class_id) & (detections.confidence > confidence_threshold)]

        if len(detections_cash_register_open) > 0:
            if self.cash_register_open_start_time is None:
                self.cash_register_open_start_time = time()
            
            elapsed_time = time() - self.cash_register_open_start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            counter_text = f"Caja Abierta: {minutes:02d}:{seconds:02d}"
            self.is_cash_register_open = True
        else:
            counter_text = "Caja Cerrada"
            self.cash_register_open_start_time = None
            self.is_cash_register_open = False

        #org_counter = (10, 150) # Posición para el texto del contador
        #annotated_frame = cv2.putText(annotated_frame, counter_text, org_counter, self.font, self.font_scale, self.text_color, self.thickness, cv2.LINE_AA)

        return annotated_frame, self.cash_register_open_start_time, self.is_cash_register_open
