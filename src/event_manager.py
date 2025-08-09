class EventManager:
    def __init__(self, model_names, config):
        self.model_names = model_names
        self.config = config
        self.transaction_events = {}
        self.event_last_seen_frame = {}
        self.events_active_in_previous_frame = {}
        self.frame_counter = 0

        self._map_event_classes()
        self._initialize_event_last_seen_frame()
        self._initialize_events_active_in_previous_frame()

    def _map_event_classes(self):
        event_classes = [
            self.config.CASH_REGISTER_CLASS_NAME,
            "genera_boleta",
            self.config.RECEIPT_CLASS_NAME,
            "pago_tarjeta",
            "dinero_mano",
            "cierra_caja"
        ]
        for class_name in event_classes:
            try:
                key_name = class_name
                self.transaction_events[key_name] = [key for key, value in self.model_names.items() if value == class_name][0]
            except IndexError:
                #print(f"Advertencia: La clase '{class_name}' no se encontró en el modelo.")
                self.transaction_events[class_name] = None

    def _initialize_event_last_seen_frame(self):
        for event_name in self.transaction_events.keys():
            self.event_last_seen_frame[event_name] = -1
        self.event_last_seen_frame["cierra_caja"] = -1

    def _initialize_events_active_in_previous_frame(self):
        self.events_active_in_previous_frame = {name: False for name in self.transaction_events.keys()}
        self.events_active_in_previous_frame["cierra_caja"] = False

    def update_frame_counter(self):
        self.frame_counter += 1

    def is_event_detected_in_current_frame(self, event_name, detections, confidence_threshold):
        class_id = self.transaction_events.get(event_name)
        if class_id is None:
            return False
        event_detections = detections[(detections.class_id == class_id) & (detections.confidence > confidence_threshold)]
        return len(event_detections) > 0

    def is_event_currently_active(self, event_name, detections, confidence_threshold):
        is_detected_now = self.is_event_detected_in_current_frame(event_name, detections, confidence_threshold)
        if is_detected_now:
            return True
        last_seen = self.event_last_seen_frame.get(event_name, -1)
        return (self.frame_counter - last_seen) <= self.config.EVENT_PERSISTENCE_FRAMES

    def is_new_event_detected(self, event_name, detections, confidence_threshold):
        is_detected_now = self.is_event_detected_in_current_frame(event_name, detections, confidence_threshold)
        was_active_in_previous_frame = self.events_active_in_previous_frame.get(event_name, False)
        return is_detected_now and not was_active_in_previous_frame

    def update_event_last_seen_frames(self, detections, confidence_threshold, is_cash_register_open):
        for event_name in self.transaction_events.keys():
            if self.is_event_detected_in_current_frame(event_name, detections, confidence_threshold):
                self.event_last_seen_frame[event_name] = self.frame_counter
        if not is_cash_register_open:
            self.event_last_seen_frame["cierra_caja"] = self.frame_counter

    def update_events_active_in_previous_frame(self, detections, confidence_threshold):
        self.events_active_in_previous_frame = {
            name: self.is_event_currently_active(name, detections, confidence_threshold)
            for name in self.transaction_events.keys()
        }
        self.events_active_in_previous_frame["cierra_caja"] = not self.is_event_currently_active(self.config.CASH_REGISTER_CLASS_NAME, detections, confidence_threshold)
