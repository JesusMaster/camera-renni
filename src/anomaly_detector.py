class AnomalyDetector:
    def __init__(self):
        pass

    def is_normal_transaction(self, payment_method, events):
        if payment_method == "pago_tarjeta":
            if "genera_boleta" in events or "entrega_boleta" in events:
                return "normal"
            else:
                return "anomalous"
        elif payment_method == "dinero_mano":
            if "genera_boleta" in events:
                return "normal"
            else:
                return "anomalous"
        elif payment_method == "caja_abierta":
            if "dinero_mano" in events and "genera_boleta" in events:
                return "normal"
            elif "genera_boleta" in events or "entrega_boleta" in events:
                return "normal"
            else:
                return "unknown"
        else:
            return "unknown"
