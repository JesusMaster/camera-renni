from abc import ABC, abstractmethod
from time import time
from src.transaction import Transaction

class State(ABC):
    @abstractmethod
    def handle(self, manager, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold):
        pass

class IdleState(State):
    def handle(self, manager, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold):
        current_time = time()
        client_present = num_clientes > 0
        cashier_present = num_cajeros > 0

        if not client_present and not cashier_present and (current_time - manager.last_activity_time) > manager.config.IDLE_RESET_TIMEOUT_SECONDS:
            manager.detector.reset_tracker()
            manager.last_activity_time = current_time
            #print(f"Tracker reseteado debido a inactividad de {manager.config.IDLE_RESET_TIMEOUT_SECONDS} segundos.")

        cooldown_active = manager.last_payment_time and (current_time - manager.last_payment_time) < manager.config.PAYMENT_COOLDOWN_SECONDS

        if client_present and cashier_present and not cooldown_active:
            is_new_cash_register_opened = manager.event_manager.is_new_event_detected(manager.config.CASH_REGISTER_CLASS_NAME, detections, confidence_threshold)
            is_new_payment_detected = manager.event_manager.is_new_event_detected("pago_tarjeta", detections, confidence_threshold) or \
                                      manager.event_manager.is_new_event_detected("dinero_mano", detections, confidence_threshold)
            is_new_bill_generated = manager.event_manager.is_new_event_detected("genera_boleta", detections, confidence_threshold)

            if is_new_cash_register_opened or is_new_payment_detected or is_new_bill_generated:
                client_id = client_track_ids[0] if client_track_ids.size > 0 else None
                cashier_id = cashier_track_ids[0] if cashier_track_ids.size > 0 else None
                if client_id is not None and cashier_id is not None:
                    manager.current_transaction = Transaction(client_id, cashier_id, manager.config)
                    manager.set_state(TransactionActiveState())
                    #print(f"Transición de estado: IDLE -> TRANSACTION_ACTIVE (Cliente ID: {client_id}, Cajero ID: {cashier_id})")
                    
                    start_events = []
                    if is_new_cash_register_opened:
                        manager.current_transaction.add_event(manager.config.CASH_REGISTER_CLASS_NAME)
                        start_events.append("caja_abierta")
                    if is_new_payment_detected:
                        if manager.event_manager.is_event_detected_in_current_frame("pago_tarjeta", detections, confidence_threshold):
                            manager.current_transaction.add_event("pago_tarjeta")
                            start_events.append("pago_tarjeta")
                        if manager.event_manager.is_event_detected_in_current_frame("dinero_mano", detections, confidence_threshold):
                            manager.current_transaction.add_event("dinero_mano")
                            start_events.append("dinero_mano")
                    if is_new_bill_generated:
                        manager.current_transaction.add_event("genera_boleta")
                        start_events.append("genera_boleta")
                    
                    #print(f"Evento de inicio detectado: {', '.join(start_events)}")


class TransactionActiveState(State):
    def handle(self, manager, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold):
        current_time = time()
        transaction = manager.current_transaction

        current_client_in_zone = transaction.client_id in client_track_ids
        current_cashier_in_zone = transaction.cashier_id in cashier_track_ids

        if current_client_in_zone:
            manager.last_client_seen_time = current_time
        if current_cashier_in_zone:
            manager.last_cashier_seen_time = current_time

        client_exit_grace_period_exceeded = not current_client_in_zone and manager.last_client_seen_time and (current_time - manager.last_client_seen_time) > manager.config.CLIENT_EXIT_GRACE_PERIOD_SECONDS
        #cashier_exit_grace_period_exceeded = not current_cashier_in_zone and manager.last_cashier_seen_time and (current_time - manager.last_cashier_seen_time) > manager.config.CASHIER_EXIT_GRACE_PERIOD_SECONDS

        if client_exit_grace_period_exceeded:
            manager.finalize_transaction(f"Cliente ID: {transaction.client_id} salió de zona por gracia")
            return
        #if cashier_exit_grace_period_exceeded:
            #manager.finalize_transaction(f"Cajero ID: {transaction.cashier_id} salió de zona por gracia")
            #return

        events_to_check = [
            manager.config.CASH_REGISTER_CLASS_NAME,
            "genera_boleta",
            "pago_tarjeta",
            "dinero_mano"
        ]
        for event_name in events_to_check:
            if manager.event_manager.is_event_currently_active(event_name, detections, confidence_threshold) and event_name not in transaction.event_set:
                transaction.add_event(event_name)
                #print(f"Evento detectado: {event_name} (Cliente ID: {transaction.client_id}, Cajero ID: {transaction.cashier_id})")

        if manager.event_manager.is_event_detected_in_current_frame("pago_tarjeta", detections, confidence_threshold) or \
           manager.event_manager.is_event_detected_in_current_frame("dinero_mano", detections, confidence_threshold):
            manager.last_payment_time = current_time

        if manager.config.CASH_REGISTER_CLASS_NAME in transaction.event_set and \
           not manager.event_manager.is_event_currently_active(manager.config.CASH_REGISTER_CLASS_NAME, detections, confidence_threshold) and \
           "cierra_caja" not in transaction.event_set:
            transaction.add_event("cierra_caja")
            #print(f"Evento detectado: cierra_caja (Inferido) (Cliente ID: {transaction.client_id}, Cajero ID: {transaction.cashier_id})")

        if manager.event_manager.is_new_event_detected(manager.config.RECEIPT_CLASS_NAME, detections, confidence_threshold):
            transaction.add_event(manager.config.RECEIPT_CLASS_NAME)
            #print(f"Evento detectado: {manager.config.RECEIPT_CLASS_NAME} (Cliente ID: {transaction.client_id}, Cajero ID: {transaction.cashier_id})")
            manager.finalize_transaction("Boleta detectada")
