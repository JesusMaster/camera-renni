import time
import json
import numpy
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from src.event_manager import EventManager
from src.states import IdleState
from src.redis_transaction_handler import RedisTransactionHandler
from src.anomaly_detector import AnomalyDetector

def convert_numerical_values_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_numerical_values_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numerical_values_to_float(elem) for elem in obj]
    elif isinstance(obj, numpy.int64):
        return float(obj)
    elif isinstance(obj, (int, float)):
        return float(obj)
    else:
        return obj

class TransactionManager:
    def __init__(self, model_names, config, detector):
        self.model_names = model_names
        self.config = config
        self.detector = detector
        self.event_manager = EventManager(model_names, config)
        self.redis_handler = RedisTransactionHandler(config.REDIS_URL)
        self.anomaly_detector = AnomalyDetector()

        self.current_state = IdleState()
        self.current_transaction = None

        self.last_activity_time = time.time()
        self.last_payment_time = None
        self.last_client_seen_time = None
        self.last_cashier_seen_time = None
        
        # Frame capture variables
        self.capture_frames = False
        self.capture_timer = None
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.redis_queue = queue.Queue()
        self.redis_worker_thread = threading.Thread(target=self._process_redis_queue, daemon=True)
        self.redis_worker_thread.start()

    def set_state(self, state):
        self.current_state = state
        # Start capturing when transaction becomes active
        if state.__class__.__name__ == "TransactionActiveState" and not self.capture_frames:
            self.start_capture()

    def start_capture(self):
        """Start frame capturing"""
        if self.config.ENABLE_CAPTURE:
            self.capture_frames = True
            print("Iniciando captura de frames")

    def stop_capture(self):
        """Stop frame capturing"""
        self.capture_frames = False
        print("Deteniendo captura de frames")
        
    def _process_redis_queue(self):
        """Background thread to process Redis operations"""
        while True:
            try:
                func, args, kwargs = self.redis_queue.get()
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Error in Redis operation: {e}")
                finally:
                    self.redis_queue.task_done()
            except Exception as e:
                print(f"Error in Redis queue processing: {e}")
                
    def _queue_redis_operation(self, func, *args, **kwargs):
        """Add a Redis operation to the queue"""
        self.redis_queue.put((func, args, kwargs))

    def process_frame(self, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold):
        self.event_manager.update_frame_counter()
        
        if num_clientes > 0 or num_cajeros > 0:
            self.last_activity_time = time.time()

        self.event_manager.update_event_last_seen_frames(detections, confidence_threshold, is_cash_register_open)
        self.current_state.handle(self, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold)
        self.event_manager.update_events_active_in_previous_frame(detections, confidence_threshold)

    def finalize_transaction(self, reason):
        if self.current_transaction:
            # First, immediately update state to prevent UI lag
            current_transaction = self.current_transaction
            self.current_transaction = None
            self.set_state(IdleState())
            self.last_client_seen_time = None
            self.last_cashier_seen_time = None
            
            # Calculate basic info needed for display
            duration = current_transaction.get_duration()
            events = current_transaction.get_events()
            events_str = ", ".join(events) if events else "Ninguno"
            
            # Get time info
            current_time = time.time()
            start_time_struct = time.localtime(current_transaction.start_time)
            end_time_struct = time.localtime(current_time)
            
            start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', start_time_struct)
            end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', end_time_struct)
            
            # Print transaction summary immediately
            print(f"Cliente ID: {current_transaction.client_id}, Cajero ID: {current_transaction.cashier_id} [Transacción Finalizada ({reason}). Eventos: {events_str}]. Inicio: {start_time_str}, Fin: {end_time_str}, Duración: {duration:.2f} segundos.")
            
            # Schedule capture stop after 5 seconds
            if self.config.ENABLE_CAPTURE and self.capture_frames:
                if self.capture_timer:
                    self.capture_timer.cancel()
                self.capture_timer = threading.Timer(5.0, self.stop_capture)
                self.capture_timer.daemon = True  # Make sure thread doesn't block program exit
                self.capture_timer.start()
                print("Programado detener captura en 5 segundos")
            
            # Process Redis operations in background
            self.thread_pool.submit(self._process_transaction_data, 
                                   current_transaction, 
                                   events, 
                                   events_str, 
                                   duration, 
                                   start_time_str, 
                                   end_time_str, 
                                   reason)
    
    def _process_transaction_data(self, transaction, events, events_str, duration, start_time_str, end_time_str, reason):
        """Process transaction data in a background thread"""
        try:
            # Save transaction to Redis
            transaction_type = f"transaccion:{transaction.client_id}:{transaction.cashier_id}"
            self._queue_redis_operation(self.redis_handler.save_transaction, transaction_type, events)
            
            # Determine payment method
            payment_method = "pago_tarjeta" if "pago_tarjeta" in events else "dinero_mano" if "dinero_mano" in events else "caja_abierta" if "caja_abierta" in events else "unknown"
            
            # Convert events to a list for anomaly detection if needed
            event_list = events.split(", ") if isinstance(events, str) else events
            
            # Detect anomalies
            anomaly_status = self.anomaly_detector.is_normal_transaction(payment_method, event_list)
            
            # Prepare log data
            log_data = {
                "Cliente ID": transaction.client_id,
                "Cajero ID": transaction.cashier_id,
                "Razon": reason,
                "Eventos": events_str,
                "Duracion": duration,
                "Inicio": start_time_str,
                "Fin": end_time_str,
                "Payment Method": payment_method,
                "Type": anomaly_status
            }
            
            # Convert numerical values and save log
            log_data = convert_numerical_values_to_float(log_data)
            self._queue_redis_operation(self.redis_handler.save_log, log_data)
        except Exception as e:
            print(f"Error processing transaction data: {e}")

    def get_transaction_status_text(self):
        if self.current_transaction:
            client_id_str = f" (Cliente ID: {self.current_transaction.client_id})"
            cashier_id_str = f" (Cajero ID: {self.current_transaction.cashier_id})"
            events_str = ", ".join(self.current_transaction.get_events())
            return f"Estado Transacción: {self.current_state.__class__.__name__}{client_id_str}{cashier_id_str} (Eventos: {events_str})"
        else:
            return f"Estado Transacción: {self.current_state.__class__.__name__}"
