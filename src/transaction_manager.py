import time
import json
import numpy
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import psutil
from src.event_manager import EventManager
from src.states import IdleState
from src.redis_transaction_handler import RedisTransactionHandler
from src.anomaly_detector import AnomalyDetector

def convert_numerical_values_to_float(obj):
    """Convertir valores numéricos numpy a float para JSON"""
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
        
        # 🔥 OPTIMIZACIÓN CRÍTICA: Setup threading optimizado
        self._setup_optimized_threading()
        
        # Frame capture variables
        self.capture_frames = False
        self.capture_timer = None

    def _setup_optimized_threading(self):
        """🔥 CONFIGURAR THREADING OPTIMIZADO BASADO EN HARDWARE"""
        # ✅ DETECCIÓN AUTOMÁTICA DE CAPACIDADES
        max_workers = getattr(self.config, 'MAX_THREAD_WORKERS', self._detect_optimal_workers())
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix="TxnMgr"
        )
        
        # ✅ OPTIMIZACIÓN: Usar deque con límite para evitar memory leaks
        batch_size = getattr(self.config, 'REDIS_BATCH_SIZE', 50)
        self.redis_operations = deque(maxlen=batch_size * 2)  # Buffer extra
        
        # ✅ OPTIMIZACIÓN: Batch processing configurable
        self.redis_batch_interval = getattr(self.config, 'REDIS_BATCH_INTERVAL', 5.0)
        self.redis_batch_timer = None
        self._start_batch_processing()
        
        # ✅ ESTADÍSTICAS DE RENDIMIENTO
        self.threading_stats = {
            'operations_queued': 0,
            'operations_processed': 0,
            'batch_count': 0,
            'start_time': time.time()
        }
        
        print(f"🧵 Threading optimizado: {max_workers} workers, batch cada {self.redis_batch_interval}s")

    def _detect_optimal_workers(self):
        """🔥 DETECTAR NÚMERO ÓPTIMO DE WORKERS"""
        try:
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if memory_gb < 2 or cpu_count < 4:
                return 1  # Low-end
            elif memory_gb < 8 or cpu_count < 8:
                return 2  # Medium
            else:
                return min(6, cpu_count // 3)  # High-end: usar ~1/3 de cores
        except:
            return 2  # Default fallback

    def _start_batch_processing(self):
        """🔥 INICIAR PROCESAMIENTO POR LOTES"""
        if self.redis_batch_timer:
            self.redis_batch_timer.cancel()
            
        self.redis_batch_timer = threading.Timer(self.redis_batch_interval, self._process_redis_batch)
        self.redis_batch_timer.daemon = True
        self.redis_batch_timer.start()

    def _process_redis_batch(self):
        """🔥 PROCESAR OPERACIONES REDIS EN LOTES"""
        if self.redis_operations:
            # ✅ OPTIMIZACIÓN: Procesar todas las operaciones pendientes en un lote
            batch_operations = list(self.redis_operations)
            self.redis_operations.clear()
            
            # Actualizar estadísticas
            self.threading_stats['batch_count'] += 1
            
            # Procesar en background sin bloquear
            if batch_operations:
                self.thread_pool.submit(self._execute_batch_operations, batch_operations)
                
        # Reprogramar siguiente batch
        self._start_batch_processing()

    def _execute_batch_operations(self, operations):
        """🔥 EJECUTAR LOTE DE OPERACIONES REDIS"""
        processed = 0
        for operation_data in operations:
            try:
                func, args, kwargs = operation_data
                func(*args, **kwargs)
                processed += 1
            except Exception as e:
                print(f"❌ Error en operación Redis batch: {e}")
        
        # Actualizar estadísticas
        self.threading_stats['operations_processed'] += processed

    def _queue_redis_operation(self, func, *args, **kwargs):
        """🔥 AÑADIR OPERACIÓN REDIS A LA COLA DE LOTES"""
        self.redis_operations.append((func, args, kwargs))
        self.threading_stats['operations_queued'] += 1
        
        # ✅ OPTIMIZACIÓN: Si la cola está muy llena, procesar inmediatamente
        max_queue_size = getattr(self.config, 'REDIS_BATCH_SIZE', 50)
        if len(self.redis_operations) >= max_queue_size:
            self._process_redis_batch()

    def set_state(self, state):
        """Cambiar estado de la transacción"""
        self.current_state = state
        # Start capturing when transaction becomes active
        if state.__class__.__name__ == "TransactionActiveState" and not self.capture_frames:
            self.start_capture()

    def start_capture(self):
        """Start frame capturing"""
        if getattr(self.config, 'ENABLE_CAPTURE', False):
            self.capture_frames = True
            print("📷 Iniciando captura de frames")

    def stop_capture(self):
        """Stop frame capturing"""
        self.capture_frames = False
        print("🛑 Deteniendo captura de frames")
        
    def process_frame(self, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold):
        """Procesar frame y actualizar estado"""
        self.event_manager.update_frame_counter()
        
        if num_clientes > 0 or num_cajeros > 0:
            self.last_activity_time = time.time()

        self.event_manager.update_event_last_seen_frames(detections, confidence_threshold, is_cash_register_open)
        self.current_state.handle(self, detections, num_clientes, num_cajeros, client_track_ids, cashier_track_ids, is_cash_register_open, confidence_threshold)
        self.event_manager.update_events_active_in_previous_frame(detections, confidence_threshold)

    def finalize_transaction(self, reason):
        """🔥 FINALIZAR TRANSACCIÓN OPTIMIZADA"""
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
            print(f"💳 Cliente ID: {current_transaction.client_id}, Cajero ID: {current_transaction.cashier_id} "
                  f"[Transacción Finalizada ({reason}). Eventos: {events_str}]. "
                  f"Inicio: {start_time_str}, Fin: {end_time_str}, Duración: {duration:.2f}s.")
            
            # Schedule capture stop after 5 seconds
            if getattr(self.config, 'ENABLE_CAPTURE', False) and self.capture_frames:
                if self.capture_timer:
                    self.capture_timer.cancel()
                self.capture_timer = threading.Timer(5.0, self.stop_capture)
                self.capture_timer.daemon = True
                self.capture_timer.start()
                print("⏰ Programado detener captura en 5 segundos")
            
            # 🔥 OPTIMIZACIÓN: Queue para batch processing en lugar de thread inmediato
            self._queue_transaction_processing(current_transaction, events, events_str, 
                                             duration, start_time_str, end_time_str, reason)
    
    def _queue_transaction_processing(self, transaction, events, events_str, duration, start_time_str, end_time_str, reason):
        """🔥 QUEUE TRANSACTION PROCESSING PARA BATCH"""
        # Prepare all data for batch processing
        transaction_data = {
            'transaction': transaction,
            'events': events,
            'events_str': events_str,
            'duration': duration,
            'start_time_str': start_time_str,
            'end_time_str': end_time_str,
            'reason': reason
        }
        
        # Use thread pool but with lower priority
        self.thread_pool.submit(self._process_transaction_data_optimized, transaction_data)

    def _process_transaction_data_optimized(self, transaction_data):
        """🔥 PROCESO OPTIMIZADO DE DATOS DE TRANSACCIÓN"""
        try:
            transaction = transaction_data['transaction']
            events = transaction_data['events']
            events_str = transaction_data['events_str']
            
            # Determine payment method
            payment_method = "pago_tarjeta" if "pago_tarjeta" in events else \
                           "dinero_mano" if "dinero_mano" in events else \
                           "caja_abierta" if "caja_abierta" in events else "unknown"
            
            # Convert events to a list for anomaly detection if needed
            event_list = events.split(", ") if isinstance(events, str) else events
            
            # Detect anomalies
            anomaly_status = self.anomaly_detector.is_normal_transaction(payment_method, event_list)
            
            # ✅ OPTIMIZACIÓN: Queue Redis operations para batch processing
            transaction_type = f"transaccion:{transaction.client_id}:{transaction.cashier_id}"
            self._queue_redis_operation(self.redis_handler.save_transaction, transaction_type, events)
            
            # Prepare log data
            log_data = {
                "Cliente ID": transaction.client_id,
                "Cajero ID": transaction.cashier_id,
                "Razon": transaction_data['reason'],
                "Eventos": events_str,
                "Duracion": transaction_data['duration'],
                "Inicio": transaction_data['start_time_str'],
                "Fin": transaction_data['end_time_str'],
                "Payment Method": payment_method,
                "Type": anomaly_status
            }
            
            # Convert numerical values and queue log save
            log_data = convert_numerical_values_to_float(log_data)
            self._queue_redis_operation(self.redis_handler.save_log, log_data)
            
        except Exception as e:
            print(f"❌ Error procesando datos de transacción: {e}")

    def get_transaction_status_text(self):
        """Obtener texto del estado actual de la transacción"""
        if self.current_transaction:
            client_id_str = f" (Cliente ID: {self.current_transaction.client_id})"
            cashier_id_str = f" (Cajero ID: {self.current_transaction.cashier_id})"
            events_str = ", ".join(self.current_transaction.get_events())
            return f"Estado Transacción: {self.current_state.__class__.__name__}{client_id_str}{cashier_id_str} (Eventos: {events_str})"
        else:
            return f"Estado Transacción: {self.current_state.__class__.__name__}"

    def get_threading_stats(self):
        """🔥 OBTENER ESTADÍSTICAS DE THREADING"""
        runtime = time.time() - self.threading_stats['start_time']
        return {
            'operations_queued': self.threading_stats['operations_queued'],
            'operations_processed': self.threading_stats['operations_processed'],
            'batch_count': self.threading_stats['batch_count'],
            'queue_size': len(self.redis_operations),
            'ops_per_second': self.threading_stats['operations_processed'] / runtime if runtime > 0 else 0,
            'runtime_seconds': runtime
        }

    def print_threading_stats(self):
        """🔥 IMPRIMIR ESTADÍSTICAS DE THREADING"""
        stats = self.get_threading_stats()
        print(f"🧵 Threading Stats: {stats['operations_processed']}/{stats['operations_queued']} ops procesadas, "
              f"{stats['batch_count']} batches, {stats['ops_per_second']:.1f} ops/s, "
              f"cola: {stats['queue_size']}")

    def __del__(self):
        """🔥 CLEANUP AL DESTRUIR EL OBJETO"""
        try:
            if hasattr(self, 'redis_batch_timer') and self.redis_batch_timer:
                self.redis_batch_timer.cancel()
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=False)
            if hasattr(self, 'capture_timer') and self.capture_timer:
                self.capture_timer.cancel()
        except:
            pass