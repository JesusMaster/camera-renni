import cv2
import time
import psutil
from src.config import Config
from src.camera import CameraStream
from src.detector import YOLODetector
from src.zones import ZoneManager
from src.monitor import CashRegisterMonitor
from src.display import FrameAnnotator
from src.transaction_manager import TransactionManager
from src.frame_processor import FrameProcessor

class AppFactory:
    @staticmethod
    def create_app():
        config = Config()
        
        # 🔥 OPTIMIZACIÓN: Configuración automática basada en hardware
        hardware_config = AppFactory._detect_hardware_capabilities()
        config = AppFactory._apply_hardware_optimizations(config, hardware_config)
        
        # First create transaction manager con configuración optimizada
        detector = YOLODetector(config.MODEL_PATH, frame_rate=config.MAX_FPS, config=config)
        model_names = detector.model.names
        transaction_manager = TransactionManager(model_names, config, detector)
        
        # Create camera with optimized settings
        camera = CameraStream(
            config.USERNAME, config.PASSWORD, config.IP, config.PORT, config.CHANNEL,
            use_video_file=config.USE_VIDEO_FILE_FOR_CAMERA, video_path=config.VIDEO_FILE_PATH,
            transaction_manager=transaction_manager
        )

        # Update detector with actual camera FPS (pero limitado)
        actual_fps = min(camera.get_fps(), config.MAX_FPS)
        detector = YOLODetector(config.MODEL_PATH, frame_rate=actual_fps, config=config)
        
        zone_manager = ZoneManager(model_names, config)
        cash_register_monitor = CashRegisterMonitor(model_names, config.CASH_REGISTER_CLASS_NAME)
        annotator = FrameAnnotator(config)
        
        frame_processor = FrameProcessor(
            detector, zone_manager, cash_register_monitor, transaction_manager, annotator, config
        )
        
        return CajeroClienteApp(camera, frame_processor, annotator, config)

    @staticmethod
    def _detect_hardware_capabilities():
        """Detectar capacidades del hardware"""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Detectar Raspberry Pi
        is_raspberry = False
        try:
            with open('/proc/cpuinfo', 'r') as f:
                if 'Raspberry' in f.read():
                    is_raspberry = True
        except:
            pass
            
        return {
            'cpu_count': cpu_count,
            'memory_gb': memory_gb,
            'is_raspberry': is_raspberry,
            'is_low_end': memory_gb < 2 or cpu_count < 4 or is_raspberry
        }

    @staticmethod
    def _apply_hardware_optimizations(config, hardware_config):
        """Aplicar optimizaciones basadas en hardware detectadas en runtime"""
        # Las optimizaciones principales ya están en Config.__init__()
        # Este método puede añadir ajustes adicionales si es necesario
        return config

class CajeroClienteApp:
    def __init__(self, camera, frame_processor, annotator, config):
        self.camera = camera
        self.frame_processor = frame_processor
        self.annotator = annotator
        self.config = config
        
        # 🔥 OPTIMIZACIÓN: Frame skipping variables
        self.frame_skip_counter = 0
        self.process_every_n_frames = getattr(config, 'PROCESS_EVERY_N_FRAMES', 1)
        
        # 🔥 OPTIMIZACIÓN: Control de timing
        self.last_process_time = time.time()
        self.min_process_interval = 1.0 / getattr(config, 'MAX_FPS', 30)
        
        # 🔥 OPTIMIZACIÓN: Cache para último frame procesado
        self.last_processed_result = None
        
        # 🔥 OPTIMIZACIÓN: Estadísticas de rendimiento
        self.performance_stats = {
            'frames_total': 0,
            'frames_processed': 0,
            'start_time': time.time(),
            'last_stats_time': time.time()
        }
        
        print(f"🚀 Frame Processing: Procesando 1 de cada {self.process_every_n_frames} frames")
        print(f"🎯 Target FPS: {getattr(config, 'MAX_FPS', 30)}, Min interval: {self.min_process_interval*1000:.1f}ms")

    def run(self):
        try:
            self.camera.connect()
            
            # 🔥 OPTIMIZACIÓN: FPS adaptativos limitados
            camera_fps = self.camera.get_fps()
            target_fps = min(camera_fps, getattr(self.config, 'MAX_FPS', 30))
            frame_delay = 1.0 / target_fps if target_fps > 0 else 1.0 / 30.0
            last_frame_time = time.time()
            
            print(f"🎯 Iniciando con FPS objetivo: {target_fps:.1f} (cámara: {camera_fps:.1f})")
            
            while True:
                current_time = time.time()
                self.performance_stats['frames_total'] += 1
                
                try:
                    frame = self.camera.read_frame()
                    
                    # 🔥 OPTIMIZACIÓN CRÍTICA: Frame Skipping Logic
                    should_process = self._should_process_frame(current_time)
                    
                    if should_process:
                        self.performance_stats['frames_processed'] += 1
                        
                        # 🔥 OPTIMIZACIÓN: Redimensionar frame si es muy grande
                        frame = self._optimize_frame_size(frame)
                        
                        # Procesar frame
                        annotated_frame = self.frame_processor.process_frame(frame, self.camera)
                        self.last_processed_result = annotated_frame
                        self.last_process_time = current_time
                        
                    else:
                        # 🔥 OPTIMIZACIÓN: Usar último resultado si no procesamos
                        annotated_frame = self.last_processed_result if self.last_processed_result is not None else frame
                        
                except ConnectionError as e:
                    print(f"🔌 Error de cámara: {e}. Reintentando...")
                    self.camera.release()
                    time.sleep(1)
                    self.camera.connect()
                    continue
                
                # 🔥 OPTIMIZACIÓN: Mostrar solo si está habilitado
                if getattr(self.config, 'SHOW_ANNOTATIONS', True):
                    self.annotator.show_frame('DETECCION CLIENTE - CAJERO', annotated_frame)
                
                # 🔥 OPTIMIZACIÓN: Control de velocidad mejorado
                if self.camera.use_video_file:
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    last_frame_time = time.time()
                
                # 🔥 OPTIMIZACIÓN: Mostrar estadísticas cada 100 frames
                if self.performance_stats['frames_total'] % 100 == 0:
                    self._print_performance_stats()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
                    
        except Exception as e:
            print(f"❌ Error en la aplicación: {str(e)}")
            
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self._print_final_stats()

    def _should_process_frame(self, current_time):
        """🔥 LÓGICA DE FRAME SKIPPING OPTIMIZADA"""
        # ✅ REGLA 1: Procesar cada N frames
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.process_every_n_frames:
            return False
            
        # ✅ REGLA 2: Respetar intervalo mínimo de tiempo
        if current_time - self.last_process_time < self.min_process_interval:
            return False
        
        # ✅ RESETEAR contador y procesar
        self.frame_skip_counter = 0
        return True

    def _optimize_frame_size(self, frame):
        """🔥 OPTIMIZACIÓN: Redimensionar frame si es necesario"""
        height, width = frame.shape[:2]
        
        # Límites basados en configuración de hardware
        max_width = 1280 if getattr(self.config, 'MAX_FPS', 30) >= 25 else 640
        
        if width > max_width:
            new_width = max_width
            new_height = int(height * (new_width / width))
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
        return frame

    def _print_performance_stats(self):
        """🔥 MOSTRAR ESTADÍSTICAS DE RENDIMIENTO"""
        current_time = time.time()
        elapsed = current_time - self.performance_stats['start_time']
        
        if elapsed > 0:
            fps_total = self.performance_stats['frames_total'] / elapsed
            fps_processed = self.performance_stats['frames_processed'] / elapsed
            skip_rate = ((self.performance_stats['frames_total'] - self.performance_stats['frames_processed']) / self.performance_stats['frames_total']) * 100 if self.performance_stats['frames_total'] > 0 else 0
            
            # 🔥 OBTENER USO DE RECURSOS
            try:
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
            except:
                cpu_percent = 0
                memory_mb = 0
            
            print(f"📊 RENDIMIENTO | FPS: {fps_total:.1f} total, {fps_processed:.1f} procesados | "
                  f"Skip: {skip_rate:.1f}% | CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.1f}MB")
        
        self.performance_stats['last_stats_time'] = current_time

    def _print_final_stats(self):
        """🔥 ESTADÍSTICAS FINALES AL CERRAR"""
        elapsed = time.time() - self.performance_stats['start_time']
        if elapsed > 0:
            fps_avg = self.performance_stats['frames_processed'] / elapsed
            total_frames = self.performance_stats['frames_total']
            processed_frames = self.performance_stats['frames_processed']
            
            print("\n" + "="*50)
            print("📊 ESTADÍSTICAS FINALES DE SESIÓN")
            print("="*50)
            print(f"⏱️  Duración sesión: {elapsed:.1f} segundos")
            print(f"🎬 Frames totales: {total_frames}")
            print(f"🔄 Frames procesados: {processed_frames}")
            print(f"📈 FPS promedio: {fps_avg:.1f}")
            print(f"🎯 Eficiencia: {(processed_frames/total_frames)*100:.1f}%" if total_frames > 0 else "🎯 Eficiencia: 0%")
            print("="*50)

if __name__ == "__main__":
    app = AppFactory.create_app()
    app.run()