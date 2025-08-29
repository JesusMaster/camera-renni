import os
import psutil

class Config:
    def __init__(self):
        # Configuración base
        self._set_base_config()
        
        # 🔥 DETECCIÓN AUTOMÁTICA DE HARDWARE
        self._detect_and_apply_optimizations()
        
    def _set_base_config(self):
        """Configuración base del sistema"""
        # General settings
        self.USERNAME = "casa1"
        self.PASSWORD = "casa2020"
        self.IP = "adminrenni.ddns.net"
        self.PORT = "554"
        self.CHANNEL = "1"
        self.MODEL_PATH = "./models/model_480_80_12n_140825.pt"
        self.CAPTURE_DIR = "capture"

        # Class names
        self.CASH_REGISTER_CLASS_NAME = "caja_abierta"
        self.RECEIPT_CLASS_NAME = "entrega_boleta"

        # Video source
        self.USE_VIDEO_FILE_FOR_CAMERA = False
        self.VIDEO_FILE_PATH = None

        # Display settings
        self.SHOW_ANNOTATIONS = True

        # Redis settings
        self.REDIS_URL = "redis://default:6XfuHicxU4GSGsKd0v6dkS50hH2YtfqH@redis-12116.c240.us-east-1-3.ec2.redns.redis-cloud.com:12116"

        # Frame capture settings
        self.ENABLE_CAPTURE = False

    def _detect_and_apply_optimizations(self):
        """🔥 DETECTAR HARDWARE Y APLICAR CONFIGURACIONES AUTOMÁTICAS"""
        try:
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            # Detectar Raspberry Pi
            is_raspberry = self._detect_raspberry_pi()
            
            # Determinar perfil de rendimiento
            if is_raspberry or memory_gb < 2 or cpu_count < 4:
                self._apply_low_end_config()
                print("🔧 PERFIL: LOW-END aplicado (RPi/Hardware Limitado)")
            elif memory_gb < 4 or cpu_count < 8:
                self._apply_medium_config()
                print("🔧 PERFIL: MEDIUM aplicado")
            else:
                self._apply_high_end_config()
                print("🔧 PERFIL: HIGH-END aplicado")
                
            print(f"📊 Hardware detectado: {cpu_count} cores, {memory_gb:.1f}GB RAM, RPi: {is_raspberry}")
            
        except Exception as e:
            print(f"⚠️ Error detectando hardware, usando configuración LOW-END: {e}")
            self._apply_low_end_config()

    def _detect_raspberry_pi(self):
        """Detectar si estamos en Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'Raspberry' in cpuinfo or 'BCM' in cpuinfo
        except:
            return False

    def _apply_low_end_config(self):
        """🔥 CONFIGURACIÓN PARA DISPOSITIVOS DE BAJO RENDIMIENTO"""
        # YOLO Optimizations
        self.CONFIDENCE_THRESHOLD = 0.70  # Más alto para menos falsos positivos
        self.TARGET_IMAGE_SIZE = 320      # Reducido significativamente
        
        # Frame Processing
        self.MAX_FPS = 10                 # Máximo 10 FPS
        self.PROCESS_EVERY_N_FRAMES = 5   # Procesar 1 de cada 5 frames
        self.CAPTURE_INTERVAL = 2.0       # Capturar cada 2 segundos
        
        # Threading Optimization
        self.MAX_THREAD_WORKERS = 1       # Solo 1 worker thread
        self.REDIS_BATCH_INTERVAL = 10.0  # Batch cada 10 segundos
        self.REDIS_BATCH_SIZE = 5         # Lotes pequeños
        
        # Memory Management
        self.MAX_MEMORY_MB = 300          # Límite de memoria estricto
        self.ENABLE_MEMORY_MONITOR = True
        
        # Tracking Optimization
        self.LOST_TRACK_BUFFER_SECONDS = 2.0     # Reducido
        self.MINIMUM_MATCHING_THRESHOLD = 0.90   # Más estricto
        self.MINIMUM_CONSECUTIVE_FRAMES = 3      # Menos frames requeridos
        self.EVENT_PERSISTENCE_FRAMES = 10       # Reducido significativamente
        
        # Timeouts más largos para dispositivos lentos
        self.CLIENT_EXIT_GRACE_PERIOD_SECONDS = 6.0
        self.CASHIER_EXIT_GRACE_PERIOD_SECONDS = 6.0
        self.IDLE_RESET_TIMEOUT_SECONDS = 60.0   # Más tiempo
        self.PAYMENT_COOLDOWN_SECONDS = 4.0

    def _apply_medium_config(self):
        """🔥 CONFIGURACIÓN PARA DISPOSITIVOS DE RENDIMIENTO MEDIO"""
        # YOLO Settings
        self.CONFIDENCE_THRESHOLD = 0.65
        self.TARGET_IMAGE_SIZE = 416
        
        # Frame Processing
        self.MAX_FPS = 15
        self.PROCESS_EVERY_N_FRAMES = 3    # Procesar 1 de cada 3 frames
        self.CAPTURE_INTERVAL = 1.0
        
        # Threading
        self.MAX_THREAD_WORKERS = 2
        self.REDIS_BATCH_INTERVAL = 5.0    # Batch cada 5 segundos
        self.REDIS_BATCH_SIZE = 10
        
        # Memory Management
        self.MAX_MEMORY_MB = 500
        self.ENABLE_MEMORY_MONITOR = True
        
        # Tracking
        self.LOST_TRACK_BUFFER_SECONDS = 5.0
        self.MINIMUM_MATCHING_THRESHOLD = 0.85
        self.MINIMUM_CONSECUTIVE_FRAMES = 5
        self.EVENT_PERSISTENCE_FRAMES = 20
        
        # Timeouts
        self.CLIENT_EXIT_GRACE_PERIOD_SECONDS = 4.0
        self.CASHIER_EXIT_GRACE_PERIOD_SECONDS = 4.0
        self.IDLE_RESET_TIMEOUT_SECONDS = 30.0
        self.PAYMENT_COOLDOWN_SECONDS = 2.0

    def _apply_high_end_config(self):
        """🔥 CONFIGURACIÓN ESPECÍFICA PARA 14-CORE, 36GB SYSTEM"""
        
        # 🎯 YOLO Settings - OPTIMIZADO PARA VELOCIDAD
        self.CONFIDENCE_THRESHOLD = 0.60  # Ligeramente más alto para menos detecciones
        self.TARGET_IMAGE_SIZE = 416       # Balance velocidad/precisión (reducido de 640)
        
        # 🚀 Frame Processing - APROVECHA TU HARDWARE
        self.MAX_FPS = 30                 # Tu hardware puede manejar más
        self.PROCESS_EVERY_N_FRAMES = 1   # Procesar TODOS los frames
        self.CAPTURE_INTERVAL = 0.2       # Más agresivo
        
        # 🧵 Threading - ESPECÍFICO PARA 14 CORES
        self.MAX_THREAD_WORKERS = 6       # ~40% de tus cores
        self.REDIS_BATCH_INTERVAL = 0.5   # Batch cada 500ms
        self.REDIS_BATCH_SIZE = 100       # Lotes grandes
        
        # 💾 Memory - APROVECHA TUS 36GB
        self.MAX_MEMORY_MB = 4000         # Generoso con tu RAM
        self.ENABLE_MEMORY_MONITOR = False # No necesario
        
        # 🎯 Tracking - OPTIMIZADO PARA VELOCIDAD
        self.LOST_TRACK_BUFFER_SECONDS = 5.0
        self.MINIMUM_MATCHING_THRESHOLD = 0.70   # Más permisivo = más rápido
        self.MINIMUM_CONSECUTIVE_FRAMES = 3      # Menos frames = más rápido
        self.EVENT_PERSISTENCE_FRAMES = 20       # Reducido para velocidad
        
        # ⏱️ Timeouts - MÁS RÁPIDOS
        self.CLIENT_EXIT_GRACE_PERIOD_SECONDS = 2.0   # Más rápido
        self.CASHIER_EXIT_GRACE_PERIOD_SECONDS = 2.0  # Más rápido
        self.IDLE_RESET_TIMEOUT_SECONDS = 15.0        # Más ágil
        self.PAYMENT_COOLDOWN_SECONDS = 1.0           # Más rápido
        
        print("🚀 HIGH-END PERSONALIZADO: Configuración ULTRA-RÁPIDA aplicada")
        print("   • TARGET: <50ms inferencia, 30 FPS reales")
        print("   • HARDWARE: Optimizado para 14 cores, 36GB RAM")

    def get_performance_profile(self):
        """Obtener información del perfil de rendimiento actual"""
        try:
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            is_raspberry = self._detect_raspberry_pi()
            
            if hasattr(self, 'MAX_FPS'):
                if self.MAX_FPS <= 10:
                    profile = "LOW-END"
                elif self.MAX_FPS <= 15:
                    profile = "MEDIUM"
                else:
                    profile = "HIGH-END"
            else:
                profile = "UNKNOWN"
                
            return {
                'profile': profile,
                'cpu_count': cpu_count,
                'memory_gb': memory_gb,
                'is_raspberry': is_raspberry,
                'max_fps': getattr(self, 'MAX_FPS', 'N/A'),
                'confidence_threshold': getattr(self, 'CONFIDENCE_THRESHOLD', 'N/A'),
                'process_every_n_frames': getattr(self, 'PROCESS_EVERY_N_FRAMES', 'N/A'),
                'max_thread_workers': getattr(self, 'MAX_THREAD_WORKERS', 'N/A')
            }
        except:
            return {'profile': 'ERROR', 'details': 'No se pudo obtener información'}

    def print_current_config(self):
        """🔥 MOSTRAR CONFIGURACIÓN ACTUAL"""
        profile_info = self.get_performance_profile()
        
        print("="*60)
        print("🚀 CONFIGURACIÓN CAMERA-RENNI OPTIMIZADA")
        print("="*60)
        print(f"📊 Perfil de Rendimiento: {profile_info['profile']}")
        print(f"🖥️  Hardware: {profile_info['cpu_count']} cores, {profile_info['memory_gb']:.1f}GB RAM")
        print(f"🍓 Raspberry Pi: {'Sí' if profile_info['is_raspberry'] else 'No'}")
        print("-"*60)
        print("⚙️ CONFIGURACIÓN APLICADA:")
        print(f"   • FPS Máximo: {profile_info['max_fps']}")
        print(f"   • Confianza: {profile_info['confidence_threshold']}")
        print(f"   • Procesar cada: {profile_info['process_every_n_frames']} frames")
        print(f"   • Workers: {profile_info['max_thread_workers']} threads")
        print(f"   • Imagen YOLO: {getattr(self, 'TARGET_IMAGE_SIZE', 'N/A')}px")
        print(f"   • Memoria límite: {getattr(self, 'MAX_MEMORY_MB', 'N/A')}MB")
        print(f"   • Batch Redis: cada {getattr(self, 'REDIS_BATCH_INTERVAL', 'N/A')}s")
        print("="*60)