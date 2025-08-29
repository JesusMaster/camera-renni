from ultralytics import YOLO
import supervision as sv
from typing import Tuple, Dict, Any
import torch
import time
import psutil

class YOLODetector:
    def __init__(self, model_path: str, frame_rate: int = 30, config: object = None):
        self.model_path = model_path
        self.config = config
        self.frame_rate = frame_rate
        
        # 🔥 CARGA OPTIMIZADA DEL MODELO
        self.model = YOLO(model_path, verbose=False, task="detect")
        
        # 🚀 SETUP DEVICE OPTIMIZADO
        self.device = self._setup_ultra_fast_device()
        
        # 🔥 CONFIGURAR MODELO PARA VELOCIDAD MÁXIMA
        self._optimize_model_for_speed()
        
        # Configurar resolución optimizada
        self.target_imgsz = getattr(config, 'TARGET_IMAGE_SIZE', 416)
        
        self.tracker = self._create_tracker()
        
        print(f"🚀 YOLO ULTRA-FAST: {self.target_imgsz}px, {self.device}, {frame_rate}fps")

    def _setup_ultra_fast_device(self):
        """🚀 CONFIGURACIÓN ULTRA-RÁPIDA"""
        
        if torch.cuda.is_available():
            device = 'cuda'
            # 🎮 OPTIMIZACIONES DE GPU
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            print(f"🎮 GPU ULTRA-FAST: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            # 🖥️ OPTIMIZACIONES DE CPU PARA 14 CORES
            cpu_count = psutil.cpu_count()
            optimal_threads = min(8, cpu_count // 2) if cpu_count > 4 else 2
            torch.set_num_threads(optimal_threads)
            torch.set_float32_matmul_precision('medium')
            print(f"🖥️ CPU ULTRA-FAST: {optimal_threads} threads optimizados de {cpu_count} disponibles")
        
        return device

    def _optimize_model_for_speed(self):
        """🔥 OPTIMIZACIÓN MÁXIMA DEL MODELO"""
        try:
            # Fusionar capas
            if hasattr(self.model, 'fuse'):
                self.model.fuse()
            
            # Modo eval para inference
            if hasattr(self.model.model, 'eval'):
                self.model.model.eval()
            
            # Optimizaciones de memoria
            if hasattr(self.model.model, 'half') and self.device == 'cuda':
                self.model.model.half()  # Precisión media en GPU
            
            print("✅ Modelo optimizado para velocidad máxima")
            
        except Exception as e:
            print(f"⚠️ Algunas optimizaciones no aplicables: {e}")

    def _create_tracker(self) -> sv.ByteTrack:
        """🔥 CREAR TRACKER OPTIMIZADO"""
        # Obtener configuraciones optimizadas del config
        lost_track_buffer_seconds = getattr(self.config, 'LOST_TRACK_BUFFER_SECONDS', 5.0)
        minimum_matching_threshold = getattr(self.config, 'MINIMUM_MATCHING_THRESHOLD', 0.70)
        minimum_consecutive_frames = getattr(self.config, 'MINIMUM_CONSECUTIVE_FRAMES', 3)
        
        calculated_lost_track_buffer = int(self.frame_rate * lost_track_buffer_seconds)
        
        return sv.ByteTrack(
            lost_track_buffer=calculated_lost_track_buffer, 
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            frame_rate=int(self.frame_rate)
        )

    def predict(self, frame: any) -> Tuple[sv.Detections, Dict[int, str]]:
        """🚀 PREDICCIÓN ULTRA-OPTIMIZADA PARA HIGH-END"""
        
        # 🔥 PARÁMETROS AGRESIVOS PARA VELOCIDAD MÁXIMA
        results = self.model.predict(
            frame, 
            verbose=False, 
            imgsz=self.target_imgsz,  # 416 para high-end
            conf=0.65,    # 🎯 MÁS ALTO para menos detecciones
            iou=0.7,      # 🎯 MÁS ALTO para menos overlaps
            device=self.device,
            half=True if self.device == 'cuda' else False,
            max_det=15,   # 🎯 REDUCIDO para velocidad
            agnostic_nms=True,
            retina_masks=False,
            
            # 🚀 OPTIMIZACIONES CRÍTICAS PARA VELOCIDAD
            augment=False,      # Sin data augmentation
            visualize=False,    # Sin visualización
            save=False,         # Sin guardar
            save_txt=False,     # Sin archivos txt
            save_conf=False,    # Sin confidence files
            save_crop=False,    # Sin crops
            show=False,         # Sin mostrar
            
            # 🎯 CONFIGURACIONES ESPECÍFICAS
            classes=None,       # Todas las clases (no filtrar)
            project=None,       # Sin proyecto
            name=None,          # Sin nombre
            exist_ok=True,      # No verificar existencia
            line_thickness=1,   # Líneas finas
            hide_labels=True,   # Ocultar labels (más rápido)
            hide_conf=True,     # Ocultar confidence (más rápido)
            
            # 🚀 OPTIMIZACIÓN DE MEMORIA
            stream=False,       # No streaming
            vid_stride=1,       # Stride normal
        )
        
        # 🔥 EARLY RETURN OPTIMIZADO
        if len(results[0].boxes) == 0:
            return sv.Detections.empty(), results[0].names
        
        # 🎯 PROCESAMIENTO MÍNIMO
        try:
            detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.int().cpu().numpy(),
            )
        except Exception as e:
            # Fallback si hay error
            return sv.Detections.empty(), results[0].names
        
        # 🔥 FILTRADO RÁPIDO
        confidence_threshold = getattr(self.config, 'CONFIDENCE_THRESHOLD', 0.65)
        valid_mask = detections.confidence >= confidence_threshold
        valid_detections = detections[valid_mask]
        
        # 🚀 TRACKING OPTIMIZADO
        try:
            tracked_detections = self.tracker.update_with_detections(detections=valid_detections)
        except (AttributeError, Exception):
            # Fallback más robusto
            try:
                tracked_detections = self.tracker.update(detections=valid_detections)
            except:
                return valid_detections, results[0].names
        
        return tracked_detections, results[0].names

    def reset_tracker(self):
        """Resetear el tracker"""
        self.tracker = self._create_tracker()
        #print("🔄 ByteTrack tracker reseteado.")