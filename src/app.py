import cv2
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
        
        # First create transaction manager
        detector = YOLODetector(config.MODEL_PATH, frame_rate=30, config=config)  # Use default FPS initially
        model_names = detector.model.names
        transaction_manager = TransactionManager(model_names, config, detector)
        
        # Now create camera with transaction manager
        camera = CameraStream(
            config.USERNAME, config.PASSWORD, config.IP, config.PORT, config.CHANNEL,
            use_video_file=config.USE_VIDEO_FILE_FOR_CAMERA, video_path=config.VIDEO_FILE_PATH,
            transaction_manager=transaction_manager
        )

        # Update detector with actual camera FPS
        detector = YOLODetector(config.MODEL_PATH, frame_rate=camera.get_fps(), config=config)
        zone_manager = ZoneManager(model_names, config)
        cash_register_monitor = CashRegisterMonitor(model_names, config.CASH_REGISTER_CLASS_NAME)
        annotator = FrameAnnotator(config)
        
        frame_processor = FrameProcessor(
            detector, zone_manager, cash_register_monitor, transaction_manager, annotator, config
        )
        
        return CajeroClienteApp(camera, frame_processor, annotator)

class CajeroClienteApp:
    def __init__(self, camera, frame_processor, annotator):
        self.camera = camera
        self.frame_processor = frame_processor
        self.annotator = annotator

    def run(self):
        try:
            self.camera.connect()
            
            # Calculate delay for proper video playback speed
            import time
            target_fps = self.camera.get_fps()
            frame_delay = 1.0 / target_fps if target_fps > 0 else 1.0 / 30.0
            last_frame_time = time.time()
            
            print(f"Iniciando reproducción con FPS objetivo: {target_fps:.1f}")
            
            while True:
                try:
                    frame = self.camera.read_frame()
                except ConnectionError as e:
                    print(f"Error de cámara: {e}. Reintentando conexión...")
                    self.camera.release()
                    self.camera.connect()
                    continue
                
                annotated_frame = self.frame_processor.process_frame(frame, self.camera)
             
                self.annotator.show_frame('DETECCION CLIENTE - CAJERO', annotated_frame)
                
                # Control playback speed for videos
                if self.camera.use_video_file:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    last_frame_time = time.time()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
                    
        except Exception as e:
            print(f"Error en la aplicación: {str(e)}")
            
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AppFactory.create_app()
    app.run()
