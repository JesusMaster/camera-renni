import cv2
import os
from time import sleep
import random
import datetime
import threading
import queue

class CameraStream:
    def __init__(self, username, password, ip, port, channel, max_retries=3, use_video_file=False, video_path=None, transaction_manager=None):
        self.username = username
        self.password = password
        self.ip = ip
        self.port = port
        self.channel = channel
        self.max_retries = max_retries
        self.cap = None
        self.use_video_file = use_video_file
        self.video_path = video_path
        self.url = f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/cam/realmonitor?channel={self.channel}&subtype=0"
        
        # Increase timeout to 60 seconds (60,000,000 microseconds)
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;60000000|stimeout;60000000"
        
        self.fps = None
        self.current_video_name = None
        self.transaction_manager = transaction_manager
        
        # Frame buffer for async processing
        self.frame_buffer = queue.Queue(maxsize=5)
        self.frame_ready = threading.Event()
        self.stop_event = threading.Event()
        self.frame_thread = None
        self.last_frame = None
        self.reconnect_lock = threading.Lock()

    def connect(self):
        if self.use_video_file:
            selected_video_path = self.video_path or self._get_random_video_file("videos")
            if not selected_video_path:
                raise FileNotFoundError("No video files found in the 'videos' folder.")
            
            print(f"Attempting to open video file: {selected_video_path}")
            self.cap = cv2.VideoCapture(selected_video_path)
            if self.cap.isOpened():
                print("Video opened successfully!")
                self.fps = self._get_actual_fps()
                self.current_video_name = os.path.basename(selected_video_path)
                
                # Start frame reading thread
                self._start_frame_thread()
                return True
            else:
                raise ConnectionError(f"Could not open video file: {selected_video_path}")
        else:
            print(f"Attempting to connect to: {self.url}")
            for attempt in range(self.max_retries):
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                if self.cap.isOpened():
                    print("Connection successful!")
                    self.fps = self._get_actual_fps()
                    self.current_video_name = None
                    
                    # Start frame reading thread
                    self._start_frame_thread()
                    return True
                else:
                    print(f"Attempt {attempt + 1} failed. Retrying...")
                    self.cap.release()
                    sleep(2)
            raise ConnectionError("Could not connect to the camera after several attempts.")
    
    def _start_frame_thread(self):
        """Start a background thread to continuously read frames"""
        if self.frame_thread is not None and self.frame_thread.is_alive():
            self.stop_event.set()
            self.frame_thread.join(timeout=1.0)
            
        self.stop_event.clear()
        self.frame_thread = threading.Thread(target=self._frame_reader_thread, daemon=True)
        self.frame_thread.start()
        
    def _frame_reader_thread(self):
        """Background thread that continuously reads frames from the camera"""
        while not self.stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                with self.reconnect_lock:
                    if not self.cap or not self.cap.isOpened():
                        try:
                            print("Frame thread detected closed connection, reconnecting...")
                            self.release()
                            self.connect()
                        except Exception as e:
                            print(f"Error reconnecting in frame thread: {e}")
                            sleep(1)
                            continue
            
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    # Try to put the frame in the buffer, but don't block if full
                    try:
                        # Remove oldest frame if buffer is full
                        if self.frame_buffer.full():
                            try:
                                self.frame_buffer.get_nowait()
                            except queue.Empty:
                                pass
                        
                        self.frame_buffer.put(frame, block=False)
                        self.last_frame = frame
                        self.frame_ready.set()
                    except queue.Full:
                        pass  # Skip this frame if buffer is full
                else:
                    # If we couldn't read a frame, wait a bit before trying again
                    sleep(0.01)
            except Exception as e:
                print(f"Error reading frame in background thread: {e}")
                sleep(0.1)

    def _get_actual_fps(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            print(f"FPS detectado de la fuente: {fps:.1f}")
            return fps
        print("No se pudo obtener FPS de la fuente, usando 30.0 FPS por defecto.")
        return 30.0

    def _get_random_video_file(self, directory):
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        video_files = [f for f in os.listdir(directory) if f.lower().endswith(video_extensions)]
        if video_files:
            return os.path.join(directory, random.choice(video_files))
        return None

    def get_fps(self):
        return self.fps if self.fps is not None else 30.0

    def get_current_video_name(self):
        return self.current_video_name

    def read_frame(self):
        # Try to get a frame from the buffer
        try:
            # Wait for a frame to be available
            if self.frame_ready.wait(timeout=1.0):
                self.frame_ready.clear()
                
                try:
                    # Get the newest frame from the buffer
                    frame = self.frame_buffer.get(block=False)
                    
                    # Capture frame if enabled and transaction manager is available
                    if (self.transaction_manager and 
                        hasattr(self.transaction_manager, 'capture_frames') and 
                        self.transaction_manager.capture_frames and
                        hasattr(self.transaction_manager.config, 'ENABLE_CAPTURE') and
                        self.transaction_manager.config.ENABLE_CAPTURE):
                        self._save_frame(frame)
                    
                    return frame
                except queue.Empty:
                    # If buffer is empty but we have a last frame, use that
                    if self.last_frame is not None:
                        return self.last_frame.copy()
        except Exception as e:
            print(f"Error getting frame from buffer: {e}")
        
        # If we couldn't get a frame from the buffer, try direct capture as fallback
        with self.reconnect_lock:
            if not self.cap or not self.cap.isOpened():
                self._reconnect()

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._reconnect()
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise ConnectionError("Persistent error reading frame after reconnection.")
            
            # Capture frame if enabled and transaction manager is available
            if (self.transaction_manager and 
                hasattr(self.transaction_manager, 'capture_frames') and 
                self.transaction_manager.capture_frames and
                hasattr(self.transaction_manager.config, 'ENABLE_CAPTURE') and
                self.transaction_manager.config.ENABLE_CAPTURE):
                self._save_frame(frame)
            
            return frame

    def _save_frame(self, frame):
        """Save frame to capture directory"""
        try:
            # Create capture directory if it doesn't exist
            capture_dir = getattr(self.transaction_manager.config, 'CAPTURE_DIR', 'capture')
            if not os.path.exists(capture_dir):
                os.makedirs(capture_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove last 3 digits of microseconds
            filename = f"frame_{timestamp}.jpg"
            filepath = os.path.join(capture_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame)
            print(f"Frame guardado: {filepath}")
            
        except Exception as e:
            print(f"Error guardando frame: {e}")

    def _reconnect(self):
        with self.reconnect_lock:
            print("Reconnecting...")
            self.release()
            self.connect()
            if not self.cap or not self.cap.isOpened():
                raise ConnectionError("Failed to reconnect.")

    def release(self):
        # Stop the frame reader thread
        if self.frame_thread and self.frame_thread.is_alive():
            self.stop_event.set()
            self.frame_thread.join(timeout=1.0)
            self.frame_thread = None
        
        # Clear the frame buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        # Release the capture object
        if self.cap:
            self.cap.release()
            print("Camera/video connection released.")
