import os

class Config:
    # General settings
    USERNAME = "casa1"
    PASSWORD = "casa2020"
    IP = "adminrenni.ddns.net"
    PORT = "554"
    CHANNEL = "1"
    MODEL_PATH = "./models/model_480_80_12s_140825_ncnn_model/"
    CONFIDENCE_THRESHOLD = 0.55
    CAPTURE_INTERVAL = 0.5
    CAPTURE_DIR = "capture"

    # Class names
    CASH_REGISTER_CLASS_NAME = "caja_abierta"
    RECEIPT_CLASS_NAME = "entrega_boleta"

    # Timeouts and grace periods
    CLIENT_EXIT_GRACE_PERIOD_SECONDS = 4.0
    CASHIER_EXIT_GRACE_PERIOD_SECONDS = 4.0
    IDLE_RESET_TIMEOUT_SECONDS = 30.0
    PAYMENT_COOLDOWN_SECONDS = 2.0

    # Tracking and event settings
    LOST_TRACK_BUFFER_SECONDS = 10.0
    MINIMUM_MATCHING_THRESHOLD = 0.8
    MINIMUM_CONSECUTIVE_FRAMES = 10
    EVENT_PERSISTENCE_FRAMES = 30

    # Video source
    USE_VIDEO_FILE_FOR_CAMERA = False
    VIDEO_FILE_PATH = None #"videos/completas/Renni_ch1_main_20250709170000_20250709171816.mp4" #None

    # Display settings from environment variables
    SHOW_ANNOTATIONS = True

    # Redis settings
    REDIS_URL = ""

    # Frame capture settings
    ENABLE_CAPTURE = False
