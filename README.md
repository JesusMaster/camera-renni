# Sistema de Detección para Cajas Registradoras

Este proyecto es un sistema de visión por computadora diseñado para monitorear las interacciones entre cajeros y clientes en un entorno de punto de venta. Utiliza un modelo de detección de objetos YOLO para analizar un flujo de video en tiempo real, identificar eventos clave y gestionar transacciones.

## Características

- **Detección de Objetos en Tiempo Real:** Utiliza un modelo YOLO para detectar objetos como "caja_abierta" y "entrega_boleta".
- **Gestión de Zonas:** Define zonas de interés en el campo de visión de la cámara para monitorear actividades específicas.
- **Monitoreo de Caja Registradora:** Supervisa el estado de la caja registradora (abierta/cerrada).
- **Gestión de Transacciones:** Realiza un seguimiento del ciclo de vida de una transacción, desde que se abre la caja hasta que se entrega el recibo.
- **Anotación de Video:** Muestra el flujo de video con anotaciones en tiempo real, incluyendo cuadros delimitadores, etiquetas y el estado actual de la transacción.
- **Configuración Flexible:** Permite una fácil configuración de los parámetros de la cámara, el modelo y la aplicación a través de un archivo de configuración.
- **Soporte para Video en Vivo y Archivos:** Puede procesar tanto una transmisión de video en vivo desde una cámara IP como un archivo de video pregrabado.

## Estructura del Proyecto

```
/
├── capture/              # Directorio para guardar capturas de video.
├── models/               # Contiene los modelos de detección de objetos.
├── src/                  # Código fuente de la aplicación.
│   ├── app.py            # Lógica principal y fábrica de la aplicación.
│   ├── camera.py         # Gestiona el flujo de video de la cámara.
│   ├── config.py         # Archivo de configuración.
│   ├── detector.py       # Encapsula el detector de objetos YOLO.
│   ├── display.py        # Se encarga de las anotaciones en los fotogramas.
│   ├── frame_processor.py # Procesa cada fotograma del video.
│   ├── monitor.py        # Monitorea la caja registradora.
│   ├── transaction_manager.py # Gestiona la lógica de las transacciones.
│   └── zones.py          # Gestiona las zonas de detección.
├── videos/               # Directorio para archivos de video de entrada.
├── main.py               # Punto de entrada de la aplicación.
├── README.md
└── requirements.txt      # Dependencias del proyecto.
```

## Instalación

1.  **Clonar el repositorio:**
    ```bash
    git clone git@github.com:JesusMaster/camera-renni.git
    cd camera-renni
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows, usa `venv\Scripts\activate`
    ```

3.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuración

El comportamiento de la aplicación se puede configurar modificando el archivo `src/config.py`. A continuación se describen las principales opciones de configuración:

-   **`USERNAME`, `PASSWORD`, `IP`, `PORT`, `CHANNEL`:** Credenciales y dirección de la cámara IP.
-   **`MODEL_PATH`:** Ruta al modelo de detección de objetos YOLO.
-   **`CONFIDENCE_THRESHOLD`:** Umbral de confianza para las detecciones.
-   **`CAPTURE_DIR`:** Directorio para guardar las capturas de video.
-   **`CASH_REGISTER_CLASS_NAME`:** Nombre de la clase que representa una caja registradora abierta.
-   **`RECEIPT_CLASS_NAME`:** Nombre de la clase que representa la entrega de un recibo.
-   **`USE_VIDEO_FILE_FOR_CAMERA`:** Poner a `True` para usar un archivo de video en lugar de una cámara en vivo.
-   **`VIDEO_FILE_PATH`:** Ruta al archivo de video que se utilizará si `USE_VIDEO_FILE_FOR_CAMERA` es `True`.
-   **`REDIS_URL`:** URL para la conexión a la base de datos Redis.

## Uso

Para ejecutar la aplicación, simplemente corra el siguiente comando desde el directorio raíz del proyecto:

```bash
python main.py
```

La aplicación se iniciará, se conectará a la fuente de video y comenzará a procesar los fotogramas. Se mostrará una ventana de OpenCV con el video anotado. Para detener la aplicación, presione la tecla 'q'.

## Dependencias

-   torch
-   opencv-python
-   ultralytics
-   supervision[tracker]
-   numpy
-   redis
-   flatbuffers
-   humanfriendly
-   ncnn
-   onnx
-   onnxruntime
-   onnxslim
-   protobuf
