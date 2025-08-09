from ultralytics import YOLO

model = YOLO('./models/model_80n_040825.pt')

model.export(format = 'ncnn',simplify=True,dynamic=False,opset=11,imgsz=480,half=True) # Export the model to NCNN format with simplification and fixed input size


# model.export(
#     format='ncnn',
#     imgsz=640,          # Tamaño de imagen (por defecto 640)
#     half=False,         # Usar FP16 (False para FP32)
#     int8=False,         # Cuantización INT8
#     dynamic=False,      # Entrada dinámica
#     simplify=True,      # Simplificar el modelo
#     opset=11           # Versión del opset
# )