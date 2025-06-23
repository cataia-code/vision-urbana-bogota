import os
import pandas as pd
from ultralytics import YOLO

# Configuración general 
DATA_YAML = "configs/data.yaml"
RESULTS_CSV = "resultados_yolos.csv"

# Modelos a entrenar 
MODELOS = [
    # YOLOv8
    ("yolov8n", "yolov8n.pt"),
    ("yolov8m", "yolov8m.pt"),
    ("yolov8s", "yolov8s.pt"),
    # YOLOv9
    ("yolov9m", "yolov9m.pt"),
    ("yolov9s", "yolov9s.pt"),
    # YOLOv10
    ("yolov10n", "yolov10n.pt"),
    ("yolov10m", "yolov10m.pt"),
    ("yolov10s", "yolov10s.pt"),
    # YOLOv11
    ("yolov11n", "yolo11n.pt"),
    ("yolov11m", "yolo11m.pt"),
    ("yolov11s", "yolo11s.pt"),
] 

# Parámetros de entrenamiento
EPOCHS = 120 # Número de épocas 
BATCH = 16 # Tamaño del batch 
IMGSZ = 640 # Tamaño de imagen 

# DataFrame para resultados 
cols = [
    "modelo", "pesos", "epochs", "batch", "imgsz",
    "mAP50-95", "mAP50", "precision", "recall", "f1",
    "fps", "latency_ms"
]
resultados = []

# Validar todos los modelos y registrar resultados 
for name, weight in MODELOS:
    print(f"\n🔎 Validando {name} ({weight})...\n")
    # Verificar si el archivo de pesos existe 
    model = YOLO(weight)
    # Validación del modelo 
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0, # Usar GPU 0 
        verbose=True # Mostrar información detallada durante la validación 
    )
    try:
        # Extraer métricas de validación 
        mAP50_95 = val_results.box.map # mAP50-95 
        mAP50 = val_results.box.map50 # mAP50
        precision = val_results.box.mp # Precisión 
        recall = val_results.box.mr # Recall 
        f1 = val_results.box.mf1 if hasattr(val_results.box, "mf1") else None # F1 score 
        fps = None # Frames por segundo (FPS) 
        latency = None # Latencia en milisegundos 
        # Extraer latencia si está disponible 
        if hasattr(val_results, "speed"):
            latency = val_results.speed.get("inference", None) # Latencia de inferencia en milisegundos 
            if latency:
                # Calcular FPS a partir de la latencia 
                fps = 1000 / latency if latency > 0 else None
    except Exception as e:
        print(f"[WARNING] No se pudieron extraer todas las métricas: {e}")
        mAP50_95 = mAP50 = precision = recall = f1 = fps = latency = None

    # Guardar resultados en el DataFrame 
    resultados.append([
        name, weight, EPOCHS, BATCH, IMGSZ,
        mAP50_95, mAP50, precision, recall, f1, fps, latency
    ])
    pd.DataFrame(resultados, columns=cols).to_csv(RESULTS_CSV, index=False)

print("\n✅ Validación de todos los modelos finalizada. Resultados guardados en", RESULTS_CSV)
