import os
import time
import pandas as pd
from ultralytics import YOLO

# === Configuraciones ===
DATA_YAML = "configs/data.yaml"
RESULTS_CSV = "resultados_yolos.csv"

# Modelos a entrenar (puedes ajustar la lista si quieres menos o más variantes)
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

# Rutas de cache de labels
TRAIN_LABEL_CACHE = "data/raw/train/labels.cache"
VAL_LABEL_CACHE = "data/raw/val/labels.cache"

# Parámetros de entrenamiento (ajusta según necesidad)
EPOCHS = 120
BATCH = 16
IMGSZ = 640

# DataFrame para resultados
cols = [
    "modelo", "pesos", "epochs", "batch", "imgsz",
    "tiempo_train_s", "mAP50-95", "mAP50", "precision", "recall", "f1",
    "fps", "latency_ms"
]
resultados = []

for name, weight in MODELOS:
    print(f"\n🚦 Entrenando {name} ({weight})...\n")

    # Eliminar cache para evitar errores
    for cache_path in [TRAIN_LABEL_CACHE, VAL_LABEL_CACHE]:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"[INFO] Eliminado cache: {cache_path}")

    # Iniciar conteo de tiempo
    start_time = time.time()
    model = YOLO(weight)
    # Entrenamiento
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,  # Usa GPU si está disponible, o quita device para automático
        project="runs_compare_all",
        name=f"{name}",
        verbose=True,
        deterministic=True,
        seed=0,
        val=True
    )
    tiempo_train = time.time() - start_time

    # Validación y métricas
    val_results = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        verbose=True
    )

    # Extracción de métricas
    try:
        # mAP
        mAP50_95 = val_results.box.map  # mAP 0.5:0.95
        mAP50 = val_results.box.map50
        # Precisión, Recall, F1
        precision = val_results.box.mp
        recall = val_results.box.mr
        f1 = val_results.box.mf1 if hasattr(val_results.box, "mf1") else None
        # FPS y Latencia (puede variar según versión, usa .speed si está)
        fps = None
        latency = None
        if hasattr(val_results, "speed"):
            latency = val_results.speed.get("inference", None)
            if latency:
                fps = 1000 / latency if latency > 0 else None
    except Exception as e:
        print(f"[WARNING] No se pudieron extraer todas las métricas: {e}")
        mAP50_95 = mAP50 = precision = recall = f1 = fps = latency = None

    # Guardar resultados
    resultados.append([
        name, weight, EPOCHS, BATCH, IMGSZ,
        round(tiempo_train, 2),
        mAP50_95, mAP50, precision, recall, f1, fps, latency
    ])

    # Guardar el avance después de cada modelo por seguridad
    pd.DataFrame(resultados, columns=cols).to_csv(RESULTS_CSV, index=False)

print("\n✅ Entrenamiento de todos los modelos finalizado. Resultados guardados en", RESULTS_CSV)