import os
import time
from ultralytics import YOLO

# Configuración general y rutas 
DATA_YAML = "configs/data.yaml"
TRAIN_LABEL_CACHE = "data/raw/train/labels.cache"
VAL_LABEL_CACHE = "data/raw/val/labels.cache"

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

# Entrenar todos los modelos y registrar resultados 
def train_all_models():
    logs = [] # Lista para almacenar logs de entrenamiento 
    for name, weight in MODELOS: # Iterar sobre cada modelo 
        print(f"\n🚦 Entrenando {name} ({weight})...\n")

        # Eliminar cache
        for cache_path in [TRAIN_LABEL_CACHE, VAL_LABEL_CACHE]:
            if os.path.exists(cache_path):
                os.remove(cache_path) # Eliminar cache para evitar errores 
                print(f"[INFO] Eliminado cache: {cache_path}")

        # Iniciar conteo de tiempo 
        start_time = time.time()
        model = YOLO(weight) # Cargar el modelo YOLO con los pesos especificados 
        # Entrenamiento del modelo 
        model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=0,
            project="models", # Nombre del proyecto para guardar resultados 
            name=f"{name}",
            verbose=True, # Mostrar información detallada durante el entrenamiento
            deterministic=True, # Asegurar reproducibilidad
            seed=0, # Semilla para reproducibilidad 
            val=True # Validar durante el entrenamiento 
        )
        train_time = time.time() - start_time # Calcular tiempo de entrenamiento 
        logs.append((name, weight, train_time)) # Guardar logs de entrenamiento 
        print(f"[INFO] Entrenamiento {name} finalizado en {round(train_time, 2)} segundos")
    return logs

if __name__ == "__main__":
    train_all_models()
    print("\n✅ Entrenamiento de todos los modelos finalizado.")
