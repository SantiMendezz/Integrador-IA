import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Inicializar el modelo Nano (el más liviano, optimizado para Edge AI)
model = YOLO("yolov8n.pt") 

# Clases de interés en el dataset COCO (0: person, 2: car, 3: motorcycle, 5: bus, 7: truck)
CLASES_TRANSITO = [0, 2, 3, 5, 7]

def simular_clima_adverso(imagen_path):
    """Simula degradación en el borde aplicando ruido gaussiano (niebla/lluvia)"""
    img = cv2.imread(imagen_path)
    gauss = np.random.normal(0, 40, img.shape).astype('uint8') # Ruido controlado
    img_degradada = cv2.addWeighted(img, 0.7, gauss, 0.3, 0)
    return img_degradada

def procesar_imagen_edge(imagen_origen, es_degradada=False):
    # Cargar o generar la matriz de la imagen
    if es_degradada:
        img = simular_clima_adverso(imagen_origen)
    else:
        img = cv2.imread(imagen_origen)
        
    # Medir Latencia de Inferencia (Tiempo que le toma al "poste" procesar la imagen)
    inicio = time.time()
    resultados = model(img, verbose=False)[0]
    fin = time.time()
    
    latencia_ms = (fin - inicio) * 1000
    
    # Contabilizar detecciones válidas para el alumbrado
    conteo = {"person": 0, "vehiculo": 0}
    confianzas = []
    
    for box in resultados.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id in CLASES_TRANSITO:
            confianzas.append(conf)
            if cls_id == 0:
                conteo["person"] += 1
            else:
                conteo["vehiculo"] += 1
                
    confianza_promedio = np.mean(confianzas) if confianzas else 0.0
    
    # Lógica de decisión del Agente (Control de Actuador - Dimmer)
    if conteo["person"] > 0 or conteo["vehiculo"] > 0:
        dimmer_salida = "100% (Intensidad Alta)"
    else:
        dimmer_salida = "20% (Ahorro Energético)"
        
    return latencia_ms, conteo, confianza_promedio, dimmer_salida

# Ejemplo de ejecución para un lote de imágenes en una carpeta
# (Pueden loopear sobre sus imágenes de prueba y guardar los datos en un .csv)