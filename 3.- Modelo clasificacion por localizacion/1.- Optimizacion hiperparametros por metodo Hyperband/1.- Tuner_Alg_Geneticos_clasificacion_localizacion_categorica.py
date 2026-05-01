#-------------------------------------------------------------------------------
# Name:        Modelo clasificación localización categórica de imágenes sobre
#              tumores cerebrales
# Purpose:     Red neuronal que nos identifica si una resonancia presenta un
#              tumor cerebral, de que tipo y en que lugar.
#
# Author:      Raúl Ríos Redondo
#
# Created:     24/04/206
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

from ultralytics          import YOLO

#-----------------------------------------------------------------------------
# Generación de la arquitectura YOLO 11 para el tratamiento de las
# imágenes tumorales mediante detección. Retornamos el modelo creado.
#-----------------------------------------------------------------------------
def create_model():

  # 1.- Obtener el modelo small de YOLO11s
    model = YOLO("yolo11s.pt")


  # 2. Definir el espacio de búsqueda (Hiperparámetros a ajustar)
    espacio_busqueda = {
            "lr0": (1e-5, 1e-1),          # Tasa de aprendizaje inicial
            "lrf": (0.01, 1.0),           # Factor de tasa de aprendizaje final
            "momentum": (0.6, 0.98),      # Momentum del optimizador
            "weight_decay": (0.0, 0.001), # Regularización
            "warmup_epochs": (0.0, 5.0),  # Épocas de calentamiento
            "warmup_momentum": (0.0, 0.95),
            "warmup_bias_lr": (0.0, 0.1),

            # --- Ganancias de Pérdida (Loss Gains) ---
            "box": (0.05, 0.2),           # Peso de la pérdida de caja delimitadora
            "cls": (0.2, 4.0),            # Peso de la pérdida de clasificación
            "dfl": (0.4, 2.0),            # Peso de la pérdida Distribution Focal Loss

    }


#   3.- Ajueste de Hiperparámetros por Algoritmos Genéticos (Genetic Algorithms - GA)
    results = model.tune(     data             ="config_yolo11.yaml",
                              epochs           =10, #100,
                              dropout          =0.5,
                              patience         =5,
                              seed             =42,
                              imgsz            =512,
                              batch            =32,
                              iterations       =100,
                              workers          =8,   #Ryzen 7
                              resume           =True,
                              plots            =True,
                              save             =True,
                              val              =True,
                              space            =espacio_busqueda)

    # 4. Los resultados y mejores hiperparámetros.
    print("Mejores hiperparámetros:", results )


    return model


#-------------------------------------------------------------------------------
#  Programa principal
#-------------------------------------------------------------------------------
def main():

#   Ejecutar la optimización del modelo.
    create_model()

if __name__ == '__main__':
    main()
