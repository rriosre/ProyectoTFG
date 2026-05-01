#-------------------------------------------------------------------------------
# Name:        Modelo clasificación localización categórica de imágenes sobre
#              tumores cerebrales
# Purpose:     Red neuronal que nos identifica si una tomografía presenta un
#              tumor cerebral, de que tipo y en que lugar.
#
# Author:      Raúl Ríos Redondo
#
# Created:     05/05/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

from ultralytics import YOLO
import cv2       as cv2


#-------------------------------------------------------------------------------
#  Programa principal
#-------------------------------------------------------------------------------
def main():

#-------------------------------------------------------------------------------
#  Seleccionar el modelo creado
#-------------------------------------------------------------------------------
    model   = YOLO("modelo_clasificacion_localizacion_categorica.pt")

#-------------------------------------------------------------------------------
#  Evaluar el directorio de predicción
#-------------------------------------------------------------------------------
    results= model.val(
            data='config_yolo11_prediccion.yaml', # Archivo YAML con las rutas
            split='test',                         # Indica que use el set de validación
            imgsz=512,                            # Tamaño de imagen (debe ser el mismo del entrenamiento)
            batch=16                              # Tamaño del lote
    )

#-------------------------------------------------------------------------------
#  Mostrar las métricas.
#-------------------------------------------------------------------------------

    for i, name in results.names.items():
        precision  = results.box.p[i]      # Precisión por clase
        recall     = results.box.r[i]      # Recall por clase
        ap50       = results.box.ap50[i]   # mAP@0.5 por clase

        print(f"Clase {i} ({name}):")
        print(f"  - Precisión: {precision:.4f}")
        print(f"  - Recall:    {recall:.4f}")
        print(f"  - mAP@50:    {ap50:.4f}")
        print("-" * 30)

    # Resumen general (mAP global)
    print(f"mAP50-95 Global: {results.box.map:.4f}")

#-------------------------------------------------------------------------------
#  Predecir una imagen
#-------------------------------------------------------------------------------
##    results = model.predict(source="Tr-pi_0363.jpg",
##                            show=False,
##                            conf=0.10,
##                            save=False,
##                            show_labels = True,
##                            show_boxes  = True)
##
##    # Process results list
##    for result in results:
##        boxes     = result.boxes  # Boxes object for bounding box outputs
##        masks     = result.masks  # Masks object for segmentation masks outputs
##        keypoints = result.keypoints  # Keypoints object for pose outputs
##        probs     = result.probs  # Probs object for classification outputs
##        obb       = result.obb      # Oriented boxes object for OBB outputs
##        result.show()         # display to screen
##        print('verbose',result.verbose())
##        result.save(filename="result.jpg")  # save to disk
##        cv2.waitKey(0)
    pass

if __name__ == '__main__':
    main()
