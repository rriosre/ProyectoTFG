#-------------------------------------------------------------------------------
# Name:        Modelo clasificación localización categórica de imágenes sobre
#              tumores cerebrales
# Purpose:     Red neuronal que nos identifica si una tomografía presenta un
#              tumor cerebral, de que tipo y en que lugar.
#
# Author:      Raúl Ríos Redondo
#
# Created:     30/04/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

from ultralytics import YOLO
import numpy     as np

#-------------------------------------------------------------------------------
#  Programa principal
#-------------------------------------------------------------------------------
def main():

#   1.- Obtener el modelo de YOLO11s
    model = YOLO("yolo11s.pt")


#   2.- Entrenamiento con los mejores parámetros de la optimización
    history = model.train(  data      ="config_yolo11.yaml",
                            cfg       ="best_hyperparameters.yaml",
                            epochs    =100,
                            imgsz     =512,
                            batch     =32,
                            dropout   =0.5,
                            optimizer ="AdamW",
                            patience  =20,
                            seed      =42,
                            workers   =8,
                            plots     =True,
                            save      =True,
                            val       =True )

#   3.- Grabar histórico del modelo
    np.save('history_clasificacion_localizacion_categorica.npy', history)


if __name__ == '__main__':
    main()
