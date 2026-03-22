#-------------------------------------------------------------------------------
# Name:        Métricas - Modelo clasificación binaria
# Purpose:     Visualizamos los datos de métrica y gráficos del modelo de
#              clasificación binaria.
#
# Author:      Raúl Ríos Redondo
#
# Created:     23/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
import numpy as np

def main():

    # Leer fichero con el histórico de entrenamiento
    history = np.load('history_clasificacion_binaria.npy', allow_pickle=True).item()

    # Obtener objeto hitory
    h = history.history

    # Obtener el número total de épocas
    num_epocas = len(h['loss'])

    # Mostrar título
    print(f"{'Epoch':<13} | {'Loss':<6} | {'Acc':<7} | {'Recall':<6} | {'Prec':<6}")
    print("-" * 80)

    # Simulamos la salida del entrenamiento
    for i in range(num_epocas):

        # Extraemos valores de la época actual
        loss  = h.get('loss', [0])[i]
        acc   = h.get('binary_accuracy', h.get('acc', [0]))[i]
        rec   = h.get('recall', [0])[i]
        prec  = h.get('precision', [0])[i]


        # Imprimir fila con formato
        print(f"Epoch {i+1:03d}/{num_epocas:03d} | "
              f"{loss:.4f} | {acc:.4f}  | {rec:.4f} | {prec:.4f}")

    # Mensaje final.
    print("-" * 80)
    print("Lectura de historial completada.")

    pass

if __name__ == '__main__':
    main()
