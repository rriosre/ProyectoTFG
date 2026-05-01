#-------------------------------------------------------------------------------
# Name:        Explicabilidad - Modelo clasificación categorico por localización
# Purpose:     Utilización del método SHAP para realizar la explicabilidad
#              de nuestro modelo de clasificación categorico por localización.
#
# Author:      Raúl Ríos Redondo
#
# Created:     30/04/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from ultralytics import YOLO

import shap
import numpy as np
import cv2

def main():

    img_path = 'IMA11954_NO_TUMOR.jpg'
    clases   = ['Glioma', 'Meningioma', 'Hipofisario', 'No tumor',  ]

    # Cargar modelo
    model    = YOLO('modelo_clasificacion_localizacion_categorica.pt')

    # Redimensionar al tamaño de entrada del modelo a (224x224)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (224, 224)), cv2.COLOR_BGR2RGB)

    # Función de predicción para SHAP adaptada para DETECCIÓN
    def predict_func(images):
        final_preds = []
        for img in images:

            # YOLO espera valores de 0-255 en uint8
            results = model.predict(img, verbose=False, conf=0.1)[0]

            # Inicializamos un vector de ceros por cada clase
            prob_vector = np.zeros(len(clases))

            # Si hay detecciones, extraemos la confianza de la mejor caja por clase
            if len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    # Guardamos la confianza más alta encontrada para esta clase
                    if conf > prob_vector[cls_id]:
                        prob_vector[cls_id] = conf

            final_preds.append(prob_vector)

        return np.array(final_preds)

    # Obtener la detección principal para la visualización
    res_inicial = model.predict(img_rgb, verbose=False)[0]

    if len(res_inicial.boxes) > 0:
        idx_predicha = int(res_inicial.boxes.cls[0].item())
        nombre_predicha = clases[idx_predicha]
        print(f"Detección principal: {nombre_predicha} (Conf: {res_inicial.boxes.conf[0]:.2f})")
    else:
        print("No se detectó nada inicialmente. Usando clase 0 por defecto para el mapa SHAP.")
        idx_predicha = 0
        nombre_predicha = clases[0]

    # Configurar SHAP
    masker = shap.maskers.Image("blur(128,128)", img_rgb.shape)

    # El explainer vincula la función de predicción con el masker
    explainer = shap.Explainer(predict_func, masker, output_names=clases)

    # Calcular SHAP
    print(f"Calculando SHAP para la clase: {nombre_predicha}...")
    shap_values = explainer(np.array([img_rgb]), max_evals=1000, batch_size=1)

    # Visualizar la imagen original y la producida por SHAP
    clase_shap_array = shap_values.values[:, :, :, :, idx_predicha]

    # Extraemos la imagen original
    imagen_array = np.array([img_rgb])

    # Generar el gráfico
    print("Generando gráfico...")
    shap.image_plot(
        clase_shap_array,
        imagen_array.astype(float) / 255.0, # Normalizado a 0-1
        labels=[[nombre_predicha]]
    )

    pass

if __name__ == '__main__':
    main()
