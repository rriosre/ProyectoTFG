#-------------------------------------------------------------------------------
# Name:        Explicabilidad - Modelo clasificación categórica por localización
# Purpose:     Utilización del método LIME para realizar la explicabilidad
#              de nuestro modelo de clasificación categórica por localización.
#
# Author:      Raúl Ríos Redondo
#
# Created:     10/04/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from lime                                 import lime_image
from skimage.segmentation                 import mark_boundaries
from ultralytics                          import YOLO

import matplotlib.pyplot as plt
import numpy as np
import cv2

def main():

    # Cargar el modelo de clasificación categorica
    model = YOLO('modelo_clasificacion_localizacion_categorica.pt')

    # Nombres de las clases para referencia
    class_names = {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary' , 3: 'No tumor'}

    # Cargar la imagen a analizar.
    img_path    = 'IMA18372_HIPOFISARIO.jpg'
    img_bgr     = cv2.imread(img_path )
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    # Función utilizada, para predicir una imagen en YOLO
    def predict_fn(images):
        # LIME envía una lista de imágenes (numpy arrays)
        images_uint8 = [img.astype(np.uint8) for img in images]
        results = model.predict(images_uint8, verbose=False)

        # Extraemos las probabilidades de las 4 clases: 0, 1, 2, 3
        all_probs = []
        for res in results:

            # Inicializamos vector de 4 clases en 0
            prob_vector = np.zeros(len(class_names))

             # Si el modelo detecta al menos una caja (zona afectada)
            if len(res.boxes) > 0:

                # Tomamos la detección con mayor confianza
                best_box = res.boxes[0]
                cls_idx  = int(best_box.cls.item())
                conf     = float(best_box.conf.item())

                # Asignamos la confianza a la clase detectada
                if cls_idx < len(class_names):
                    prob_vector[cls_idx] = conf
            else:
                # Si no detecta nada, asignamos confianza a "No tumor" (clase 2)
                prob_vector[2] = 0.5

            all_probs.append(prob_vector)

        return np.array(all_probs)

    # Crear el explainer de LIME para el tratamiento de una imagen
    explainer = lime_image.LimeImageExplainer()

    # Generar la explicación usando el modelo cargado
    explanation = explainer.explain_instance(
        img_resized,
        predict_fn,
        top_labels  = 4,   # Tratamiento para un modelo categórico
        hide_color  = 0,
        num_samples = 1000 # Puntos de tratamiento
    )

    # Obtener predicción real para el título
    res_final = model(img_resized)[0]
    if len(res_final.boxes) > 0:
        clase_id = int(res_final.boxes[0].cls.item())
        conf     = float(res_final.boxes[0].conf.item())
    else:
        clase_id = 2
        conf     = 0.0

   # Obtener la máscara de la explicabilidad de la imagen
    temp, mask = explanation.get_image_and_mask(
                                                 label=clase_id,         # Clase predicha
                                                 positive_only =True,
                                                 num_features  =5,       # Primeras 5 zonas de búsqueda.
                                                 hide_rest     =False
                                                )

    # Visualización
    plt.figure(figsize=(15, 5))

    # Imagen Original con la caja de YOLO (si existe)
    plt.subplot(1, 3, 1)
    img_with_box = img_resized.copy()
    if len(res_final.boxes) > 0:
        box = res_final.boxes[0].xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(img_with_box, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 3)
    plt.imshow(img_with_box)
    plt.title(f"Detección YOLO\n{class_names[clase_id]} ({conf:.2%})")
    plt.axis('off')

    # Explicación LIME
    plt.subplot(1, 3, 2)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME: Zonas que justifican la detección")
    plt.axis('off')

    # Mapa de Calor (Heatmap)
    plt.subplot(1, 3, 3)
    dict_heatmap = dict(explanation.local_exp[clase_id])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(img_resized)
    plt.imshow(heatmap, alpha=0.6, cmap='RdYlGn')
    plt.title("Mapa de Calor de Relevancia")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Visualizar el mejor resultado
    res_final = model(img_resized)[0]  # Tomamos el primer resultado del lote

    if len(res_final.boxes) > 0:

        # Extraemos datos de la detección.
        box          = res_final.boxes[0]   # La primera es la de mayor confianza
        clase_idx    = int(box.cls.item())
        nombre_clase = class_names[clase_idx]
        confianza    = float(box.conf.item())

        # Coordenadas de la caja [x1, y1, x2, y2]
        coords = box.xyxy.cpu().numpy()[0].astype(int)

        print("-" * 50)
        print("RESULTADO DE LA PREDICCIÓN:")
        print(f"  - Clase detectada: {nombre_clase}")
        print(f"  - Confianza:       {confianza:.2%}")
        print(f"  - Ubicación (Box): [x1:{coords[0]}, y1:{coords[1]}, x2:{coords[2]}, y2:{coords[3]}]")
        print("-" * 50)
    else:
        print("-" * 50)
        print("RESULTADO DE LA PREDICCIÓN: No se detectó ninguna zona afectada (No tumor).")
        print("-" * 50)
        pass

if __name__ == '__main__':
    main()
