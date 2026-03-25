#-------------------------------------------------------------------------------
# Name:        Explicabilidad - Modelo clasificación categórica
# Purpose:     Utilización del método LIME para realizar la explicabilidad
#              de nuestro modelo de clasificación categórica.
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
from tensorflow.keras.models              import load_model
from keras.layers                         import LeakyReLU
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

def main():

    # Cargar el modelo de clasificación categorica
    model = load_model('modelo_clasificacion_categorica.h5', custom_objects={'LeakyReLU': LeakyReLU})

    # Nombres de las clases para referencia
    class_names = {0: 'Glioma', 1: 'Meningioma', 2: 'No tumor', 3: 'Pituitary'}

    # Normalizar la imagen a examinar.
    img_path = 'IMA18372_HIPOFISARIO.jpg'
    img      = load_img(img_path, target_size=(512, 512))
    img_mri  = img_to_array(img) / 255.0

   # Crear el explainer de LIME para el tratamiento de una imagen
    explainer = lime_image.LimeImageExplainer()

    # Generar la explicación usando el modelo cargado
    explanation = explainer.explain_instance(
        img_mri.astype('double'),
        model.predict,
        top_labels  = 4, # Tratamiento para un modelo categórico
        hide_color  = 0,
        num_samples = 1000 # Puntos de tratamiento
    )

    # Determinar la clase que predijo el modelo.
    preds = model.predict(np.expand_dims(img_mri, axis=0))
    predicted_class_idx = np.argmax(preds[0])
    print(f"Predicción del modelo: {class_names[predicted_class_idx]} ({preds[0][predicted_class_idx]:.2%})")

    # Obtener la máscara de la explicabilidad de la imagen
    temp, mask = explanation.get_image_and_mask(
                                                    label=predicted_class_idx,  # Clase predicha
                                                    positive_only =True,
                                                    num_features  =5,           # Primeras 5 zonas de búsqueda.
                                                    hide_rest     =False
                                                )

    # Configurar la visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Panel izquierdo: Imagen original (procesada para el modelo)
    ax1.imshow(img_mri)
    ax1.set_title(f'MRI Original\nPredicho: {class_names[predicted_class_idx]}')
    ax1.axis('off')

    # Panel derecho: Explicación de LIME
    ax2.imshow(mark_boundaries(temp, mask))
    ax2.set_title(f'LIME: Regiones que apoyan a {class_names[predicted_class_idx]}')
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    #------------------------------------------------------------------
    #  Obtener mapa de calor de la explicabilidad con LIMe.
    #------------------------------------------------------------------

    # Obtener la explicación para la clase principal
    ind = explanation.top_labels[0]
    dict_heatmap = explanation.local_exp[ind]

    # Crear una matriz vacía del tamaño de la segmentación de LIME
    heatmap = np.zeros(explanation.segments.shape)

    # Llenar la matriz con la importancia (peso) de cada región
    for feature, weight in dict_heatmap:
        heatmap[explanation.segments == feature] = weight

    # Visualización del mapa de calor
    plt.figure(figsize=(8, 8))
    plt.imshow(img_mri)                            # Imagen original de fondo
    plt.imshow(heatmap, alpha=0.6, cmap='RdYlGn')  # Heatmap encima con transparencia
    plt.colorbar(label="Importancia para el diagnóstico")
    plt.title(f"Mapa de Calor LIME para: {class_names[ind]}")
    plt.axis('off')
    plt.show()
    pass

if __name__ == '__main__':
    main()
