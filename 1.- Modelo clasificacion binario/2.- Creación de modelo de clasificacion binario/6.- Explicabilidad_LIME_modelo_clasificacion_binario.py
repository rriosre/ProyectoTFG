#-------------------------------------------------------------------------------
# Name:        Explicabilidad - Modelo clasificación binaria
# Purpose:     Utilización del método LIME para realizar la explicabilidad
#              de nuestro modelo de clasificación binaria.
#
# Author:      Raúl Ríos Redondo
#
# Created:     25/03/2026
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

    # Cargar el modelo de clasificación binaria
    model = load_model('modelo_clasificacion_binaria.h5', custom_objects={'LeakyReLU': LeakyReLU})

    # Normalizar la imagen a examinar.
    img_path = 'IMA11972_NO_TUMOR.jpg'
    img      = load_img(img_path, target_size=(512, 512))
    img_mri  = img_to_array(img) / 255.0

   # Crear el explainer de LIME para el tratamiento de una imagen
    explainer = lime_image.LimeImageExplainer()

    # Generar la explicación usando el modelo cargado
    explanation = explainer.explain_instance(
        img_mri.astype('double'),
        model.predict,
        top_labels  = 1, # Tratamiento para un modelo binario y activació sigmoid
        hide_color  = 0,
        num_samples = 1000 # Puntos de tratamiento
    )

    # Obtener la máscara de la explicabilidad de la imagen
    temp, mask = explanation.get_image_and_mask(
                                                    label = 0,            # Índice único de la clase 'Tumor/ No tumor'
                                                    positive_only =True,
                                                    num_features  =5,     # Primeras 5 zonas de búsqueda.
                                                    hide_rest     =False
                                                )

    # Configurar la visualización
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Panel izquierdo: Imagen original (procesada para el modelo)
    ax1.imshow(img_mri)
    ax1.set_title('Resonancia Original (MRI)')
    ax1.axis('off')

    # Panel derecho: Explicación de LIME
    ax2.imshow(mark_boundaries(temp, mask))
    ax2.set_title('Explicación LIME: Regiones predictoras')
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
    plt.title(f"Mapa de Calor LIME para la Clase {ind}")
    plt.axis('off')
    plt.show()
    pass

if __name__ == '__main__':
    main()
