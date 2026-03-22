#-------------------------------------------------------------------------------
# Name:        Explicabilidad - Modelo clasificación binaria
# Purpose:     Utilización del método SHAP para realizar la explicabilidad
#              de nuestro modelo de clasificación binaria.
#
# Author:      Raúl Ríos Redondo
#
# Created:     25/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers                         import LeakyReLU

import tensorflow        as tf
import numpy             as np
import shap

#-----------------------------------------------------------------------------
# Normalización de cada una de las imágenes a entrenar.
#-----------------------------------------------------------------------------
def preprc_func(img):
    img = img.astype(np.float32) / 255.0
    return img

def main():

    # Ruta de entrenamiento con 100 imagenes.
    l_directory_train = 'D:\\Copia 22-09-2025\\TFG\\2.- Modelo clasificacion binario\\2.- Creación de modelo de clasificacion binario\\3.- Datos SHAP'

    # Cargar el modelo de clasificación binaria
    model = tf.keras.models.load_model('modelo_clasificacion_binaria.h5', custom_objects={'LeakyReLU': LeakyReLU})

    # Asignamos la carpeta de entrenamiento para la técnica SHAP.
    train_datagen = ImageDataGenerator( preprocessing_function = preprc_func )

    train_generator = train_datagen.flow_from_directory(
        l_directory_train,
        target_size=(512, 512),
        batch_size=200,
        class_mode='binary',
    )

     # Creamos el background para utilizarlo con el GradienteExplainer
    x_batch,_  = next(train_generator)
    background = x_batch.astype('float32') # 200 imágenes en total


    # Normalizar la imagen a examinar.
    img_path      = 'IMA11972_NO_TUMOR.jpg'
    img           = load_img(img_path, target_size=(512, 512))
    img_mri       = img_to_array(img)
    img_mri_batch = np.expand_dims(img_mri, axis=0)
    x_test_image  = preprc_func(img_mri_batch)

    # Utilizamos GradientExplainer por ser más estable
    explainer = shap.GradientExplainer(model, background)

    # Generación de el mapa de importacnia por cada píxel.
    shap_values = explainer.shap_values(x_test_image ) #, nsamples=100)

    # Pasa de (1, 512, 512, 3) -> (512, 512, 3)
    img_mri_plot = np.squeeze(x_test_image)

    # En modelos Sigmoid, shap_values es una lista [array].
    # Debemos pasar el array interno para que coincida con la imagen.
    if isinstance(shap_values, list):
        # Quitamos dimensiones de tamaño 1 de cada elemento de la lista
        shap_values_plot = [np.squeeze(val) for val in shap_values]
    else:
        shap_values_plot = np.squeeze(shap_values)

    # Mostramos la predicción de probablidad
    print(f"Probabilidad de Tumor: {model.predict(x_test_image)[0]}")

    # Ahora el mapa de calor tendrá el mismo ancho que la original
    if not isinstance(shap_values_plot, list):
        shap_values_plot = [shap_values_plot]

    # Mostramos la imagen original y la graficada por SHAP.
    shap.image_plot(shap_values_plot, img_mri_plot)

    pass

if __name__ == '__main__':
    main()
