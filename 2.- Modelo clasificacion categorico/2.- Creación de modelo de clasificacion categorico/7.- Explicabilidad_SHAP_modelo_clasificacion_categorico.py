#-------------------------------------------------------------------------------
# Name:        Explicabilidad - Modelo clasificación categorico
# Purpose:     Utilización del método SHAP para realizar la explicabilidad
#              de nuestro modelo de clasificación categorico.
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

    # Clases de tumores
    class_names = {0: 'Glioma', 1: 'Meningioma', 2: 'No tumor', 3: 'Tumor hipofisario'}

    # Ruta de entrenamiento con 100 imagenes.
    l_directory_train = 'C:\\Users\\kowalski333\\Desktop\\TFG\\3.- Modelo clasificacion categorico\\2.- Creación de modelo de clasificacion categorico\\3.- Datos SHAP'

    # Cargar el modelo de clasificación binaria
    model = tf.keras.models.load_model('modelo_clasificacion_categorica.h5', custom_objects={'LeakyReLU': LeakyReLU})

    # Asignamos la carpeta de entrenamiento para la técnica SHAP.
    train_datagen = ImageDataGenerator( preprocessing_function = preprc_func )

    train_generator = train_datagen.flow_from_directory(
        l_directory_train,
        target_size=(512, 512),
        batch_size=400,
        class_mode='categorical',
    )

     # Creamos el background para utilizarlo con el GradienteExplainer
    x_batch,_  = next(train_generator)
    background = x_batch.astype('float32') # 400 imágenes en total


    # Normalizar la imagen a examinar.
    img_path      = 'IMA18372_HIPOFISARIO.jpg'
    img           = load_img(img_path, target_size=(512, 512))
    img_mri       = img_to_array(img)
    img_mri_batch = np.expand_dims(img_mri, axis=0)
    x_test_image  = preprc_func(img_mri_batch)

    # Utilizamos GradientExplainer por ser más estable
    explainer = shap.GradientExplainer(model, background)

    # Generación de el mapa de importacnia por cada píxel.
    shap_values = explainer.shap_values(x_test_image ) #, nsamples=100)

    # Predicción del modelo
    preds = model.predict(x_test_image)

    # Obtener índice de la clase con mayor probabilidad
    predicted_class_idx = np.argmax(preds[0])

    # Mostramos la predicción de probablidad
    print("-" * 30)
    print(f"Predicción final: {class_names[predicted_class_idx]}")
    for i, prob in enumerate(preds[0]):
        print(f"Probabilidad {class_names[i]}: {prob:.4f}")
    print("-" * 30)

    # Pasa de (1, 512, 512, 3) -> (512, 512, 3)
    img_mri_plot = np.squeeze(x_test_image)

    # Cada elemento de la lista debe tener forma (H, W, C)
    if isinstance(shap_values, list):
        shap_values_plot = [np.squeeze(val) for val in shap_values]
    else:
        # Si SHAP devuelve un array de 4 dimensiones (N, H, W, C, Canales_Salida)
        shap_values_plot = [shap_values[0, :, :, :, i] for i in range(shap_values.shape[-1])]

    # Creamos una lista que solo contenga el array de la clase ganadora
    shap_values_ganador = [shap_values_plot[predicted_class_idx]]

    # Mostramos la imagen original y la graficada por SHAP.
    shap.image_plot(shap_values_ganador, img_mri_plot)

    pass

if __name__ == '__main__':
    main()
