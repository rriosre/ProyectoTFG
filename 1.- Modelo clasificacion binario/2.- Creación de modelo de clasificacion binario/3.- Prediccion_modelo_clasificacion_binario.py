#-------------------------------------------------------------------------------
# Name:        Predicción clasificación binaria de imágenes sobre tumores cerebrales
# Purpose:     Comprobamos el modelo de clasificación binaria creaado, con datos
#              que son de validación, ni de entrenamiento.
#
# Author:      Raúl Ríos Redondo
#
# Created:     22/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from tensorflow.keras.preprocessing import image as i
from keras.layers                   import  LeakyReLU

import tensorflow               as tf
import numpy                    as np
import seaborn                  as sns
import matplotlib.pyplot        as plt

#-----------------------------------------------------------------------------
# Normalización de cada una de las imágenes a entrenar.
#-----------------------------------------------------------------------------
def preprc_func(img):
    img = img.astype(np.float32) / 255.0
    return img

#-----------------------------------------------------------------------------
# Evaluamos el modelo entrenado con un directorio de predicción
#-----------------------------------------------------------------------------
def evaluate_model_binary(i_model, i_directorio_prediccion):

    # Crear ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator( preprocessing_function = preprc_func )

    test_generator = datagen.flow_from_directory( i_directorio_prediccion,
                                                  target_size = (512, 512),
                                                  batch_size  = 32,
                                                  class_mode  = 'binary')

    # Evaluar nuestro directorio de test
    results = i_model.evaluate(test_generator, return_dict=True)

    print('Test accuracy:',                  results["binary_accuracy"])
    print('Test loss',                       results["loss"])
    print('Test precision',                  results["precision"])
    print('Test recall',                     results["recall"])
    print('Test sensitivity_at_specificity', results["sensitivity_at_specificity"])
    print('Test specificity_at_sensitivity', results["specificity_at_sensitivity"])

    # Crear la matriz (formato estándar: [[TN, FP], [FN, TP]])
    cm = np.array([[results["tn"], results["fp"]],
                   [results["fn"], results["tp"]]])

    # Grafico
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

#-----------------------------------------------------------------------------
#  Cargamos una imagen, para su predicción.
#-----------------------------------------------------------------------------
def load_file(i_path):

    img   = i.load_img(i_path, target_size = (512, 512))
    x     = i.img_to_array(img) / 255.0
    image = x.reshape((1,) + x.shape)

    return image

#-----------------------------------------------------------------------------
# Ejecución del modelo principal.
#-----------------------------------------------------------------------------
def main():

    # Variables globales
    l_directorio_prediccion = 'C:\\Users\\kowalski333\\Desktop\\TFG\\2.- Modelo clasificacion binario\\2.- Creación de modelo de clasificacion binario\\1.- Datos tumor y no tumor\\3.- Prediccion'
    l_imagen_prediccion     = 'IMA02920_TUMOR.jpg'
    l_modelo_clasificacion  = 'modelo_clasificacion_binaria.h5'

    # Recuperar el modelo entrenado.
    model = tf.keras.models.load_model(l_modelo_clasificacion, custom_objects={'LeakyReLU': LeakyReLU})

    # Evaluación de nuestro modelo entrenado.
    evaluate_model_binary(model, l_directorio_prediccion)

    # Predecir una imagen al azar.
    image   = load_file(l_imagen_prediccion)
    classes = model.predict(image)

    print(classes) # Sacar valor de predicción

    # Determinar el tipo de imagen.
    if classes > 0.5: # Si mayor de 0.5 Tumor.
        print('Tumor')
    else:
        print('No Tumor')
    pass

if __name__ == '__main__':
    main()
