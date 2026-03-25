#-------------------------------------------------------------------------------
# Name:        Predicción clasificación categórica de imágenes sobre tumores cerebrales
# Purpose:     Comprobamos el modelo de clasificación categórico creado, con datos
#              que no son de validación, ni de entrenamiento.
#
# Author:      Raúl Ríos Redondo
#
# Created:     29/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from tensorflow.keras.preprocessing import image as i
from sklearn.metrics                import confusion_matrix, ConfusionMatrixDisplay
from keras.layers                   import  LeakyReLU
from sklearn.metrics                import classification_report

import matplotlib.pyplot        as plt
import tensorflow               as tf
import numpy                    as np

#-----------------------------------------------------------------------------
# Normalización de cada una de las imágenes a entrenar.
#-----------------------------------------------------------------------------
def preprc_func(img):
    img = img.astype(np.float32) / 255.0
    return img

#-----------------------------------------------------------------------------
# Evaluamos el modelo entrenado con un directorio de predicción
#-----------------------------------------------------------------------------
def evaluate_model_category(i_model, i_directorio_prediccion):

    # Crear ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator( preprocessing_function = preprc_func )

    test_generator = datagen.flow_from_directory(
                                                i_directorio_prediccion,
                                                target_size =(512, 512),
                                                batch_size  =32,
                                                class_mode  ='categorical',
                                                shuffle     =False)

    # Realizar predicciones
    predictions = i_model.predict(test_generator)

    # Obtener etiquetas reales y nombres de clases
    y_true       = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    y_pred       = np.argmax(predictions, axis=1)     # Obtener el índice de la clase con mayor probabilidad

    # Calcular y graficar la Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Greens, xticks_rotation=45)
    plt.title('Matriz de Confusión: Clasificación de Tumores Cerebrales')
    plt.show()

    # Imprime un reporte completo: Precision, Recall, F1 y Support por clase
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_labels))


#-----------------------------------------------------------------------------
#  Cargamos una imagen, para su predicción.
#-----------------------------------------------------------------------------
def load_file(i_path):
    img = i.load_img(i_path, target_size = (512, 512))
    x   = i.img_to_array(img) / 255.0
    image = x.reshape((1,) + x.shape)

    return image

#-----------------------------------------------------------------------------
# Ejecución del modelo principal.
#-----------------------------------------------------------------------------
def main():

    # Variables globales
    l_directorio_prediccion = 'C:\\Users\\kowalski333\\Desktop\\TFG\\3.- Modelo clasificacion categorico\\2.- Creación de modelo de clasificacion categorico\\1.- Datos categorico\\3.- Prediccion'
    l_imagen_prediccion     = 'IMA18333.jpg'
    l_modelo_clasificacion  = 'modelo_clasificacion_categorica.h5'

    # Recuperar el modelo entrenado.
    model = tf.keras.models.load_model(l_modelo_clasificacion, custom_objects={'LeakyReLU': LeakyReLU})

    # Evaluación de nuestro modelo entrenado.
#    evaluate_model_category(model, l_directorio_prediccion)

    # Predecir una imagen al azar.
    image   = load_file(l_imagen_prediccion)
    classes = model.predict(image)

    print(classes) # Sacar valor de predicción

    l_num_class = np.argmax(classes)
    l_valor     = np.max(classes)
    print ('Valor numérico:',{l_valor})
    # Determinar el tipo de tumor.
    match l_num_class:
        case 0:
            print("Tumor Glioma")
        case 1:
            print("Tumor Meningioma")
        case 2:
            print("No Tumor")
        case 3:
            print("Tumor Hipofisario")
    pass

if __name__ == '__main__':
    main()
