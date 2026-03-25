#-------------------------------------------------------------------------------
# Name:        Optimización para el modelo clasificación categórica de imágenes
#              sobre tumores cerebrales
# Purpose:     Se compone del proceso de optimización mediante "Hyperband",
#              de la arquitectura para clasificación categórica de los tumores
#              cerebrales (gliomas, meningiomas y tumores hipofisarios)
#
# Author:      Raúl Ríos Redondo
#
# Created:     19/03/206
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from contextlib   import redirect_stdout

import keras_tuner as kt
import tensorflow  as tf
import numpy       as np

#-----------------------------------------------------------------------------
# Normalización de cada una de las imágenes a entrenar.
#-----------------------------------------------------------------------------
def preprc_func(img):
    img = img.astype(np.float32) / 255.0
    return img

#-----------------------------------------------------------------------------
# Generación de los objetos de entrenamiento y validación. Retornamos estos.
#-----------------------------------------------------------------------------
def load_data(i_directory_train, i_directory_test):

#   Objeto de preprocesamiento.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator( preprocessing_function = preprc_func )

#   Datos de entrenamiento
    train_generator = datagen.flow_from_directory(
            i_directory_train,
            target_size =(512, 512),
            batch_size  =32,
            shuffle     =True,
            seed        = 10,
            class_mode  ='categorical')

#   Datos de validación
    test_generator = datagen.flow_from_directory(
            i_directory_test,
            target_size =(512, 512),
            batch_size  =32,
            shuffle     =True,
            seed        = 10,
            class_mode  ='categorical')

    return train_generator, test_generator

#-----------------------------------------------------------------------------
# Generación de la arquitectura del modelo CNN para el tratamiento de las
# imágenes tumorales, con los hyperparámetros a optimizar.
# Retornamos el modelo creado.
#-----------------------------------------------------------------------------
def create_model(hp):

    input_shape = (512,512, 3) # Imagen 512x512 a color

    # Activaciones
    activations_map = {
        'relu': 'relu',
        'leaky': tf.keras.layers.LeakyReLU(alpha=0.01)
    }

#   Arquitectura clasificacion categórica de tumores cerebrales
    model = Sequential()

    model.add(Conv2D(filters     = hp.Choice('conv_1_filters', values=[32, 64]),
                     kernel_size = hp.Choice('conv_1_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     input_shape = input_shape,
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_1_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('Dropout_1', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Conv2D(filters     = hp.Choice('conv_2_filters', values=[32, 64]),
                     kernel_size = hp.Choice('conv_2_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_2_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hp.Choice('Dropout_2', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Conv2D(filters     = hp.Choice('conv_3_filters', values=[128, 256, 512]),
                     kernel_size = hp.Choice('conv_3_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_3_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('Dropout_3', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Conv2D(filters     = hp.Choice('conv_4_filters', values=[128, 256, 512]),
                     kernel_size = hp.Choice('conv_4_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_4_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hp.Choice('Dropout_4', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Conv2D(filters     = hp.Choice('conv_5_filters', values=[128, 256, 512]),
                     kernel_size = hp.Choice('conv_5_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_5_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('Dropout_5', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Conv2D(filters     = hp.Choice('conv_6_filters', values=[128, 256, 512]),
                     kernel_size = hp.Choice('conv_6_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_6_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(hp.Choice('Dropout_6', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Conv2D(filters     = hp.Choice('conv_7_filters', values=[128, 256, 512]),
                     kernel_size = hp.Choice('conv_7_kernel',  values=[3, 4, 5]),
                     strides     = (1, 1),
                     padding     = 'same',
                     activation= activations_map[hp.Choice('conv_7_activation', values=['relu','leaky'])]))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(hp.Choice('Dropout_7', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Flatten())
    model.add(Dense(units     = hp.Int('dense_1_units', min_value=32, max_value=1024, step=32),
                    activation= activations_map[hp.Choice('dense_1_activation', values=['relu','leaky'])]))
    model.add(Dropout(hp.Choice('Dropout_11', values=[0.25, 0.30, 0.40, 0.50])))

    model.add(Dense(4,    activation= 'softmax', dtype='float32' ))

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]))

    # Ejecución del modelo con parámetros de compilación
    model.compile( loss      = 'categorical_crossentropy',
                   optimizer = optimizer,
                   metrics   = [ 'accuracy'])

    return model

#-----------------------------------------------------------------------------
# Realizamos la exploración de los mejores parámetros para la red neuronal
# mediante el optimizador Hyperband
#-----------------------------------------------------------------------------
def explore_hypermodel(i_dataset_train, i_dataset_test):

    #Buscador de parámetros HyperbandOptimization
    tuner = kt.Hyperband(
            hypermodel  =create_model,
            objective   ="val_accuracy",
            max_epochs  =10,
            factor      =5,
            seed        =10,
            overwrite   =True,
            directory   ="D:\logs",
            project_name="Category_Turner",
    )

    # Ejecutar la búsqueda de hiperparámetros
    tuner.search(i_dataset_train, epochs=10, validation_data=i_dataset_test)

   # Visualizar summary
    print('1.- Space Summary')
    tuner.search_space_summary()

    print('2.- Results Summary')
    tuner.results_summary()

    # Obtener el mejor modelo encontrado
    best_model = tuner.get_best_models(num_models=1)[0]

    print('3.- Best Model Summary')
    best_model.summary()

    # Grabar los resultados obtenidos
    with open('results_summary_clasificacion_categorica.txt', 'w') as f1:
        with redirect_stdout(f1):
            tuner.results_summary()

    with open('best_model_summary_clasificacion_categorica.txt', 'w') as f2:
        with redirect_stdout(f2):
            best_model.summary()

    #   Devolver la mejor optimización
    return best_model

#-----------------------------------------------------------------------------
# Ejecutamos todo el proceso de entrenamiento.
#-----------------------------------------------------------------------------
def main():

    # Directorios de trabajo
    l_directory_train = 'D:\\Copia 22-09-2025\\TFG\\3.- Modelo clasificacion categorico\\1.- Optimizacion hiperparametros por metodo Hyperband\\1.- Datos optimización categorico\\1.- Entrenamiento'
    l_directory_test  = 'D:\\Copia 22-09-2025\\TFG\\3.- Modelo clasificacion categorico\\1.- Optimizacion hiperparametros por metodo Hyperband\\1.- Datos optimización categorico\\2.- Validacion'

    # Cargar datos.
    (dataset_train, dataset_test) = load_data(l_directory_train, l_directory_test)

    # Obtener el mejor modelo
    explore_hypermodel(dataset_train, dataset_test)

    pass

if __name__ == '__main__':
    main()

