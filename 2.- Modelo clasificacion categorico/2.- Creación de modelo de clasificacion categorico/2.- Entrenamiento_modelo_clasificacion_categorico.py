#-------------------------------------------------------------------------------
# Name:        Modelo clasificación categórica de imágenes sobre tumores cerebrales
# Purpose:     Red neuronal que nos identifica si una tomografía presenta un glioma,
#              un meningioma, un tumor hipofisario o no tiene tumor.
#
# Author:      Raúl Ríos Redondo
#
# Created:     28/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
from keras.models                         import Sequential
from keras.layers                         import Dense,  Dropout, Flatten
from keras.layers                         import Conv2D
from keras.layers                         import MaxPooling2D
from keras.layers                         import BatchNormalization
from ray.train.tensorflow.keras           import ReportCheckpointCallback

import tensorflow  as tf
import numpy       as np
import os

#-----------------------------------------------------------------------------
# Normalización de cada una de las imágenes a entrenar.
#-----------------------------------------------------------------------------
def preprc_func(img):
    img = img.astype(np.float32) / 255.0
    return img

#-----------------------------------------------------------------------------
# Generación de los objetos de entrenamiento y test y retorno de estos.
#-----------------------------------------------------------------------------
def load_data(i_directory_train, i_directory_test):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator( preprocessing_function = preprc_func )

    train_generator = datagen.flow_from_directory(
            i_directory_train,
            target_size =(512, 512),
            batch_size  =32,
            shuffle     =True,
            seed        = 10,
            class_mode  ='categorical')

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
# imágenes tumorales. Retornamos el modelo creado.
#-----------------------------------------------------------------------------
def create_model():

    input_shape = (512,512,3)

    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=input_shape, padding='same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), strides=(1, 1), padding='same', activation= tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (5, 5), strides=(1, 1), padding='same', activation= tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation= tf.keras.layers.LeakyReLU(alpha=0.01) ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.50))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation= tf.keras.layers.LeakyReLU(alpha=0.01) ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.40))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation= 'relu' ))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.30))

    model.add(Flatten())
    model.add(Dense(512, activation= tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(Dropout(0.40))

    model.add(Dense(4,    activation= 'softmax', dtype='float32' ))

    model.summary()

    return model

#-----------------------------------------------------------------------------
# Ejecutamos la compilación y entrenamiento del modelo.
# Parámetros:
#      - i_model: Modelo CNN
#      - i_train_images: Apuntador a los datos de entrenamiento.
#      - i_test_images:  Apuntador a los datos de test.
#-----------------------------------------------------------------------------
def train_model(i_model, i_train_images, i_test_images):

    # Optimizer
    optimizer = tf.keras.optimizers.AdamW(learning_rate = 0.01)

    # Ejecución del modelo con parámetros de compilación
    i_model.compile( loss      = 'categorical_crossentropy',
                     optimizer = optimizer,
                      metrics   = [
                                    # 1. Exactitud estándar
                                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),

                                    # 2. Precisión y Sensibilidad (Recall)
                                    # Nota: En multiclase con one-hot, estas calculan el promedio global
                                    tf.keras.metrics.Precision(name='precision'),
                                    tf.keras.metrics.Recall(name='recall'),

                                    # 3. Área bajo la curva (útil para ver qué tan bien separa las clases)
                                    tf.keras.metrics.AUC(name='auc'),

                                    # 4. Top-K Accuracy (si la clase real está entre las 3 mejores predicciones)
                                    tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),


                        ])

    # Diferentes callbacks
    reduce_lr  = tf.keras.callbacks.LearningRateScheduler(lambda x: 0.003 * 0.9 ** x)

    metrics_list = [
            # Métricas de Entrenamiento
            'loss',
            'accuracy',
            'precision',
            'recall',
            'auc',

            # Métricas de Validación
            'val_loss',
            'val_accuracy',
            'val_precision',
            'val_recall',
            'val_auc'
     ]

    metrics    = ReportCheckpointCallback( metrics=metrics_list)

    # Callback para TensoBoard
    log_dir         = "D:\\Logs"
    early_stop      = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 5 )
    log_tensorboard = tf.keras.callbacks.TensorBoard(log_dir        = log_dir,
                                                     histogram_freq = 0,
                                                     write_graph    = True,
                                                     write_images   = True,
                                                     update_freq    = 'epoch',
                                                     embeddings_freq= 1,
                                                     profile_batch = '1,10')


    batch_size = 32
    steps_per_epoch  = i_train_images.n // batch_size
    validation_steps = i_test_images.n  // batch_size

    # Ejecutamos el entrenamiento de los datos y comparación con los de test.
    history = i_model.fit(i_train_images,
                          steps_per_epoch = steps_per_epoch,
                          epochs = 20,
                          callbacks= [reduce_lr, log_tensorboard, early_stop, metrics],
                          validation_data  = i_test_images,
                          validation_steps = validation_steps)


    # Grabar el modelo entrenado y su histórico.
    i_model.save('modelo_clasificacion_categora.h5')
    np.save('history_clasificacion_categora.npy',history)


#-----------------------------------------------------------------------------
# Ejecutamos todo el proceso de entrenamiento.
#-----------------------------------------------------------------------------
def main():

    # Inicializar variables de ambiente necesarias.
    os.environ['TF_GPU_ALLOCATOR']      = 'cuda_malloc_async'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # Directorios de trabajo
    l_directory_train = 'D:\\Copia 22-09-2025\\TFG\\3.- Modelo clasificacion categorico\\2.- Creación de modelo de clasificacion categorico\\1.- Datos categorico\\1.- Entrenamiento'
    l_directory_test  = 'D:\\Copia 22-09-2025\\TFG\\3.- Modelo clasificacion categorico\\2.- Creación de modelo de clasificacion categorico\\1.- Datos categorico\\2.- Validacion'

    # Cargar datos.
    (dataset_train, dataset_test) = load_data( l_directory_train, l_directory_test )

    # Crear modelo.
    model = create_model()

    # Entrenar modelo.
    train_model(model,dataset_train, dataset_test)

    pass

if __name__ == '__main__':
    main()

