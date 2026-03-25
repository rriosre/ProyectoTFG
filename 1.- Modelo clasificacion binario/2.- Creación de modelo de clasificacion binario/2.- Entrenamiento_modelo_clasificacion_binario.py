#-------------------------------------------------------------------------------
# Name:        Modelo clasificación binaria de imágenes sobre tumores cerebrales
# Purpose:     Red neuronal que nos identifica si una tomografía presenta un
#              tumor cerebral o no.
#
# Author:      Raúl Ríos Redondo
#
# Created:     22/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------
# Carga de librerías y apis.
from keras.models                import Sequential
from keras.layers                import Dense, Dropout, Flatten
from keras.layers                import Conv2D
from keras.layers                import MaxPooling2D
from keras.layers                import BatchNormalization
from ray.train.tensorflow.keras  import ReportCheckpointCallback

import tensorflow as tf
import numpy      as np

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
            target_size=(512, 512),
            batch_size=32,
            class_mode='binary')

    test_generator = datagen.flow_from_directory(
            i_directory_test,
            target_size=(512, 512),
            batch_size=32,
            class_mode='binary')

    return train_generator, test_generator

#-----------------------------------------------------------------------------
# Generación de la arquitectura del modelo CNN para el tratamiento de las
# imágenes tumorales. Retornamos el modelo creado.
#-----------------------------------------------------------------------------
def create_model():

    input_shape = (512,512, 3)

    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape = input_shape, padding='same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (5, 5), padding='same', activation= tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation= tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (5, 5), padding='same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (5, 5), padding='same', activation= 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(224,  activation= tf.keras.layers.LeakyReLU(alpha=0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256,  activation= 'relu'))
    model.add(Dense(1,    activation= 'sigmoid'))

    return model

#-----------------------------------------------------------------------------
# Ejecutamos la compilación y entrenamiento del modelo.
# Parámetros:
#      - i_model: Modelo CNN
#      - i_train_images: Apuntador a los datos de entrenamiento.
#      - i_test_images:  Apuntador a los datos de test.
#-----------------------------------------------------------------------------
def train_model(i_model, i_train_images, i_test_images, i_img_tumor, i_img_no_tumor):

    i_model.summary()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)


    # Ejecución del modelo con parámetros de compilación
    i_model.compile( loss      ='binary_crossentropy',
                     optimizer = optimizer,
                     metrics   = [
                            # Accuracy
                            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                            tf.keras.metrics.Accuracy(name="accuracy"),

                            # Precision / Recall
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.PrecisionAtRecall(0.8, name="precision_at_recall"),
                            tf.keras.metrics.RecallAtPrecision(0.8, name="recall_at_precision"),
                            tf.keras.metrics.SensitivityAtSpecificity(0.8, name="sensitivity_at_specificity"),
                            tf.keras.metrics.SpecificityAtSensitivity(0.8, name="specificity_at_sensitivity"),

                             # AUC
                            tf.keras.metrics.AUC(name="auc_roc"),
                            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),

                            # Conteos
                            tf.keras.metrics.TruePositives(name="tp"),
                            tf.keras.metrics.TrueNegatives(name="tn"),
                            tf.keras.metrics.FalsePositives(name="fp"),
                            tf.keras.metrics.FalseNegatives(name="fn"),

                            ])

    # Diferentes callbacks
    reduce_lr       = tf.keras.callbacks.LearningRateScheduler(lambda x: 0.003 * 0.9 ** x)

    metrics_list = [
    # Entrenamiento
    "loss","accuracy","precision","recall","precision_at_recall",
    "recall_at_precision","sensitivity_at_specificity","specificity_at_sensitivity",
    "auc_roc","auc_pr","tp","tn","fp","fn",

    # Validación
    "val_loss","val_accuracy","val_precision","val_recall","val_precision_at_recall",
    "val_recall_at_precision","val_sensitivity_at_specificity","val_specificity_at_sensitivity",
    "val_auc_roc","val_auc_pr","val_tp","val_tn","val_fp","val_fn"
    ]

    metrics    = ReportCheckpointCallback( metrics=metrics_list )

    # Callback para TensoBoard
    log_dir         = "D:\\Logs"
    early_stop      = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss', patience = 5 )
    log_tensorboard = tf.keras.callbacks.TensorBoard(log_dir        = log_dir ,
                                                     histogram_freq = 1,
                                                     write_graph    = True,
                                                     write_images   = True,
                                                     update_freq    = 'epoch',
                                                     embeddings_freq= 1,
                                                     profile_batch = '1,10')
    # Cálculo de batch_size
    batch_size = 64
    steps_per_epoch  = i_train_images.n // batch_size
    validation_steps = i_test_images.n  // batch_size

    # Aplicar balanceo
    l_img_tumor     = i_img_tumor    # Imagenes de entrenamiento actuales
    l_img_no_tumor  = i_img_no_tumor # Imagenes de entrenamiento actuales
    l_total         = l_img_tumor + l_img_no_tumor

    l_peso_tumor       = l_total / ( 2 * l_img_tumor )
    l_peso_no_tumor    = l_total / ( 2 * l_img_no_tumor )
    class_weights_dict = {0:l_peso_no_tumor,1:l_peso_tumor }

    # Ejecutamos el entrenamiento de los datos y comparación con los de test.
    history = i_model.fit(i_train_images,
                          steps_per_epoch  = steps_per_epoch,
                          epochs           = 20,
                          callbacks        =[reduce_lr, log_tensorboard, early_stop, metrics],
                          class_weight     = class_weights_dict, # Utilizar balanceo
                          validation_data  = i_test_images,
                          validation_steps = validation_steps)

    # Grabar el modelo entrenado y su histórico.
    i_model.save('modelo_clasificacion_binaria.h5')
    np.save('history_clasificacion_binaria.npy',history)

#-----------------------------------------------------------------------------
# Ejecutamos todo el proceso de entrenamiento.
#-----------------------------------------------------------------------------
def main():

    # Directorios de trabajo
    l_directory_train = 'D:\\Copia 22-09-2025\\TFG\\2.- Modelo clasificacion binario\\2.- Creación de modelo de clasificacion binario\\1.- Datos tumor y no tumor\\1.- Entrenamiento'
    l_directory_test  = 'D:\\Copia 22-09-2025\\TFG\\2.- Modelo clasificacion binario\\2.- Creación de modelo de clasificacion binario\\1.- Datos tumor y no tumor\\2.- Validacion'
    l_img_tumor       =  44507 # Num imagenes con tumor.
    l_img_no_tumor    =  13005 # Num imagenes con tumor.

    # Cargar datos.
    (dataset_train, dataset_test) = load_data( l_directory_train, l_directory_test )

    # Crear modelo.
    model = create_model()

    # Entrenar modelo.
    train_model(model,dataset_train, dataset_test, l_img_tumor, l_img_no_tumor)

    pass

if __name__ == '__main__':
    main()

