#-------------------------------------------------------------------------------
# Name:        Utilidades para trabajar con redes neuronales.
# Purpose:     Se compone de las siguientes funciones.
#                  - Funciones para análisis exploratorio de datos.
#                  - Funciones para mostrar gráficos estadísticos.
#                  - Funciones para el tramiento con las redes neuronales.
# Author:      Raúl Ríos Redondo
#
# Created:     09/03/206
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Importar librerias
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
import numpy             as np
import cv2               as cv2
import Image             as Img
import pandas            as pd
import shutil            as sh
import os
import array

#####################################################################################
#####################################################################################
#   Funciones Análisis exploratorio de datos.
#####################################################################################
#####################################################################################

#-------------------------------------------------------------------------------
#   Modificar el valor de la categoría del tumor en el directorio de las
#   etiquetas de las imágenes.
#            - i_directory: Directorio de las etiquetas de las imágenes.
#            - i_value:     Código tumor a modificar.
#-------------------------------------------------------------------------------
def update_category_brain_tumor(i_directory, i_value):

    list_files = os.listdir(i_directory)

    for file in list_files:

        path_completo = os.path.join(i_directory,file)
        if os.path.isfile(path_completo):
            nombre, extension = os.path.splitext(file) #
            if extension == '.txt':
                f = open(path_completo, 'r+t') # leemos el fichero de etiquetas
                lines = f.readlines()
                f.seek(0)

                for registro in lines:
                    vector = array.array('u',registro )
                    vector[0] = i_value
                    registro = vector.tounicode()
                    f.write(registro) #Escribir label en el fichero

                f.close()

#-------------------------------------------------------------------------------
#   Crea un fichero txt con las imágenes que pertenezcan a una categoría
#   específica.
#   Parámetros:
#       - i_directory: Directorio con las imágenes a tratar.
#       - i_categoria: Indicar el nombre de la categoria de las imágenes tratadas.
#-------------------------------------------------------------------------------
def create_labels_txt_dir(i_directory, i_categoria='Clase1'):

    list_files = os.listdir(i_directory)
    fichero_labels = i_directory + '\\' + 'labels.txt'
    f = open(fichero_labels, 'x') # Creamos el fichero de etiquetas

    for file in list_files:

        path_completo = os.path.join(i_directory,file)
        if os.path.isfile(path_completo):
            registro = file + ';' + file + ';' + i_categoria + "\n"
            f.write(registro) #Escribir label en el fichero

    f.close()

#-------------------------------------------------------------------------------
#   Nospermite transformar un directorio coon sus imagenes a una determinadoa
#   resolución y nombre. Crea también un fichero con los labels de validación.
#   Parámetros:
#       - i_directory: Directorio con las imágenes a tratar.
#       - i_num_image: Contador del nombre de las imágenes.
#       - i_size: Indicar el tamaño de la imagen. El mismo de altura y ancho.
#       - i_categoria: Indicar el nombre de la categoria de las imágenes tratadas.
#-------------------------------------------------------------------------------
def tranform_dir(i_directory, i_num_image=1 , i_size=512, i_categoria='Clase1'):

    list_files = os.listdir(i_directory)
    fichero_labels = i_directory + '\\' + 'labels.txt'
    f = open(fichero_labels, 'x') # Creamos el fichero de etiquetas

    for file in list_files:

        path_completo = os.path.join(i_directory,file)
        if os.path.isfile(path_completo):
            imagen  = Img.Image()
            imagen.read_file(path_completo)
            imagen.resize(i_size,i_size)
            imagen.save_file(path_completo)

            nombre, extension = os.path.splitext(file) #
            file_dest = 'IMA' + str(i_num_image).zfill(5) + extension
            path_completo_dest = os.path.join(i_directory,file_dest)
            os.renames(path_completo, path_completo_dest)

            registro = file + ';' + file_dest + ';' + i_categoria + "\n"
            f.write(registro) #Escribir label en el fichero

            i_num_image = i_num_image + 1

    f.close()

#-------------------------------------------------------------------------------
#   Visualizar información de un directorio de imagenes. Nos dice el total de
#   imágenes
#       - i_directory: Directorio con las imágenes a tratar.
#-------------------------------------------------------------------------------
def information_dir(i_directory):
    list_files = os.listdir(i_directory)
    print(list_files[:5])
    print('Total images:', len(list_files))


#-------------------------------------------------------------------------------
#   Visualizar información de un directorio de etiquestas, contando los que hay
#   de una determinada clase.
#   - i_directory_labels: Directorio con las eqiquetas de las imágenes
#   - i_clase:            La categoría a contar del directorio de etiquetas.
#-------------------------------------------------------------------------------
def information_category_dir(i_directory_labels, i_clase):
    list_files = os.listdir(i_directory_labels) # Leer directorio de labels

    l_count_class = 0
    for file in list_files: # Mirar cada fichero

        path_completo_label = os.path.join(i_directory_labels,file)

        if os.path.isfile(path_completo_label):
            nombre, extension = os.path.splitext(file) #
            if extension == '.txt':
                f = open(path_completo_label, 'r') # Leemos el fichero de etiquetas
                lines = f.readlines()
                f.seek(0)
                for registro in lines:
                    vector = array.array('u',registro )
                    if i_clase == vector[0]:
                       l_count_class = l_count_class + 1
                f.close()

    return l_count_class

#-------------------------------------------------------------------------------
#   Visualizar imágenes de un directorio, en formato 5x5.
#   parametros:
#       i_directory : Directorio donde están las imágenes
#       i_num_images : Número de imagenes a visualizar (máximo 25).
#       i_size: Si queremos redimensionar las imagenes a visualizar.
#       i_grey: Si queremos ver en tonos grises las imagenes a visualizar.
#-------------------------------------------------------------------------------
def display_images(i_directory, i_num_images, i_size, i_gray):
    nrows = 5
    ncols = 5

    pic_index = 0 # Indice para iterar sobre las imagenes

    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)

    pic_index+=8

    if i_num_images > 25: #No mayor de 25 imágenes
        i_num_images = 25

    lt_fnames = os.listdir(i_directory)
    next_pix = [os.path.join(i_directory, fname)
                    for fname in lt_fnames[pic_index-8:i_num_images]]

    print(next_pix)
    for i, img_path in enumerate (next_pix) :
        img = mpimg.imread(img_path)

        if i_size != '':
            img = cv2.resize(img,(i_size,i_size))

        if i_gray != '':
            plt. imshow (img, cmap='gray')
        else:
            plt. imshow (img)

    plt.show()

#-------------------------------------------------------------------------------
#   Extraer en directorio de clasificación por localización, los ficheros
#   de una determinada categoría.
#        - i_directory_labels: Directorio con las eqiquetas de las imágenes
#        - i_directory_images: Directorio con las imágenes
#        - i_directory_tmp:    Directorio con las imágenes y etiquetas copiadas
#        - i_clase:            La categoría a extraer de los directorios de
#                              etiquetas e imágenes.
#
#-------------------------------------------------------------------------------
def extract_images_category(i_directory_labels, i_directory_images, i_directory_tmp, i_clase):

    list_files = os.listdir(i_directory_labels) # Leer directorio de labels

    for file in list_files:

        path_completo_label = os.path.join(i_directory_labels,file)

        if os.path.isfile(path_completo_label):
            nombre, extension = os.path.splitext(file) #
            if extension == '.txt':
                f = open(path_completo_label, 'r') # Leemos el fichero de etiquetas
                lines = f.readlines()
                f.seek(0)

                for registro in lines:
                    vector = array.array('u',registro )
                    if i_clase == vector[0]:
                        file_image = nombre + '.png'
                        if file_image in os.listdir(i_directory_images):
                            path_completo_image = os.path.join(i_directory_images,file_image)
                            sh.copy(path_completo_image, i_directory_tmp)
                            sh.copy(path_completo_label, i_directory_tmp)
                        else:
                            file_image = nombre + '.jpg'
                            if file_image in os.listdir(i_directory_images):
                                path_completo_image = os.path.join(i_directory_images,file_image)
                                sh.copy(path_completo_image, i_directory_tmp)
                                sh.copy(path_completo_label, i_directory_tmp)
                f.close()


#####################################################################################
#####################################################################################
#   Funciones de tratamiento con redes neuronales.
#####################################################################################
#####################################################################################

#-------------------------------------------------------------------------------
#   Visualizar gráfico de estructura de capas del modelo.
#        - i_model: Modelo creado a visulizar.
#        - i_file:  Fichero donde guardar la estructura del modelo visualizado.
#-------------------------------------------------------------------------------
def plot_model(i_model, i_file):
    keras.utils.plot_model (i_model, i_file, show_shapes=True)

#-------------------------------------------------------------------------------
#   Mostrar gráfico para visualizar la comparativa del entrenamiento y la
#   validación.
#        - i_history : Pasamos fichero del histórico del entrenamiento de la red.
#-------------------------------------------------------------------------------
def plot_history_mse(i_history):

    hist          = pd.DataFrame (i_history.history)
    hist['epoch'] = i_history.epoch

    plt.figure()
    plt.xlabel ('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],

    label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show ()

#-------------------------------------------------------------------------------
#   Mostrar gráfico para visualizar la comparativa del entrenamiento y la
#   validación de la precisión y la pérdida
#        - i_history : Pasamos fichero del histórico del entrenamiento de la red.
#-------------------------------------------------------------------------------
def plot_history_accuracy_lost(i_history):
    history_dict= i_history.history
    print(history_dict.keys())

    acc      = i_history.history['acc']
    val_acc  = i_history.history['val_acc']
    loss     = i_history.history['loss']
    val_loss = i_history.history['val_loss']

    epochs  = range(1,len(acc)+1,1)

#   Mostrar precisión.
    plt.plot (epochs, acc,     'r--', label='Training acc')
    plt.plot (epochs, val_acc, 'b',   label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.figure()

#   Mostrar pérdida.
    plt.plot(epochs, loss,     'r--')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend()
    plt.figure()
    plt.show()

#####################################################################################
#####################################################################################
#   Funciones de gráficos estadísticos.
#####################################################################################
#####################################################################################

#-------------------------------------------------------------------------------
#   Visualizar gráficos de barras vertical
#        - i_colums_labels: Nombre de las columnas.
#        - i_colums_values: Valores de las columnas.
#        - i_title:         Título del gráfico.
#        - i_ylabel:        Título del eje y.
#        - i_xlabel:        Título del eje x.
#-------------------------------------------------------------------------------
def vertical_bar_chart(i_colums_labels, i_colums_values, i_title, i_ylabel, i_xlabel):
    plt.bar(
        i_colums_labels,
        i_colums_values,
        label = "Datos",
        color = "blue",
        align = "center"
        )

    plt.title(i_title)
    plt.xlabel(i_xlabel)
    plt.ylabel(i_ylabel)
    plt.legend()
    plt.grid(False)
    plt.show()

#-------------------------------------------------------------------------------
#   Visualizar gráficos de particiones / queso.
#        - i_values: Valores de las particiones.
#        - i_labels: Nombre de las particiones.
#        - i_colors: Colores de cada partición.
#        - i_title:  título del gráfico.
#-------------------------------------------------------------------------------
def pie_chart(i_values, i_labels, i_colors, i_title):

    # Crear gráfico
    plt.pie(i_values, labels=i_labels, autopct=lambda p: f'{p*sum(i_values)/100 :.0f}', colors=i_colors, startangle=90)
    plt.axis("equal") # Asegura que sea un círculo
    plt.title(i_title)
    plt.show()