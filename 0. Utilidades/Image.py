#-------------------------------------------------------------------------------
# Name:        Clase Image.
# Purpose:     Se compone de una clase Image con funcionalidades para el
#              tratamiento de las imágenes.
#
# Author:      Raúl Ríos Redondo
#
# Created:     09/03/206
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Importar librerias
import cv2               as cv2
import matplotlib.pyplot as plt
import numpy             as np

#-------------------------------------------------------------------------------
# Clase Image.
#-------------------------------------------------------------------------------
class Image:

#   Constructor
    def __init__(self):
        self.path_fichero      = ''
        self.img               = 0
        self.canal_R           = 0
        self.canal_G           = 0
        self.canal_B           = 0
        self.forma             = 0
        self.tipo              = 0

#   Devolver el canal rojo de la imagen
    def get_R(self):
        return self.canal_R

#   Devolver el canal verde de la imagen
    def get_G(self):
        return self.canal_G

#   Devolver el canal azul de la imagen
    def get_B(self):
        return self.canal_B

#   Modificar el canal rojo de la imagen
    def set_R(self, i_canal_R):
        self.canal_R = i_canal_R
        self.img = cv2.merge((self.canal_R,self.canal_G,self.canal_B))

#   Modificar el canal verde de la imagen
    def set_G(self, i_canal_G):
        self.canal_G = i_canal_G
        self.img = cv2.merge((self.canal_R,self.canal_G,self.canal_B))

#   Modificar el canal azul de la imagen
    def set_B(self, i_canal_B):
        self.canal_B = i_canal_B
        self.img = cv2.merge((self.canal_R,self.canal_G,self.canal_B))

#   Devolver las dimensiones de la imagen
    def get_shape(self):
        return self.forma

#   Devolver el tipo de dato utilizado en los valores de la imagen
    def get_type(self):
        return self.tipo

#   Devolver la matriz de la imagen
    def get_image(self):
        return self.img

#   Modificar la matriz de la imagen
    def set_image(self, i_img):
        self.img = i_img
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # Correccion COLOR
        self.reset_values()

#   Leer una imagen de un fichero
    def read_file(self, i_path_fichero):
        self.path_fichero      = i_path_fichero
        self.img = cv2.imread(i_path_fichero)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # Correccion COLOR
        self.reset_values()

#   Visualizar la imagen
    def display_image(self):
        plt.imshow(self.img)
        plt.show()
        cv2.waitKey(0)

#   Visualizar la imagen sin pausa.
    def display_image_no_wait(self):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Frame", img)

#   Visualizar los canales de la imagen
    def display_canals(self):

        # Mostramos nuestros canales
        fig = plt.figure()

        # CANAL ROJO
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(self.canal_R)
        ax1.set_title("CANAL ROJO")

        # CANAL VERDE
        ax2 = fig.add_subplot(2,2,2)
        ax2.imshow(self.canal_G)
        ax2.set_title("CANAL VERDE")

        # CANAL AZUL
        ax3 = fig.add_subplot(2,2,3)
        ax3.imshow(self.canal_B)
        ax3.set_title("CANAL AZUL")

        plt.show()
        cv2.waitKey(0)

#   Guardar la imagen en un fichero.
    def save_file(i_path_fichero):

        if i_path_fichero == "":
            cv2.imwrite(self.path_fichero, self.img)
        else:
            cv2.imwrite(i_path_fichero, self.img)

#   Devolvemos un trozo de la imagen.
    def cut(self, i_fila1, i_col1, i_fila2, i_col2):
        img    = self.img[i_fila1:i_fila2, i_col1:i_col2]
        imagen = Image()
        imagen.set_image(img)
        return imagen

#   Redimensionamos la imagen
    def resize(self, i_x, i_y):
        self.img = cv2.resize(self.img, None, fx = i_x, fy = i_y)
        self.reset_values()

#   Rotar la imagen. Los valores posibles son 0, 1, -1.
    def rotate(self, i_value_rotate):
        self.img = cv2.flip(self.img, i_value_rotate)
        self.reset_values()

#   Aumentar o disminuir el brillo de una imagen.
#   Valores "+" aumenta y "-" disminuye el brillo.
    def set_brightness(self, i_operacion, i_value):
        matriz_brillo = np.ones(self.img.shape, dtype='uint8') * i_value

        if i_operacion == '+':
            self.img      = cv2.add(self.img, matriz_brillo)

        if i_operacion == '-':
            self.img      = cv2.subtract(self.img, matriz_brillo)

        self.reset_values()

#   Indicamos el umbral de la imagne en binario.
#   Umbral: i_debajo = 0, i_encima = 255,
#           i_operacion = (N/I)(Normal o Invertido)
    def set_threshold(self, i_debajo, i_encima, i_operacion):

        if i_operacion == 'N':
            l_oper = cv2.THRESH_BINARY

        if i_operacion == 'I':
            l_oper =  cv2.THRESH_BINARY_INV

        _, self.img = cv2.threshold(self.img, i_debajo, i_encima, l_oper)

        self.reset_values()

#   Realizamos operaciones de bit a la imagen
#   La imagen de entrada se une a la propia.
#   Valores de i_operacion (AND/OR/XOR)
    def set_bitwise(self, i_operacion, i_img):

        if i_operacion == 'AND':
            self.img = cv2.bitwise_and(self.img, i_img, mask = None)

        if i_operacion == 'OR':
            self.img = cv2.bitwise_or(self.img, i_img, mask = None)

        if i_operacion == 'XOR':
            self.img = cv2.bitwise_or(self.img, i_img, mask = None)

        self.reset_values()

#   Inicializa los parámetros de la imagen en funciones de uso.
    def reset_values(self):
        self.forma = self.img.shape
        self.tipo  = self.img.dtype

        if len(self.forma) == 3:
            self.canal_R, self.canal_G, self.canal_B = cv2.split(self.img)

#------------------------------------------------------------------------
#   Operaciones de procesamiento de imagen
#------------------------------------------------------------------------

#   Devolvemos la imagen desenfocada.
#   Valores i_x, i_y nos indica e tipo de desenfoque en la imagen

    def blur_image(self, i_x,i_y):
        self.img = cv2.blur(self.img, (i_x, i_y))
        self.reset_values()

#   Detectamos los bordes de la imagen
#   Valores i_x, i_y nos indica el umbral superior e imferior
#   Indica a partir del valor de pixel que pone umbral de negro y blanco.

    def detect_border_image(self, i_sup,i_inf):
        self.img = cv2.Canny(self.img, i_sup, i_inf)
        self.reset_values()

#   Detectamos las esquinas de la imagen, en función de unos parámetros de
#   entrada.
#   Parametros para detector de esquinas
#   esquinas_param = dict(maxCorners  =  Maximo numero de esquinas a detectar
#                         qualityLevel =  Umbral minimo para la deteccion de esquinas
#                         minDistance  =  Distacia entre pixeles
#                         blockSize    =  Area de pixeles

    def detect_corners_image(self, i_esquinas_param):

        # Conversion a escala de grises
        imagen_gris = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Calculamos las caracteristicas de las esquinas
        vector_esquinas = cv2.goodFeaturesToTrack(imagen_gris, **i_esquinas_param)

        # Preguntamos si detectamos esquinas con esas caracteristicas
        if vector_esquinas is not None:
            # Iteramos
            for x, y in np.float32(vector_esquinas).reshape(-1,2):

                # Convertimos en enteros
                x,y = int(x), int(y)

                # Dibujamos la ubicacion de las esquinas
                cv2.circle(self.img, (x,y), 10, (255,0,0), 1)

#   Devolvemos las siguientes operaciones de coincidencias.
#   Devuelve las dos imagenes, la primera es la nuestra
#   y la segunda con los puntos coincidentes.
#
#   Param:
#       i_img_comparar: Es la imagen a comparar
#       i_puntos: Número de puntos máximo de coincidencias.

    def image_coincidences(self,i_img_comparar, i_puntos):

#       Pasar imagenes a gris.
        img_patron    = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imp_comparada = cv2.cvtColor(i_img_comparar, cv2.COLOR_BGR2GRAY)

        # Declaramos el objeto
        orb = cv2.ORB_create(i_puntos)

        # Extraemos la info de la img patron
        keypoint1, descriptor1 = orb.detectAndCompute(img_patron, None)

        # Extraemos la info de la imagen a comparar.
        keypoint2, descriptor2 = orb.detectAndCompute(imp_comparada, None)

        # Dibujamos puntos
        img_display   = cv2.drawKeypoints(self.img,
                                          keypoint1,
                                          outImage = np.array([]),
                                          color =(255,0,0),
                                          flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        img_comp_display = cv2.drawKeypoints(i_img_comparar,
                                             keypoint1,
                                             outImage = np.array([]),
                                             color =(255,0,0),
                                             flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        return img_display, img_comp_display

#   Devolvemos la imagen con las coincidencias.
#
#   Param:
#       i_img_comparar: Es la imagen a comparar
#       i_puntos: Número de puntos máximo de coincidencias.
#
    def image_matches(self,i_img_comparar, i_puntos):

#       Pasar imagenes a gris.
        img_patron    = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imp_comparada = cv2.cvtColor(i_img_comparar, cv2.COLOR_BGR2GRAY)

        # Declaramos el objeto
        orb = cv2.ORB_create(i_puntos)

        # Extraemos la info de la img patron
        keypoint1, descriptor1 = orb.detectAndCompute(img_patron, None)

        # Extraemos la info de la imagen a comparar.
        keypoint2, descriptor2 = orb.detectAndCompute(imp_comparada, None)

        # Dibujamos puntos
        img_display   = cv2.drawKeypoints(self.img,
                                          keypoint1,
                                          outImage = np.array([]),
                                          color =(255,0,0),
                                          flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        img_comp_display = cv2.drawKeypoints(i_img_comparar,
                                             keypoint1,
                                             outImage = np.array([]),
                                             color =(255,0,0),
                                             flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # 1. Creamos un objeto comparador de descriptores
        obj_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        obj_matches = obj_matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        obj_matches = sorted(obj_matches,
                             key = lambda x: x.distance,
                             reverse = False)

        # 3. Filtramos los resultados
        puntos_buenos = int(len(obj_matches) * 0.1)
        obj_matches = obj_matches[:puntos_buenos]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(self.img,
                                      keypoint1,
                                      i_img_comparar,
                                      keypoint2,
                                      obj_matches,
                                      None)


        return img_matches


#   Devolvemos la imagen alineada a la original, según las coincidencias.
#
#   Param:
#       i_img_comparar: Es la imagen a comparar
#       i_puntos: Número de puntos máximo de coincidencias.

    def image_align(self,i_img_comparar, i_puntos):

#       Pasar imagenes a gris.
        img_patron    = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imp_comparada = cv2.cvtColor(i_img_comparar, cv2.COLOR_BGR2GRAY)

        # Declaramos el objeto
        orb = cv2.ORB_create(i_puntos)

        # Extraemos la info de la img patron
        keypoint1, descriptor1 = orb.detectAndCompute(img_patron, None)

        # Extraemos la info de la imagen a comparar.
        keypoint2, descriptor2 = orb.detectAndCompute(imp_comparada, None)

        # Dibujamos puntos
        img_display   = cv2.drawKeypoints(self.img,
                                          keypoint1,
                                          outImage = np.array([]),
                                          color =(255,0,0),
                                          flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        img_comp_display = cv2.drawKeypoints(i_img_comparar,
                                             keypoint1,
                                             outImage = np.array([]),
                                             color =(255,0,0),
                                             flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # 1. Creamos un objeto comparador de descriptores
        obj_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        obj_matches = obj_matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        obj_matches = sorted(obj_matches,
                             key = lambda x: x.distance,
                             reverse = False)

        # 3. Filtramos los resultados
        puntos_buenos = int(len(obj_matches) * 0.1)
        obj_matches = obj_matches[:puntos_buenos]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(self.img,
                                      keypoint1,
                                      i_img_comparar,
                                      keypoint2,
                                      obj_matches,
                                      None)

        # ¿Como calculamos la homografia de la imagen?
        # 1. Creamos listas con el tamaño del total de keypoints
        puntos1 = np.zeros((len(obj_matches), 2), dtype = np.float32)
        puntos2 = np.zeros((len(obj_matches), 2), dtype = np.float32)

        # 2. Extraemos los puntos
        for i, match in enumerate(obj_matches):

            # Puntos de imagen
            puntos1[i, :] = keypoint1[match.queryIdx].pt

            # Puntos de frames
            puntos2[i, :] = keypoint2[match.trainIdx].pt

        # 3. Extraemos la homografia
        h, mask = cv2.findHomography(puntos2, puntos1, cv2.RANSAC)

        # 4. Dibujamos
        alto, ancho, canales = self.img.shape
        img_coincidencias = cv2.warpPerspective(i_img_comparar, h, (ancho, alto))

        return img_coincidencias

#------------------------------------------------------------------------
#  Nos permite detectar  un determinado objeto en una imagen
#  Para ello pasamos los siguientes parámetros:
#  - i_net_model:  Pasamos el modelo de la red neuronal
#  - i_net_config: Pasamos la red entrenada.
#  - i_net_clases: Pasamos el fichero con nombres de las clases o categorias a detectar
#  - i_width_net:  Indicames el ancho en píxeles de la imagen mínimia a buscar.
#  - i_high_net:   Indicames el alto  en píxeles de la imagen mínimia a buscar.
#  - i_media:      Indicamos los valores médios de los canales de color [x,y,z]
#  - i_threshold:  Indicamos la valor mínimo en % de detección de un objeto.
#------------------------------------------------------------------------
    def detect_object_tensorflow( self,
                                  i_net_model,
                                  i_net_config,
                                  i_net_clases,
                                  i_width_net,
                                  i_high_net,
                                  i_media,
                                  i_threshold):

        # Leemos el modelo
        red_neuronal = cv2.dnn.readNetFromTensorflow(i_net_model, i_net_config)
        self.rotate(1)

        # Extraemos info de los frames
        altoframe  = self.img.shape[0]
        anchoframe = self.img.shape[1]

        # Preprocesamos la imagen
        # Images - Factor de escala - tamaño - media de color - Formato de color(BGR-RGB) - Recorte
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (i_width_net, i_high_net), i_media, swapRB = True, crop = False)

        # Ejecutamos el modelo
        red_neuronal.setInput(blob)
        detecciones = red_neuronal.forward()

        # Extraemos las etiquetas del archivo
        with open(i_net_clases) as cl:
            labels = cl.read().split("\n")

        # Iteramos
        for i in range(detecciones.shape[2]):

            # Extraemos la confianza de esa deteccion
            conf_detect = detecciones[0,0,i,2]
            clase       = int(detecciones[0, 0, i, 1])

            # Si superamos el umbral (70% de probabilidad de que sea un objeto)
            if conf_detect > i_threshold:

                # Extraemos las coordenadas
                xmin = int(detecciones[0, 0, i, 3] * anchoframe)
                ymin = int(detecciones[0, 0, i, 4] * altoframe)
                xmax = int(detecciones[0, 0, i, 5] * anchoframe)
                ymax = int(detecciones[0, 0, i, 6] * altoframe)

                # Dibujamos el rectangulo
                cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)

                # Texto que vamos a mostrar
                name_object = str(labels[clase])
                label = name_object + " Conf: %.4f" % conf_detect

                # Tamaño del fondo del label
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Colocamos fondo al texto
                cv2.rectangle(self.img, (xmin, ymin - label_size[1]),
                                     (xmin + label_size[0], ymin + base_line),
                                     (0,0,0), cv2.FILLED)

                # Colocamos el texto
                cv2.putText(self.img, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
