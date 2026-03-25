#-------------------------------------------------------------------------------
# Name:        Arquitectura del modelo de clasificación categórico
# Purpose:     Dibujar la arquitectura utilizada para la clasificación categórica.
#
# Author:      Raúl Ríos Redondo
#
# Created:     28/03/2026
# Copyright:   (c) Raúl Ríos - UOC
# Licence:     3.0 España de Creative Commons
#-------------------------------------------------------------------------------

# Carga de librerías y apis.
import tensorflow          as tf
import visualkeras         as vk

#-----------------------------------------------------------------------------
# Ejecutamos todo el proceso para presentar la arquitectura en un fichero.
#-----------------------------------------------------------------------------
def main():

    # Variables globales
    l_modelo_clasificacion  = 'modelo_clasificacion_categorica.h5'

    # Recuperar el modelo entrenado.
    model = tf.keras.models.load_model(l_modelo_clasificacion)

    # Solución manual: inyectar el atributo output_shape en cada capa para poder visualizar con visualkeras
    for layer in model.layers:
        layer.output_shape = layer.output.shape

    # Crear imagen de la arquitectura del modelo.
    vk.layered_view(    model,                                                      # Modelo
                        legend      = True,                                         # Muestra la leyenda de colores
                        draw_volume = True,                                         # Activa la vista 3D (volumen)
                        spacing     = 30,                                           # Espacio entre bloques para que no se amontonen
                        scale_xy    = 1.5,                                          # Escala el ancho/alto de los bloques
                        scale_z     = 0.1,                                          # Escala la profundidad (útil si tienes muchos filtros)
                        max_xy      = 400,                                          # Limita el tamaño máximo visual para que no sea gigante
                        max_z       = 100,                                          # Limita el grosor máximo visual
                        to_file     = 'arquitectura_clasificacion_categorica.png' ) # Guarda el resultado en fichero).

    pass

if __name__ == '__main__':
    main()