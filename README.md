# Detección de tumores cerebrales mediante análisis de resonancias magnéticas (RMI) por redes neuronales convolucionales.

**Trabajo de Fin de Grado**  
**Grado en Ingeniería Informática**  
**Universitat Oberta de Catalunya (UOC)**

## Descripción del Proyecto

<div align="justify">
Actualmente, el cáncer cerebral representa un 2% del total de los cánceres diagnosticados, lo que ha provocado que no reciba una especial atención en los tratamientos oncológicos, frente a tumores más frecuentes.  A pesar de ello, en las últimas décadas y debido al envejecimiento progresivo de la población, la incidencia de esta enfermedad ha aumentado, lo que hace necesario desarrollar estrategias que permitan prevenir su crecimiento y evitar el aumento de casos.
  
En los últimos años, el “Aprendizaje profundo”  ha proporcionado herramientas que facilitan la detección de patrones en las imágenes médicas, como la localización de tumores, fracturas o el diagnóstico de diversas enfermedades. Entre las principales técnicas empleadas se encuentran las redes neuronales convolucionales (CNN), que permiten identificar características y patrones específicos en las imágenes médicas. Su aplicación en el campo de la sanidad puede contribuir a reducir los tiempos de diagnóstico y, en el caso concreto de los tumores cerebrales, a favorecer una intervención más temprana y eficaz.

El objetivo de este trabajo es desarrollar modelos de clasificación y localización, integrando diversos conjuntos de datos públicos de resonancias magnéticas  cerebrales. Como parte del desarrollo, se diseñarán diversas arquitecturas de redes neuronales convolucionales que permitan detectar el cáncer cerebral. Con este fin, se desarrollará un  modelo de clasificación binaria (tumor / no tumor), un modelo de clasificación categórica (glioma, meningiomas, pituitario) y un modelo de localización basado en la arquitectura YOLO para la detección de objetos en tiempo real. 
</div>

## Tecnologías

* **Python 3.12.3**
* **TensorFlow 2.19.0**
* **Keras 3.10.0**
* **YOLO 11s**
* **TensorBoard 2.19.0**
* **OpenCV 4.12.0**
* **LIME y SHAP(XAI)**
