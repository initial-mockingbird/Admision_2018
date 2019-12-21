# Triangulacion de Imagenes

Como proyecto de admision del periodo Enero-Marzo 2018 del GIA, se insto a diseñar un algoritmo genetico el cual
pudiera aproximar mediante triangulos cualquier imagen dada. El proyecto presentado plantea una solucion sencilla
la cual converge rapidamente. Sin embargo, **no** debe tomarse como una version definitiva, pues aun tiene 
una gran refactorizacion pendiente.

## Instalacion

Basta con Clonar el repositorio e instalar las dependencias señaladas en el archivo: requirements.txt mediante:

```
$ pip3 install -r requirements.txt 
```

Y listo para correr!

## Ejecutando el script

Si es primera vez que se ejecuta el script, o el archivo de configuracion no existe (ya sea porque se elimino o se
renombro), entonces basta con llamar al script mediante la consola con python:

```
python3 SERIO.py
```

Y el creara el archivo de configuracion y aplicara la triangulacion sobre una de las imagenes de muestra.

Si se quiere hacer cambios sobre el archivo de configuracion, tales como cambiar la imagen a triangular, 
la ruta de guardado, u cualquier otro dato, basta con modificar los campos correspondiente en el propio archivo.

## Resultados

![best_girl](./BESTGIRL.gif.mp4)

![monte](./monte.gif.mp4)

## Authors

* **Daniel Pinto** [github](https://github.com/PurpleBooth)

