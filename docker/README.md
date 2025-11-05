# Docker

Este directorio contiene todo lo necesario para realizar la inferencia del modelo utilizando Docker.

## 1. Ubicación del directorio

Desde la raíz del repositorio, ingresar a la carpeta `docker`:

D:/Downloads> cd mlops-docker-example/docker  
D:/Downloads/mlops-docker-example/docker>

## 2. Construcción de la imagen

Ejecutar el siguiente comando para construir la imagen Docker:

D:/Downloads/mlops-docker-example/docker> docker build -t inference-python-test .

- `-t inference-python-test` etiqueta la imagen con ese nombre.  
- El contexto `.` incluye todos los archivos en `docker/` (Dockerfile, inferencia.py, pipeline.pkl, transformadores/, etc.).

## 3. Ejecución del contenedor

Para montar la carpeta local `files/` dentro del contenedor en `/files`, ejecutar:

D:/Downloads/mlops-docker-example/docker> docker run --rm ^
  -v "D:/Downloads/mlops-docker-example/files:/files" ^
  inference-python-test

- `--rm` elimina el contenedor al finalizar la ejecución.  
- `-v "<host_path>/files:/files"` monta la carpeta de entrada/salida.  
- El script leerá `/files/input.csv` y generará `/files/output.csv`.

## 4. Verificación de resultados

Al finalizar la ejecución, se generará el archivo `output.csv` dentro de la carpeta local `files/`.

Asegúrese de que el archivo `input.csv` contenga todas las columnas requeridas por el pipeline del modelo.

Si se modifica el nombre de la imagen (etiqueta `-t`), se debe utilizar el mismo nombre en el comando `docker run`.
