# Mejora de Identificación Postural Usando Sensor Fusion para Neurorrehabilitación

# Descripción

Este proyecto es un sistema de seguimiento corporal estéreo desarrollado con Python y Unity. Permite rastrear y proyectar puntos 3D a partir de las señales de cámaras utilizando MediaPipe, proporcionando visualizaciones en tiempo real de los movimientos corporales.

# Requisitos Previos

Antes de ejecutar el proyecto, necesitas instalar varios programas, paquetes y dependencias:

- **Python 3.12.4** (o superior)
- **Visual Studio Code** (opcional, pero recomendado)
- **Unity 2022.3.35f1** (para visualización 3D)

# Instalación

## 1. Instalar Python y Clonar el Proyecto
Clona el repositorio del proyecto utilizando el siguiente comando:

```bash
git clone https://github.com/victorvalchez/Stereo_BodyTrack
```

## 2. Instalar Dependencias
Para gestionar las dependencias, se recomienda crear un entorno virtual. Puedes dejar que Visual Studio Code lo haga automáticamente o hacerlo manualmente ejecutando:
```bash
pip install -r requirements.txt
```

## 3. Instalar Unity
Descarga Unity desde el sitio oficial e instálalo. Una vez instalado, usa Unity Hub para añadir el proyecto haciendo clic en el botón "Add" y seleccionando el directorio del proyecto que puedes descargar en [Descargar Proyecto](https://drive.google.com/drive/folders/11CK9SoaLqwoo_ZBa9L_b_ySLqcZdeiGH?usp=sharing).

Si se te solicita instalar una versión específica de Unity, asegúrate de descargar e instalar la versión requerida (2022.3.35f1).

# Uso

## 1. Estructura del Proyecto

Tu proyecto debería tener la siguiente estructura:

```plaintext
/.
├── calibration/
│   ├── calibration_params.yaml
│   ├── calibration.py
├── camera_parameters/
│   ├── camera0_intrinsics.dat
│   ├── camera0_rot_trans.dat
│   ├── camera1_intrinsics.dat
│   ├── camera1_rot_trans.dat
├── media/
│   ├── video_cam0.mp4
│   ├── video_cam1.mp4
├── Unity/
│   ├── AnimationCode.cs
│   ├── lineCode.cs
│   ├── UPDReceive.cs
├── main.py
├── record_test.py
├── requirements.txt
├── stereo_bodytrack.py
├── utils.py
```
## 2. Calibración de las Cámaras
**Nota**: Este paso no es necesario si estás utilizando los videos de prueba proporcionados.

La calibración es esencial para una proyección 3D precisa. Modifica el archivo calibration_params.yaml para establecer los IDs de las cámaras y la resolución según tu configuración.

También necesitarás un patrón de calibración. Se recomienda un patrón de 5x8 con cuadrados de 32mm. Puedes descargar el patrón desde este [enlace](https://calib.io/pages/camera-calibration-pattern-generator?srsltid=AfmBOorWAdZwvts8H00XiOTF9MQ__J5qtyfBtNdpWwUR8ZQE7PjaZoGq).

Una vez listo, ejecuta el siguiente comando para iniciar el proceso de calibración:
```bash
python ./calibration/calibration.py ./calibration/calibration_params.yaml
```
El proceso de calibración guardará los parámetros de las cámaras en el directorio ```camera_parameters/```.

## 3. Ejecución del Proyecto
Después de la calibración, puedes ejecutar el proyecto utilizando Unity y Python:

1. Abre el proyecto en Unity y espera a que se inicialice.
2. Ejecuta el siguiente comando en la terminal para iniciar el script de Python:
```bash
python ./main.py <id_cam0> <id_cam1>
```
Puedes omitir los IDs de las cámaras si estás utilizando videos pregrabados.

3. Establece el puerto del servidor UDP en Unity para que coincida con el puerto usado en el script de Python (el puerto predeterminado es 12345).
4. Presiona el botón "Play" en Unity para comenzar a visualizar los resultados.

# Notas Adicionales
- MediaPipe puede ralentizar el procesamiento de videos, lo que podría llevar a visualizaciones menos precisas cuando se usan videos pregrabados.
- Asegúrate de que el patrón de calibración esté plano y sin distorsiones para obtener resultados precisos.
