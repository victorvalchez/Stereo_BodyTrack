import numpy as np
import matplotlib.pyplot as plt
from stereo_bodytrack import run_mp  # Asegúrate de importar la función correctamente desde tu script principal
import mediapipe as mp
from utils import get_projection_matrix
import cv2 as cv
import sys

def calculate_distance_3d(point1, point2):
    """
    Calcula la distancia euclidiana en 3D entre dos puntos.

    :param point1: Primer punto en 3D (x, y, z).
    :param point2: Segundo punto en 3D (x, y, z).
    :return: distancia euclidiana entre los dos puntos en 3D.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def parse_keypoints_3d(keypoints_3d, index):
    """
    Extrae las coordenadas (x, y, z) de un punto clave específico desde el array lineal.

    :param keypoints_3d: array lineal con los puntos 3D.
    :param index: índice del punto clave (0 a 32).
    :return: una tupla con las coordenadas (x, y, z).
    """
    base_index = index * 3
    return keypoints_3d[base_index:base_index + 3]

def plot_distances(distances, title="3D Keypoint Distances Over Time"):
    """
    Grafica las distancias entre puntos clave en 3D a lo largo del tiempo.
    
    :param distances: array de distancias.
    :param title: título del gráfico.
    """
    plt.figure(figsize=(10, 6))
    for label, dist in distances.items():
        plt.plot(dist, label=label)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Distance (3D)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def remove_outliers_iqr(data):
    """
    Elimina outliers de los datos usando el método IQR.

    :param data: Lista de valores.
    :return: Lista filtrada sin outliers.
    """
    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (1.5 * iqr)
    upper_bound = quartile_3 + (1.5 * iqr)
    return [x for x in data if lower_bound <= x <= upper_bound]

def test_run(kpts_3d):
    # Lista de distancias para almacenar las mediciones
    distances = {
        'Right Wrist to Left Wrist': [],
        'Right Elbow to Left Elbow': [],
        'Right Shoulder to Left Shoulder': []
    }
    
    mp_pose = mp.solutions.pose

    # Calcular las distancias entre puntos clave relevantes en 3D
    for frame_3d_keypoints in kpts_3d:
        # print("Lista de keypoints: ", frame_3d_keypoints)
        # Extraer las coordenadas de cada punto clave relevante
        rw = frame_3d_keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        #print("Punto clave de muñeca derecha: ", rw)
        lw = frame_3d_keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
        #print("Punto clave de muñeca izquierda: ", lw)
        re = frame_3d_keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        le = frame_3d_keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        rs = frame_3d_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        ls = frame_3d_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        
        # Distancia entre las muñecas derecha e izquierda en 3D
        rw_lw_dist = calculate_distance_3d(rw, lw)
        distances['Right Wrist to Left Wrist'].append(rw_lw_dist)
        
        # Distancia entre los codos derecho e izquierdo en 3D
        re_le_dist = calculate_distance_3d(re, le)
        distances['Right Elbow to Left Elbow'].append(re_le_dist)
        
        # Distancia entre los hombros derecho e izquierdo en 3D
        rs_ls_dist = calculate_distance_3d(rs, ls)
        distances['Right Shoulder to Left Shoulder'].append(rs_ls_dist)
        
    # Filtrar outliers usando IQR antes de graficar
    for key in distances:
        distances[key] = remove_outliers_iqr(distances[key])
    
    # Graficar las distancias 3D
    plot_distances(distances, title="3D Keypoint Distances Across Frames")

if __name__ == "__main__":
    # He creado dos videos de muestra para probar el código. Si no se proporciona ningún ID de cámara, se utilizarán estos videos.
    camera1_input = './media/video_cam0.mp4'
    camera2_input = './media/video_cam1.mp4'

    # Obtener el ID de la cámara de la línea de comandos
    if len(sys.argv) == 3:
        camera1_input = int(sys.argv[1])
        camera2_input = int(sys.argv[2])
        P0 = get_projection_matrix(int(sys.argv[1]))
        P1 = get_projection_matrix(int(sys.argv[2]))
    else:
        print('No se ha proporcionado ningún ID.\n\t Utilización: python main.py <camera1_id> <camera2_id>')
        print('Usando videos de ejemplo...\n')
        P0 = get_projection_matrix(1)
        P1 = get_projection_matrix(0)

    # Get projection matrices
    # Warning!: P0 should be the projection matrix of the front camera so that the representation in Unity is facing forward. 
    #   Change the camera_id if needed
    

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(camera1_input, camera2_input, P0, P1)
    test_run(kpts_3d)
