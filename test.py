import numpy as np
import matplotlib.pyplot as plt
from stereo_bodytrack import run_mp  # Asegúrate de importar la función correctamente desde tu script principal
import mediapipe as mp
from utils import get_projection_matrix
import cv2 as cv
import sys

def calculate_distance_3d(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def parse_keypoints_3d(keypoints_3d, index):
    base_index = index * 3
    return keypoints_3d[base_index:base_index + 3]

def plot_distances_with_reference(distances, reference_lines, title="3D Keypoint Distances Over Time"):
    plt.figure(figsize=(10, 6))
    
    # Graficar las distancias calculadas
    for label, dist in distances.items():
        plt.plot(dist, label=label)

    # Graficar las líneas de referencia que varían en función del frame
    for label, ref in reference_lines.items():
        plt.plot(ref, linestyle='--', label=f'{label} Reference', color='blue')

    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Distancia en cm")
    plt.legend()
    plt.grid(True)
    plt.show()

def remove_outliers_iqr(data):
    quartile_1, quartile_3 = np.percentile(data, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (1.5 * iqr)
    upper_bound = quartile_3 + (1.5 * iqr)
    return [x for x in data if lower_bound <= x <= upper_bound]

def test_run(kpts_3d):
    distances = {
        'Muñeca Derecha a Muñeca Izquierda': [],
        'Codo Derecho a Codo Izquierdo': [],
        'Hombro Derecho a Hombro Izquierdo': []
    }
    
    scale_factor = 45 / 20  # Factor de escala = 
    
    mp_pose = mp.solutions.pose

    for frame_3d_keypoints in kpts_3d:
        rw = frame_3d_keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        lw = frame_3d_keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
        re = frame_3d_keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        le = frame_3d_keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        rs = frame_3d_keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        ls = frame_3d_keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        
        rw_lw_dist = calculate_distance_3d(rw, lw) * scale_factor
        distances['Muñeca Derecha a Muñeca Izquierda'].append(rw_lw_dist)
        
        re_le_dist = calculate_distance_3d(re, le) * scale_factor
        distances['Codo Derecho a Codo Izquierdo'].append(re_le_dist)
        
        rs_ls_dist = calculate_distance_3d(rs, ls) * scale_factor
        distances['Hombro Derecho a Hombro Izquierdo'].append(rs_ls_dist)
    
    for key in distances:
        distances[key] = remove_outliers_iqr(distances[key])
    
    # Definir las líneas de referencia según los rangos de frames
    reference_lines = {
        'Muñeca Derecha a Muñeca Izquierda': [],
    }

    # Ejemplo de rangos de frames y distancias esperadas para las poses
    # Ajusta estos valores según tus necesidades
    frame_ranges = [
        (0, 50, 55),   # Frames 0 a 50: 55 cm (Manos relajadas hacia abajo)
        (51, 100, 125), # Frames 51 a 100: 125 cm (Mano derecha en T)
        (101, 150, 95), # Frames 101 a 150: 95 cm (Mano derecha al frente)
        (151, 200, 140)  # Frames 151 a 200: 140 cm (Mano derecha hacia arriba)
    ]
    
    num_frames = len(distances['Muñeca Derecha a Muñeca Izquierda'])

    for label in reference_lines:
        for i in range(num_frames):
            for start, end, value in frame_ranges:
                if start <= i <= end:
                    reference_lines[label].append(value)
                    break

    plot_distances_with_reference(distances, reference_lines, title="3D Keypoint Distances Across Frames")

if __name__ == "__main__":
    camera1_input = './media/video_cam0.mp4'
    camera2_input = './media/video_cam1.mp4'

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
    
    kpts_cam0, kpts_cam1, kpts_3d = run_mp(camera1_input, camera2_input, P0, P1)
    test_run(kpts_3d)
