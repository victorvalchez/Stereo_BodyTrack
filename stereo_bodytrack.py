import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from utils import DLT, get_projection_matrix, write_keypoints_to_disk, estimate_3d_point_from_single_view
import socket

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Ayudantes de dibujo
mp_drawing_styles = mp.solutions.drawing_styles  # Estilos de dibujo
mp_pose = mp.solutions.pose  # Modelo de pose de Mediapipe

frame_shape = [720, 1280]

# 32 puntos clave son detectados por mediapipe. Lo que significa que se detecta todo el cuerpo.
pose_keypoints = [i for i in range(33)]

# Crear un objeto de socket (UDP)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 12345)

def run_mp(input_stream1, input_stream2, P0, P1):
    # Flujo de video de entrada
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # Establecer la resolución de la cámara si se usa una webcam a 1280x720. Cualquier tamaño mayor causará algún retraso en la detección de manos
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # Crear objetos detectores de puntos clave del cuerpo.
    pose0 = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    pose1 = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)

    # Contenedores para los puntos clave detectados para cada cámara. Estos se llenan en cada cuadro.
    # Esto te llevará a problemas de memoria si ejecutas el programa sin detenerlo
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []
    while True:

        # Leer cuadros del flujo
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1: break

        # Recortar a 720x720.
        # Nota: los parámetros de calibración de la cámara están configurados a esta resolución. Si cambias esto, asegúrate de también cambiar los parámetros intrínsecos de la cámara
        # if frame0.shape[1] != 720:
            # frame0 = frame0[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]
            # frame1 = frame1[:,frame_shape[1]//2 - frame_shape[0]//2:frame_shape[1]//2 + frame_shape[0]//2]

        # Convertir la imagen BGR a RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # Para mejorar el rendimiento, opcionalmente marca la imagen como no escribible para
        # pasar por referencia.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        # Revertir cambios
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        # Verificar la detección de puntos clave
        frame0_keypoints = []
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints: continue # Solo guardar puntos clave que están indicados en pose_keypoints
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame0,(pxl_x, pxl_y), 3, (0,0,255), -1) # Agregar puntos de detección de puntos clave en la figura
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
        else:
            # Si no se encuentran puntos clave, simplemente llena los datos del cuadro con [-1,-1] para cada punto clave
            frame0_keypoints = [[-1, -1]]*len(pose_keypoints)

        # Esto mantendrá los puntos clave de este cuadro en memoria
        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                if i not in pose_keypoints: continue
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame1,(pxl_x, pxl_y), 3, (0,0,255), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)

        else:
            # Si no se encuentran puntos clave, simplemente llena los datos del cuadro con [-1,-1] para cada punto clave
            frame1_keypoints = [[-1, -1]]*len(pose_keypoints)

        # Actualizar el contenedor de puntos clave
        kpts_cam1.append(frame1_keypoints)

        # Nuestra propia implementación de DLT
        # Calcular la posición 3D
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT(P0, P1, uv1, uv2)
            frame_p3ds.append(_p3d)

        frame_p3ds = np.array(frame_p3ds).reshape((33, 3))
        
        # Enviar los datos a Unity a través de UDP
        s.sendto(str(frame_p3ds).encode(), server_address)

        # Agregar los puntos clave 3D a la lista
        kpts_3d.append(frame_p3ds)

        # Descomentar estos si quieres ver las detecciones completas de puntos clave
        mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        #
        mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv.imshow('cam1', frame1)
        cv.imshow('cam0', frame0)

        k = cv.waitKey(1)
        if k & 0xFF == 27: break # 27 es la tecla ESC.


    cv.destroyAllWindows()
    for cap in caps:
        cap.release()

    
    return np.array(kpts_cam0), np.array(kpts_cam1), np.array(kpts_3d)