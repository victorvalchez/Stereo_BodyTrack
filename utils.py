import numpy as np

def _make_homogeneous_rep_matrix(R, t):
    """
    Crea una matriz de representación homogénea a partir de una matriz de rotación y un vector de traslación.
    
    Parámetros:
    R: Matriz de rotación (3x3)
    t: Vector de traslación (3,)
    
    Retorna:
    Matriz de representación homogénea (4x4)
    """
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

# Transformación lineal directa
def DLT(P1, P2, point1, point2):
    """
    Calcula la posición 3D de un punto a partir de dos vistas utilizando la Transformación Lineal Directa (DLT).
    
    Parámetros:
    P1: Matriz de proyección de la primera cámara
    P2: Matriz de proyección de la segunda cámara
    point1: Coordenadas 2D del punto en la primera vista
    point2: Coordenadas 2D del punto en la segunda vista
    
    Retorna:
    Coordenadas 3D del punto
    """
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    from scipy import linalg
    U, s, Vh = linalg.svd(B, full_matrices = False)

    return Vh[3,0:3]/Vh[3,3]

def read_camera_parameters(camera_id):
    """
    Lee los parámetros intrínsecos de la cámara desde un archivo.
    
    Parámetros:
    camera_id: ID de la cámara
    
    Retorna:
    Matriz de parámetros intrínsecos y coeficientes de distorsión
    """
    inf = open('camera_parameters/camera' + str(camera_id) + '_intrinsics.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):
    """
    Lee los parámetros de rotación y traslación de la cámara desde un archivo.
    
    Parámetros:
    camera_id: ID de la cámara
    savefolder: Carpeta donde se guardan los archivos de parámetros
    
    Retorna:
    Matriz de rotación y vector de traslación
    """
    inf = open(savefolder + 'camera' + str(camera_id) + '_rot_trans.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    """
    Convierte puntos a coordenadas homogéneas.
    
    Parámetros:
    pts: Puntos en coordenadas cartesianas
    
    Retorna:
    Puntos en coordenadas homogéneas
    """
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):
    """
    Calcula la matriz de proyección de la cámara.
    
    Parámetros:
    camera_id: ID de la cámara
    
    Retorna:
    Matriz de proyección de la cámara
    """
    # Leer parámetros de la cámara
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    # Calcular matriz de proyección
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    """
    Escribe los puntos clave en un archivo.
    
    Parámetros:
    filename: Nombre del archivo
    kpts: Puntos clave a escribir
    
    Retorna:
    None
    """
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()
    
def estimate_3d_point_from_single_view(P, uv, depth=1.0):
    """
    Estima el punto 3D desde una sola vista de cámara utilizando una profundidad predefinida.
    
    Parámetros:
    P: Matriz de proyección de la cámara
    uv: Punto clave 2D
    depth: Valor de profundidad asumido
    
    Retorna:
    Coordenadas 3D del punto
    """
    # Convertir punto 2D a coordenadas homogéneas
    uv_homogeneous = np.array([uv[0], uv[1], 1.0])
    
    # Retroproyectar el punto 2D al espacio 3D
    # P_inv es la pseudo-inversa de la matriz de proyección de la cámara
    P_inv = np.linalg.pinv(P)
    point_3d_homogeneous = P_inv @ uv_homogeneous
    
    # Escalar el punto por la profundidad asumida
    point_3d = point_3d_homogeneous[:3] * depth / point_3d_homogeneous[2]
    
    return point_3d

if __name__ == '__main__':
    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)