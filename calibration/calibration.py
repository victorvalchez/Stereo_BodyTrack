import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os

# Contiene los parámetros de calibración de las cámaras que se encuentran en el archivo calibration_settings.yaml
calibration_settings = {}

# Con las matrices de proyección de las cámaras, se puede triangular el punto 3D. (Como hace cv2.triangulatePoints)
def DLT(P1, P2, point1, point2):
    '''
    Función para triangular un punto 3D a partir de las matrices de proyección de las cámaras y 
    las coordenadas de los puntos en las imágenes.
    '''
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    # print('Triangulated point: ', Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


# Cargamos los parámetros de calibración de la cámara
def parse_calibration_settings_file(filename):
    '''
    Función para cargar los parámetros de calibración de las cámaras desde un archivo de configuración.
    '''
    global calibration_settings

    if not os.path.exists(filename):
        print('El archivo no existe:', filename)
        quit()
    
    else:
        # print('Using for calibration settings: ', filename)
        with open(filename) as f:
            calibration_settings = yaml.safe_load(f)

    # Comprobamos que las claves necesarias estén presentes en el archivo de configuración (por si hemos seleccionado uno incorrecto)
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


# Cargamos los parámetros de calibración de la cámara y abrimos la captura de video
def save_frames_single_camera(camera_name):
    '''
    Función para guardar los frames de calibración de una cámara.
    '''
    # Creamos el directorio para guardar los frames si no existe
    if not os.path.exists('./calibration/frames'):
        os.mkdir('./calibration/frames')

    # Obtenemos los parámetros de calibración de la cámara
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # Abrimos la captura de video y establecemos la resolución
    # Note: if unsupported resolution is used, this does NOT raise an error.
    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width)
    cap.set(4, height)
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    # Bucle para guardar los frames de la cámara
    while True and saved_count < number_to_save:
        '''
        Cada cierto tiempo, se guarda un frame de la cámara. 
        El tiempo de espera entre frames se establece en cooldown_time.
        '''
        ret, frame = cap.read()
        if ret == False:
            # Si no se recibe ningún dato de video de la cámara, salimos del programa.
            print("No se han obtenido datos de video de la cámara. Saliendo...")
            quit()

        frame_small = cv.resize(frame, None, fx = 1/view_resize, fy=1/view_resize)

        # Esperamos a que se presione la tecla de espacio para comenzar la recolección de datos
        if not start:
            cv.putText(frame_small, "Presiona ESPACIO para empezar a recoger frames.", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            # Guardamos el frame cuando el cooldown llega a 0.
            if cooldown <= 0:
                savename = os.path.join('./calibration/frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame)
                saved_count += 1
                cooldown = cooldown_time

        # Mostramos el frame en la ventana
        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)
        
        if k == 27: # ESC
            # Si se presiona ESC en cualquier momento, el programa se cerrará.
            quit()

        if k == 32: # ESPACIO
            # Presionar ESPACIO para comenzar la recolección de datos
            start = True

    cv.destroyAllWindows()


# Calibrar una cámara para obtener los parámetros intrínsecos.
def calibrate_camera_for_intrinsic_parameters(frames_path):
    '''
    Función para calibrar una cámara y obtener los parámetros intrínsecos.
    '''
    # NOTE: frames_path contiene la ubicación de los frames de esa cámara: "frames/camera0*".
    images_names = glob.glob(frames_path)

    # Leemos cada frame
    images = [cv.imread(imagen, 1) for imagen in images_names]

    # Criterios usados por el patrón de ajedrez 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # Coordenadas de los cuadrados en el espacio del mundo del tablero de ajedrez
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    # Dimensiones de los frames. Los frames deben ser del mismo tamaño.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Coordenadas de los pixeles del tablero de ajedrez
    imgpoints = [] # 2d points in image plane.

    # Coordenadas del tablero de ajedrez en el espacio del mundo del tablero de ajedrez.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Encuentra las esquinas del tablero de ajedrez
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Tamaño de la ventana de convolución para mejorar las coordenadas del tablero de ajedrez
            conv_size = (11, 11)

            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'Si los puntos detectados no son precisos presiona S para omitir este frame.',
                       (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('Frame', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('Omitiendo frame...')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('RMSE:', ret)
    print('Matriz de la cámara:\n', cmtx)
    print('Coeficientes de distorsión:', dist)

    return cmtx, dist

# Guardar los parámetros intrínsecos de la cámara
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):
    '''
    Función para guardar los parámetros intrínsecos de la cámara.
    '''
    if not os.path.exists('./camera_parameters'):
        os.mkdir('./camera_parameters')

    out_filename = os.path.join('./camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


# Guardar frames tomados por las dos cámaras a la vez
def save_frames_two_cams(camera0_name, camera1_name):
    '''
    Función para guardar los frames de calibración emparejados para ambas cámaras.
    '''
    # Creamos el directorio para guardar los frames stereo si no existe
    if not os.path.exists('./calibration/frames_pair'):
        os.mkdir('./calibration/frames_pair')

    # Ajustes de la calibración
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']    
    number_to_save = calibration_settings['stereo_calibration_frames']

    # Abrimos las capturas de video
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    # Establecemos la resolución de las cámaras
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    cooldown = cooldown_time
    start = False
    saved_count = 0
    while True and saved_count < number_to_save:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('No se han obtenido datos de video de las cámaras. Saliendo...')
            quit()

        frame0_small = cv.resize(frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv.putText(frame0_small, "Asegúrate de que ambas cámaras vean el patrón correctamente.", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.putText(frame0_small, "Presiona ESPACIO para empezar a recoger frames.", (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
        
        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            
            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50,100), cv.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

            #save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('./calibration/frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame0)

                savename = os.path.join('./calibration/frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, frame1)

                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)
        
        if k == 27:
            #if ESC is pressed at any time, the program will exit.
            quit()

        if k == 32:
            #Press spacebar to start data collection
            start = True

    cv.destroyAllWindows()


# Abrimos los frames de calibración emparejados y calibramos el estéreo para las transformaciones de coordenadas de cam0 a cam1
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    '''
    Función para calibrar el estéreo y obtener las matrices de rotación y traslación.
    '''
    # Obtenemos los nombres de los frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    # Leemos los frames
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    # NOTE: Cambiar estos valores si la calibración no es buena
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Obtener los parámetros del tablero de ajedrez
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    # Coordenadas de los cuadrados en el espacio del mundo del tablero de ajedrez
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('Frames0', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('Frames1', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('Omitiendo frame...')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('RSME: ', ret)
    
    cv.destroyAllWindows()
    return R, T

# Convertimos la matriz de rotación R y el vector de traslación T en una matriz de representación homogénea
def _make_homogeneous_rep_matrix(R, t):
    '''
    Función para convertir la matriz de rotación R y el vector de traslación T en una matriz de representación homogénea.
    '''
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P

# Convertimos los datos de calibración de la cámara en una matriz de proyección
def get_projection_matrix(cmtx, R, T):
    '''
    Función para obtener la matriz de proyección de la cámara.
    '''
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# Después de calibrar, podemos ver los ejes de coordenadas desplazados en los feeds de video directamente como comprobación
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 95.):
    
    # TODO: COmentar o eliminar
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)  # Obtenemos las matrices de proyección
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27: break

    cv.destroyAllWindows()
    
def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    # Crear el directorio para guardar los parámetros de calibración si no existe
    if not os.path.exists('./camera_parameters'):
        os.mkdir('./camera_parameters')

    camera0_rot_trans_filename = os.path.join('./camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('./camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("Ejecutar con el nombre del archivo de configuración: 'python3 calibrate.py calibration_params.yaml'")
        quit()
    
    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])


    """Paso 1. Guardar los frames de calibración para cada cámara"""
    save_frames_single_camera('camera0') #save frames for camera0
    save_frames_single_camera('camera1') #save frames for camera1


    """Paso 2. Calibrar las cámaras para obtener los parámetros intrínsecos"""
    #camera0 intrinsics
    frames_path = os.path.join('./calibration/frames', 'camera0*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(frames_path) 
    save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    #camera1 intrinsics
    frames_path = os.path.join('./calibration/frames', 'camera1*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(frames_path)
    save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk


    """Paso 3. Guardar los frames de calibración emparejados para ambas cámaras"""
    save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Paso 4. Utilizar los frames emparejados para calibrar el estéreo y obtener las matrices de rotación y traslación"""
    frames_prefix_c0 = os.path.join('./calibration/frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('./calibration/frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Paso 5. Guardar los parámetros de calibración donde la cámara0 define el origen del sistema de coordenadas"""
    # La matriz de rotación y el vector de traslación de la cámara0 se establecen en 0 y 0 respectivamente
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))
    # Para evitar confusiones, la matriz de rotación y el vector de traslación de la cámara1 se renombran a R1 y T1
    R1 = R
    T1 = T 
    
    save_extrinsic_calibration_parameters(R0, T0, R1, T1)
    
    if calibration_settings['check_calibration']:
        # Comprobamos si la calibración es correcta
        camera0_data = [cmtx0, dist0, R0, T0]
        camera1_data = [cmtx1, dist1, R1, T1]
        check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 75.)