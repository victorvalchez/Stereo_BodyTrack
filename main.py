import sys
from stereo_bodytrack import run_mp
from utils import get_projection_matrix, write_keypoints_to_disk

if __name__ == '__main__':

    # He creado dos videos de muestra para probar el código. Si no se proporciona ningún ID de cámara, se utilizarán estos videos.
    camera1_input = './media/video_cam0.mp4'
    camera2_input = './media/video_cam1.mp4'

    # Obtener el ID de la cámara de la línea de comandos
    if len(sys.argv) == 3:
        camera1_input = int(sys.argv[1])
        camera2_input = int(sys.argv[2])
    else:
        print('No se ha proporcionado ningún ID.\n\Utilización: python main.py <camera1_id> <camera2_id>')
        print('Usando videos de ejemplo...\n')

    # Get projection matrices
    # Warning!: P0 should be the projection matrix of the front camera so that the representation in Unity is facing forward. 
    #   Change the camera_id if needed
    P0 = get_projection_matrix(1)
    P1 = get_projection_matrix(0)

    kpts_cam0, kpts_cam1, kpts_3d = run_mp(camera1_input, camera2_input, P0, P1)

    # Store the obtained data for other purposes
    write_keypoints_to_disk('./data/kpts_cam0.dat', kpts_cam0)
    write_keypoints_to_disk('./data/kpts_cam1.dat', kpts_cam1)
    write_keypoints_to_disk('./data/kpts_3d.dat', kpts_3d)