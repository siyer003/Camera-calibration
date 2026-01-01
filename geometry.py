import torch
from typing import Tuple

'''
Please do Not change or add any imports. 
'''

# --------------------------------------------------- Rotation ----------------------------------------------------------

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 tensor represents the rotation matrix from xyz to XYZ.
    '''
    rot_xyz2XYZ = torch.eye(3, dtype=torch.float32)
    # Your implementation start here
    if alpha < 0 or alpha > 90 or beta < 0 or beta > 90 or gamma < 0 or gamma > 90:
        raise ValueError("alpha, beta, gamma should be in range [0, 90] to avoid gimbal lock")
    
    # Converting the  angles from degrees to radians
    x_rad = torch.deg2rad(torch.tensor(float(alpha), dtype=torch.float32))
    y_rad = torch.deg2rad(torch.tensor(float(beta), dtype=torch.float32))
    z_rad = torch.deg2rad(torch.tensor(float(gamma), dtype=torch.float32))

    Rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(x_rad), -torch.sin(x_rad)],
                       [0, torch.sin(x_rad), torch.cos(x_rad)]], dtype=torch.float32)
    Ry = torch.tensor([[torch.cos(y_rad), 0, torch.sin(y_rad)],
                       [0, 1, 0],
                       [-torch.sin(y_rad), 0, torch.cos(y_rad)]], dtype=torch.float32)
    Rz = torch.tensor([[torch.cos(z_rad), -torch.sin(z_rad), 0],
                          [torch.sin(z_rad), torch.cos(z_rad), 0],
                          [0, 0, 1]], dtype=torch.float32)
    # The rotation order is XYZ = Rx * Ry * Rz
    rot_xyz2XYZ = torch.mm(Rx, torch.mm(Ry, Rz))
    return rot_xyz2XYZ


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 tensor represents the rotation matrix from XYZ to xyz.
    '''
    rot_XYZ2xyz = torch.eye(3, dtype=torch.float32)
    # Your implementation start here:
    if alpha < 0 or alpha > 90 or beta < 0 or beta > 90 or gamma < 0 or gamma > 90:
        raise ValueError("alpha, beta, gamma should be in range [0, 90] to avoid gimbal lock")
    
    # The rotation matrix XYZ to xyz is the inverse of xyz to XYZ
    # And the inverse of a rotation matrix is its transpose as rotation matrices are orthogonal matrices
    rot_mat = findRot_xyz2XYZ(alpha, beta, gamma) # Rotation matrix from xyz to XYZ
    rot_XYZ2xyz = rot_mat.T # Transpose of the rotation matrix is the rotation matrix from XYZ to xyz
    return rot_XYZ2xyz

import numpy as np
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER,findChessboardCorners, cornerSubPix, drawChessboardCorners

# --------------------------------------------------- Camera Calibration ----------------------------------------------------------

def find_corner_img_coord(image: torch.Tensor) -> torch.Tensor:
    '''
    Args: 
        image: Input image of size 3xMxN.
        M is the height of the image.
        N is the width of the image.
        3 is the channel of the image.

    Return:
        A tensor of size 18x2 that represents the 18 checkerboard corners' pixel coordinates. 
        The pixel coordinate is as usually defined such that the top-left corner is (0, 0)
        and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = torch.zeros(18, 2, dtype=torch.float32)
    # print(image.shape)
    gray_img = image.squeeze(0).numpy().astype(np.uint8)
    pattern_size = (7,3)  # Number of inner corners per a chessboard row and column
        
   # Detect corners
    ret, corners = findChessboardCorners(gray_img, pattern_size, None)

    if not ret:
        print("Chessboard not found!")
        return img_coord
    # Refine corners
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_subpix = cornerSubPix(gray_img, corners, (11,11), (-1,-1), criteria)
    
    # # Drawing for debug
    # vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # drawChessboardCorners(vis, pattern_size, corners_subpix, ret)
    # cv2.imwrite("detected_corners.png", vis)
    
    # We reshape corners to a 2D array for easier indexing
    corners2 = corners_subpix.reshape(-1, 2)

    # We need to select the 18 corners that correspond to the two planes, so we select them manually
    zy_indices = [0,1,2, 7,8,9, 14,15,16]
    xz_indices = [4,5,6, 11,12,13, 18,19,20]

    corners_zy = corners2[zy_indices]
    corners_xz = corners2[xz_indices]

    # We concatenate the two sets of corners to get the final 18 corners
    img_coord = np.concatenate((corners_zy, corners_xz))

    # # Visualizing the new set of selected corners. 
    # vis_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    # cv2.drawChessboardCorners(vis_img, (3, 3), corners_zy.reshape(-1, 1, 2), True)
    # cv2.drawChessboardCorners(vis_img, (3, 3), corners_xz.reshape(-1, 1, 2), True)
    # cv2.imwrite("test.png", vis_img)

    # Converting to torch tensor
    img_coord = torch.from_numpy(img_coord.astype(np.float32))

    return img_coord


def find_corner_world_coord(img_coord: torch.Tensor) -> torch.Tensor:
    '''
    Args: 
        img_coord: The image coordinate of the corners.
        Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.

    Return:
        A torch.Tensor of size 18x3 that represents the 18
        (21 detected points minus 3 points on the z axis look at the figure in the documentation carefully)... 
        ...checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image.
        The output results should be in milimeters.
    '''
    world_coord = torch.zeros(18, 3, dtype=torch.float32)

    # You can only use torch in this function
    # Your implementation start here:
    
    #Since the grid size is 10mm, we can directly define the world coordinates of the corners. 
    # Wrt to the origin (0,0,0) at the corner where the two planes meet.
    XZ_plane = torch.tensor([
        [10, 0, 30], [20, 0, 30], [30, 0, 30],
        [10, 0, 20], [20, 0, 20], [30, 0, 20],
        [10, 0, 10], [20, 0, 10], [30, 0, 10]
    ], dtype=torch.float32)
    
    YZ_plane = torch.tensor([
        [0, 30, 30], [0, 20, 30], [0, 10, 30],
        [0, 30, 20], [0, 20, 20], [0, 10, 20],
        [0, 30, 10], [0, 20, 10], [0, 10, 10]
    ], dtype=torch.float32)
    
    world_coord = torch.cat((YZ_plane, XZ_plane), dim=0)
    
    return world_coord


def find_intrinsic(img_coord: torch.Tensor, world_coord: torch.Tensor) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 18 point to calculate the intrinsic parameters.

    Args: 
        img_coord: The image coordinate of the 18 corners. This is a 18x2 tensor.
        world_coord: The world coordinate of the 18 corners. This is a 18x3 tensor.

    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    M = determine_projection_matrix(img_coord, world_coord)
    fx, fy, cx, cy = determine_K_matrix(M)
    
    return fx, fy, cx, cy


def find_extrinsic(img_coord: torch.Tensor, world_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Use the image coordinates, world coordinates of the 18 point and the intrinsic
    parameters to calculate the extrinsic parameters.

    Args: 
        img_coord: The image coordinate of the 18 corners. This is a 18x2 tensor.
        world_coord: The world coordinate of the 18 corners. This is a 18x3 tensor.
    Returns:
        R: The rotation matrix of the extrinsic parameters.
            It is a 3x3 tensor.
        T: The translation matrix of the extrinsic parameters.
            It is a 1-dimensional tensor with length of 3.
    '''
    R = torch.eye(3, dtype=torch.float32)
    T = torch.zeros(3, dtype=torch.float32)

    M = determine_projection_matrix(img_coord, world_coord)
    fx, fy, cx, cy = determine_K_matrix(M)
    K = torch.tensor([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]], dtype=torch.float32)
    K_inv = torch.inverse(K)
    RT = torch.mm(K_inv, M)
    R = RT[:, :3] # Rotation matrix is the first three columns of RT
    T = RT[:, 3] # Translation vector is the last column of RT
   
    # # Checks to see if R is valid rotation matrix
    # I = torch.eye(3, dtype=torch.float32)
    # assert torch.allclose(R.T @ R, I, atol=1e-2), "R is not orthogonal"
    # assert torch.allclose(torch.det(R), torch.tensor(1.0), atol=1e-2), "det(R) != 1"
    # # print(torch.det(R))
    return R, T


def determine_projection_matrix(img_coord: torch.Tensor, world_coord: torch.Tensor) -> torch.Tensor:
    N = img_coord.shape[0]
    A = torch.zeros((2 * N, 12), dtype=torch.float64)

    for i in range(N):
        X, Y, Z = world_coord[i]
        u, v = img_coord[i]
        A[2*i] = torch.tensor([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u], dtype=torch.float64)
        A[2*i+1] = torch.tensor([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v], dtype=torch.float64)

    # Performing SVD
    U, S, Vh = torch.linalg.svd(A)
    P = Vh[-1, :]          # last row is the solution
    if P[-1] < 0:
        P = -P             # This fixes the sign ambiguity as the last element should be positive. I was incorrectly getting a negative value for Rotation matrix determinant before this fix.
    M = P.reshape(3, 4)

    # Normalizing the projection matrix
    m3 = M[2, :3]
    M = M / torch.norm(m3)

    return M.float()

def determine_K_matrix(M: torch.Tensor) -> Tuple[float, float, float, float]:
    # Separting the rows of M
    m1 = M[0, :3]
    m2 = M[1, :3]
    m3 = M[2, :3]
    # Using the formulas to compute fx, fy, cx, cy
    cx = torch.dot(m1, m3)
    cy = torch.dot(m2, m3)
    
    fx = torch.norm(m1 - cx * m3)
    fy = torch.norm(m2 - cy * m3)
    
    fx = fx.item()
    fy = fy.item()
    cx = cx.item()
    cy = cy.item()

    return fx, fy, cx, cy
#---------------------------------------------------------------------------------------------------------------------
