import numpy as np
import utils


def quaternion_product(ql: np.ndarray, qr: np.ndarray) -> np.ndarray:
    """Perform quaternion product according to either (10.21) or (10.34).

    Args:
        ql (np.ndarray): Left quaternion of the product of either shape (3,) (pure quaternion) or (4,)
        qr (np.ndarray): Right quaternion of the product of either shape (3,) (pure quaternion) or (4,)

    Raises:
        RuntimeError: Left or right quaternion are of the wrong shape
        AssertionError: Resulting quaternion is of wrong shape

    Returns:
        np.ndarray: Quaternion product of ql and qr of shape (4,)s
    """
    if ql.shape == (4,):
        eta_left = ql[0]
        epsilon_left = ql[1:].reshape((3, 1))
    elif ql.shape == (3,):
        eta_left = 0
        epsilon_left = ql.reshape((3, 1))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, left quaternion shape incorrect: {ql.shape}"
        )

    if qr.shape == (4,):
        q_right = qr.copy()
        eta_right = q_right[0]
        epsilon_right = qr[1:].reshape((3,1))
    elif qr.shape == (3,):
        eta_right = 0
        epsilon_right = qr.reshape((3,1))
    else:
        raise RuntimeError(
            f"utils.quaternion_product: Quaternion multiplication error, right quaternion wrong shape: {qr.shape}"
        )

    quaternion_product = np.zeros((4,1))  # TODO: Implement quaternion product
    quaternion_product[0] = (eta_left*eta_right - (epsilon_left.T)@epsilon_right)[0]
    quaternion_product[1:] = (eta_left*epsilon_right +eta_right*epsilon_left + utils.cross_product_matrix(epsilon_left)@epsilon_right)[0]
    
    # Ensure result is of correct shape
    quaternion_product = quaternion_product.ravel()
    assert quaternion_product.shape == (
        4,
    ), f"utils.quaternion_product: Quaternion multiplication error, result quaternion wrong shape: {quaternion_product.shape}"
    return quaternion_product


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray, debug: bool = True
) -> np.ndarray:
    """Convert a quaternion to a rotation matrix

    Args:
        quaternion (np.ndarray): Quaternion of either shape (3,) (pure quaternion) or (4,)
        debug (bool, optional): Debug flag, could speed up by setting to False. Defaults to True.

    Raises:
        RuntimeError: Quaternion is of the wrong shape
        AssertionError: Debug assert fails, rotation matrix is not element of SO(3)

    Returns:
        np.ndarray: Rotation matrix of shape (3, 3)
    """
    if quaternion.shape == (4,):
        eta = quaternion[0]
        epsilon = quaternion[1:]
    elif quaternion.shape == (3,):
        eta = 0
        epsilon = quaternion.copy()
    else:
        raise RuntimeError(
            f"quaternion.quaternion_to_rotation_matrix: Quaternion to multiplication error, quaternion shape incorrect: {quaternion.shape}"
        )

    R = np.zeros((3, 3))  # TODO: Convert from quaternion to rotation matrix

    I = np.identity(3)
    S_eps = utils.cross_product_matrix(epsilon)
    R += I + 2*eta*S_eps + 2*S_eps@S_eps

    if debug:
        assert np.allclose(
            np.linalg.det(R), 1
        ), f"quaternion.quaternion_to_rotation_matrix: Determinant of rotation matrix not close to 1"
        assert np.allclose(
            R.T, np.linalg.inv(R)
        ), f"quaternion.quaternion_to_rotation_matrix: Transpose of rotation matrix not close to inverse"

    return R


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion into euler angles

    Args:
        quaternion (np.ndarray): Quaternion of shape (4,)

    Returns:
        np.ndarray: Euler angles of shape (3,)
    """

    assert quaternion.shape == (
        4,
    ), f"quaternion.quaternion_to_euler: Quaternion shape incorrect {quaternion.shape}"

    eta = quaternion[0]
    e_1 = quaternion[1]
    e_2 = quaternion[2]
    e_3 = quaternion[3]

    
    phi = np.arctan2(2*(e_3*e_2+eta*e_1),eta**2-e_1**2-e_2**2+e_3**2)  # TODO: Convert from quaternion to euler angles
    theta = np.arcsin(2*(eta*e_2-e_1*e_3))  # TODO: Convert from quaternion to euler angles
    psi = np.arctan2(2*(e_1*e_2+eta*e_3),eta**2+e_1**2-e_2**2-e_3**2)  # TODO: Convert from quaternion to euler angles

    euler_angles = np.array([phi, theta, psi])
    assert euler_angles.shape == (
        3,
    ), f"quaternion.quaternion_to_euler: Euler angles shape incorrect: {euler_angles.shape}"

    return euler_angles


def euler_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """Convert euler angles into quaternion

    Args:
        euler_angles (np.ndarray): Euler angles of shape (3,)

    Returns:
        np.ndarray: Quaternion of shape (4,)
    """

    assert euler_angles.shape == (
        3,
    ), f"quaternion.euler_to_quaternion: euler_angles shape wrong {euler_angles.shape}"

    half_angles = 0.5 * euler_angles
    c_phi2, c_theta2, c_psi2 = np.cos(half_angles)
    s_phi2, s_theta2, s_psi2 = np.sin(half_angles)

    quaternion = np.array(
        [
            c_phi2 * c_theta2 * c_psi2 + s_phi2 * s_theta2 * s_psi2,
            s_phi2 * c_theta2 * c_psi2 - c_phi2 * s_theta2 * s_psi2,
            c_phi2 * s_theta2 * c_psi2 + s_phi2 * c_theta2 * s_psi2,
            c_phi2 * c_theta2 * s_psi2 - s_phi2 * s_theta2 * c_psi2,
        ]
    )

    assert quaternion.shape == (
        4,
    ), f"quaternion.euler_to_quaternion: Quaternion shape incorrect {quaternion.shape}"

    return quaternion
