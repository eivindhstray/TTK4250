import numpy as np
from mytypes import ArrayLike


def cross_product_matrix(n: ArrayLike, debug: bool = True) -> np.ndarray:

    ''' 
    VERIFIED WORKING
    '''
    
    assert len(n) == 3, f"utils.cross_product_matrix: Vector not of length 3: {n}"
    vector = np.array(n, dtype=float).reshape(3)

    S = np.zeros((3, 3))  # TODO: Create the cross product matrix
    n_1 = n[0]
    n_2 = n[1]
    n_3 = n[2]
    #S[0,1] = - n_3
    #S[1,0] = n_3
    #S[0,2] = n_2
    #S[2,0] = - n_2
    #S[1,2] = - n_1
    #S[2,1] = n_1
    S = np.array([[0, -n_3, n_2],
                  [n_3, 0, -n_1],
                  [-n_2, n_1, 0]], 
                  dtype=float)
    if debug:
        assert S.shape == (
            3,
            3,
        ), f"utils.cross_product_matrix: Result is not a 3x3 matrix: {S}, \n{S.shape}"
        assert np.allclose(
            S.T, -S
        ), f"utils.cross_product_matrix: Result is not skew-symmetric: {S}"

    return S

