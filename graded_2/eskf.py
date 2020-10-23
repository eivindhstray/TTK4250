# %% imports
from typing import Tuple, Sequence, Any
from dataclasses import dataclass, field
from cat_slice import CatSlice

import numpy as np
import scipy.linalg as la

from quaternion import (
    euler_to_quaternion,
    quaternion_product,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)

# from state import NominalIndex, ErrorIndex
from utils import cross_product_matrix


# %% indices
POS_IDX = CatSlice(start=0, stop=3)
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10)
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)

ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)


# %% The class
@dataclass
class ESKF:
    sigma_acc: float
    sigma_gyro: float

    sigma_acc_bias: float
    sigma_gyro_bias: float

    p_acc: float = 0
    p_gyro: float = 0

    S_a: np.ndarray = np.eye(3)
    S_g: np.ndarray = np.eye(3)
    debug: bool = True

    g: np.ndarray = np.array([0, 0, 9.82])  # Ja, i NED-land, der kan alt gå an

    Q_err: np.array = field(init=False, repr=False)

    def __post_init__(self):
        if self.debug:
            print(
                "ESKF in debug mode, some numeric properties are checked at the expense of calculation speed"
            )

        self.Q_err = (
            la.block_diag(
                self.sigma_acc * np.eye(3),
                self.sigma_gyro * np.eye(3),
                self.sigma_acc_bias * np.eye(3),
                self.sigma_gyro_bias * np.eye(3),
            )
            ** 2
        )

    def predict_nominal(
        self,
        x_nominal: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
    ) -> np.ndarray:
        """Discrete time prediction, equation (10.58)

        Args:
            x_nominal (np.ndarray): The nominal state to predict, shape (16,)
            acceleration (np.ndarray): The estimated acceleration in body for the predicted interval, shape (3,)
            omega (np.ndarray): The estimated rotation rate in body for the prediction interval, shape (3,)
            Ts (float): The sampling time

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            np.ndarray: The predicted nominal state, shape (16,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict_nominal: x_nominal incorrect shape {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.predict_nominal: acceleration incorrect shape {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.predict_nominal: omega incorrect shape {omega.shape}"

        # Extract states
        position = x_nominal[POS_IDX]
        velocity = x_nominal[VEL_IDX]
        quaternion = x_nominal[ATT_IDX]

       
        acceleration_bias = x_nominal[ACC_BIAS_IDX]
        gyroscope_bias = x_nominal[GYRO_BIAS_IDX]

        if self.debug:
            assert np.allclose(
                np.linalg.norm(quaternion), 1, rtol=0, atol=1e-15
            ), "ESKF.predict_nominal: Quaternion not normalized."
            assert np.allclose(
                np.sum(quaternion ** 2), 1, rtol=0, atol=1e-15
            ), "ESKF.predict_nominal: Quaternion not normalized and norm failed to catch it."

        R = quaternion_to_rotation_matrix(quaternion, debug=self.debug)

        position_prediction = position+Ts*velocity + ((1/2)*Ts**2)*(R@acceleration+self.g)   
        
        velocity_prediction = velocity+Ts*(R@acceleration+self.g)

        quaternion_prediction = np.array(
            [1, 0, 0, 0]
        )  # TODO: Calculate predicted quaternion
        k = Ts*R@omega
        
        k_norm_2 = la.norm(k,2)
        rhs_quat = np.zeros((4,))
        rhs_quat[0] = np.cos(k_norm_2/2)
        rhs_quat[1:] = np.sin(k_norm_2/2)*(k.T)/k_norm_2
        rhs_quat = rhs_quat.T
       
        
        # Normalize quaternion
        quaternion_prediction = quaternion_product(quaternion,rhs_quat)  # TODO: Normalize
        quaternion_prediction/=la.norm(quaternion_prediction,2)
        

        p_ab = self.p_acc
        p_wb = self.p_gyro

        acceleration_bias_prediction = np.zeros(
            (3,)
        )  # TODO: Calculate predicted acceleration bias
        acceleration_bias_prediction += np.exp(-p_ab*np.eye(3)*Ts)@acceleration_bias
        gyroscope_bias_prediction = np.zeros(
            (3,)
        )  # TODO: Calculate predicted gyroscope bias
        gyroscope_bias_prediction += np.exp(-p_wb*np.eye(3)*Ts)@gyroscope_bias
        x_nominal_predicted = np.concatenate(
            (
                position_prediction,
                velocity_prediction,
                quaternion_prediction,
                acceleration_bias_prediction,
                gyroscope_bias_prediction,
            )
        )

        assert x_nominal_predicted.shape == (
            16,
        ), f"ESKF.predict_nominal: x_nominal_predicted shape incorrect {x_nominal_predicted.shape}"

        return x_nominal_predicted

    def Aerr(
        self, x_nominal: np.ndarray, acceleration: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        """Calculate the continuous time error state dynamics Jacobian.

        Args:
            x_nominal (np.ndarray): The nominal state, shape (16,)
            acceleration (np.ndarray): The estimated acceleration for the prediction interval, shape (3,)
            omega (np.ndarray): The estimated rotation rate for the prediction interval, shape (3,)

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            np.ndarray: Continuous time error state dynamics Jacobian, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.Aerr: x_nominal shape incorrect {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.Aerr: acceleration shape incorrect {acceleration.shape}"
        assert omega.shape == (3,), f"ESKF.Aerr: omega shape incorrect {omega.shape}"

        # Rotation matrix
        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)

        # Allocate the matrix
        A = np.zeros((15, 15))

        # Set submatrices
        A[POS_IDX * VEL_IDX] = np.zeros((3,))
        A[VEL_IDX * ERR_ATT_IDX] = np.zeros((3,))
        A[VEL_IDX * ERR_ACC_BIAS_IDX] = np.zeros((3,))
        A[ERR_ATT_IDX * ERR_ATT_IDX] = np.zeros((3,))
        A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] = np.zeros((3,))
        A[ERR_ACC_BIAS_IDX * ERR_ACC_BIAS_IDX] = np.zeros((3,))
        A[ERR_GYRO_BIAS_IDX * ERR_GYRO_BIAS_IDX] = np.zeros((3,))

        A[VEL_IDX*POS_IDX] = np.eye(3)
        A[ATT_IDX[1:4]*VEL_IDX] = -R@(cross_product_matrix(acceleration-x_nominal[ERR_ACC_BIAS_IDX]))
        A[ATT_IDX[1:4]*ATT_IDX[1:4]] = -cross_product_matrix(omega-x_nominal[ERR_GYRO_BIAS_IDX])
        A[ERR_ATT_IDX*VEL_IDX] = -R
        A[ERR_ATT_IDX*ERR_ATT_IDX] = self.p_acc*np.eye(3)
        A[ERR_GYRO_BIAS_IDX*ATT_IDX[1:4]] = -np.eye(3)
        A[ERR_GYRO_BIAS_IDX*ERR_GYRO_BIAS_IDX] = self.p_gyro*np.eye(3)
        # Bias correction
        A[VEL_IDX * ERR_ACC_BIAS_IDX] = A[VEL_IDX * ERR_ACC_BIAS_IDX] @ self.S_a
        A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] = (
            A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] @ self.S_g
        )

        assert A.shape == (
            15,
            15,
        ), f"ESKF.Aerr: A-error matrix shape incorrect {A.shape}"
        return A
    
    def Gerr(self, x_nominal: np.ndarray,) -> np.ndarray:
        """Calculate the continuous time error state noise input matrix

        Args:
            x_nominal (np.ndarray): The nominal state, shape (16,)

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            np.ndarray: The continuous time error state noise input matrix, shape (15, 12)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.Gerr: x_nominal shape incorrect {x_nominal.shape}"

        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)

        G = np.zeros((15, 12))
        G[3:6,0:3] = -R
        G[6:9,3:6] = - np.eye(3)
        G[9:12,6:9] = np.eye(3)
        G[12:15,9:12] = -np.eye(3) 

        assert G.shape == (15, 12), f"ESKF.Gerr: G-matrix shape incorrect {G.shape}"
        return G

    def discrete_error_matrices(
        self,
        x_nominal: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the discrete time linearized error state transition and covariance matrix.

        Args:
            x_nominal (np.ndarray): The nominal state, shape (16,)
            acceleration (np.ndarray): The estimated acceleration in body for the prediction interval, shape (3,)
            omega (np.ndarray): The estimated rotation rate in body for the prediction interval, shape (3,)
            Ts (float): The sampling time

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            Tuple[np.ndarray, np.ndarray]: Discrete error matrices Tuple(Ad, GQGd)
                Ad: The discrete time error state system matrix, shape (15, 15)
                GQGd: The discrete time noise covariance matrix, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.discrete_error_matrices: x_nominal shape incorrect {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.discrete_error_matrices: acceleration shape incorrect {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.discrete_error_matrices: omega shape incorrect {omega.shape}"

        A = self.Aerr(x_nominal, acceleration, omega)
        G = self.Gerr(x_nominal)

        V = np.zeros((30, 30))
        A_shape = A.shape[0]
        V[0:A_shape,0:A_shape] = -A
        V[A_shape:,0:A_shape] = np.zeros((A_shape,A_shape))
        V[A_shape:,A_shape:] = A.T
        V[0:A_shape,A_shape:] = G@self.Q_err@G.T
        assert V.shape == (
            30,
            30,
        ), f"ESKF.discrete_error_matrices: Van Loan matrix shape incorrect {V.shape}"
          # This can be slow...
        VanLoanMatrix = la.expm(V)
        Ad = VanLoanMatrix[A_shape:,A_shape:].T
        GQGd = Ad@VanLoanMatrix[:A_shape,A_shape:]

        assert Ad.shape == (
            15,
            15,
        ), f"ESKF.discrete_error_matrices: Ad-matrix shape incorrect {Ad.shape}"
        assert GQGd.shape == (
            15,
            15,
        ), f"ESKF.discrete_error_matrices: GQGd-matrix shape incorrect {GQGd.shape}"

        return Ad, GQGd

    def predict_covariance(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
    ) -> np.ndarray:
        """Predict the error state covariance Ts time units ahead using linearized continuous time dynamics.

        Args:
            x_nominal (np.ndarray): The nominal state, shape (16,)
            P (np.ndarray): The error state covariance, shape (15, 15)
            acceleration (np.ndarray): The estimated acceleration for the prediction interval, shape (3,)
            omega (np.ndarray): The estimated rotation rate for the prediction interval, shape (3,)
            Ts (float): The sampling time

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            np.ndarray: The predicted error state covariance matrix, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict_covariance: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (
            15,
            15,
        ), f"ESKF.predict_covariance: P shape incorrect {P.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.predict_covariance: acceleration shape incorrect {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.predict_covariance: omega shape incorrect {omega.shape}"

        Ad, GQGd = self.discrete_error_matrices(x_nominal, acceleration, omega, Ts)

        

        P_predicted = Ad@P@Ad.T+GQGd

        assert P_predicted.shape == (
            15,
            15,
        ), f"ESKF.predict_covariance: P_predicted shape incorrect {P_predicted.shape}"
        return P_predicted

    def predict(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_acc: np.ndarray,
        z_gyro: np.ndarray,
        Ts: float,
    ) -> Tuple[np.array, np.array]:
        """Predict the nominal estimate and error state covariance Ts time units using IMU measurements z_*.

        Args:
            x_nominal (np.ndarray): The nominal state to predict, shape (16,)
            P (np.ndarray): The error state covariance to predict, shape (15, 15)
            z_acc (np.ndarray): The measured acceleration for the prediction interval, shape (3,)
            z_gyro (np.ndarray): The measured rotation rate for the prediction interval, shape (3,)
            Ts (float): The sampling time

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            Tuple[ np.array, np.array ]: Prediction Tuple(x_nominal_predicted, P_predicted)
                x_nominal_predicted: The predicted nominal state, shape (16,)
                P_predicted: The predicted error state covariance, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.predict: P shape incorrect {P.shape}"
        assert z_acc.shape == (3,), f"ESKF.predict: zAcc shape incorrect {z_acc.shape}"
        assert z_gyro.shape == (
            3,
        ), f"ESKF.predict: zGyro shape incorrect {z_gyro.shape}"

        # correct measurements
        r_z_acc = self.S_a @ z_acc
        r_z_gyro = self.S_g @ z_gyro

        # correct biases
        acc_bias = self.S_a @ x_nominal[ACC_BIAS_IDX]
        gyro_bias = self.S_g @ x_nominal[GYRO_BIAS_IDX]

        # debias IMU measurements
        acceleration = r_z_acc-acc_bias
        omega = r_z_gyro-gyro_bias

        # perform prediction
        x_nominal_predicted = self.predict_nominal(x_nominal,acceleration,omega,Ts)
        P_predicted = self.predict_covariance(x_nominal,P,acceleration,omega,Ts)

        assert x_nominal_predicted.shape == (
            16,
        ), f"ESKF.predict: x_nominal_predicted shape incorrect {x_nominal_predicted.shape}"
        assert P_predicted.shape == (
            15,
            15,
        ), f"ESKF.predict: P_predicted shape incorrect {P_predicted.shape}"

        return x_nominal_predicted, P_predicted

    def inject(
        self, x_nominal: np.ndarray, delta_x: np.ndarray, P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject a calculated error state into the nominal state and compensate in the covariance.

        Args:
            x_nominal (np.ndarray): The nominal state to inject the error state deviation into, shape (16,)
            delta_x (np.ndarray): The error state deviation, shape (15,)
            P (np.ndarray): The error state covariance matrix

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            Tuple[ np.ndarray, np.ndarray ]: Injected Tuple(x_injected, P_injected):
                x_injected: The injected nominal state, shape (16,)
                P_injected: The injected error state covariance matrix, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.inject: x_nominal shape incorrect {x_nominal.shape}"
        assert delta_x.shape == (
            15,
        ), f"ESKF.inject: delta_x shape incorrect {delta_x.shape}"
        assert P.shape == (15, 15), f"ESKF.inject: P shape incorrect {P.shape}"

        ### Useful concatenation of indices
        # All injection indices, minus the attitude
        INJ_IDX = POS_IDX + VEL_IDX + ACC_BIAS_IDX + GYRO_BIAS_IDX
        # All error indices, minus the attitude
        DTX_IDX = POS_IDX + VEL_IDX + ERR_ACC_BIAS_IDX + ERR_GYRO_BIAS_IDX

        x_injected = x_nominal.copy()
        x_injected[INJ_IDX] += delta_x[DTX_IDX]
        quat_injected = quaternion_product(x_nominal[ATT_IDX], np.block([1,(1/2)*delta_x[ATT_IDX[1:4]]]))
        x_injected[ATT_IDX] = quat_injected/(la.norm(quat_injected,2))
        # Covariance
        G_injected = np.zeros((15,15))# TODO: Compensate for injection in the covariances
        G_injected[0:6,0:6] = np.eye(6)
        G_injected[6:9,6:9] = np.eye(3)-cross_product_matrix((1/2)*delta_x[ATT_IDX[1:4]])
        G_injected[9:15,9:15] = np.eye(6)

        P_injected = np.zeros(
            (15, 15)
        )  # TODO: Compensate for injection in the covariances
        P_injected = P@G_injected@P.T

        assert x_injected.shape == (
            16,
        ), f"ESKF.inject: x_injected shape incorrect {x_injected.shape}"
        assert P_injected.shape == (
            15,
            15,
        ), f"ESKF.inject: P_injected shape incorrect {P_injected.shape}"

        return x_injected, P_injected

    def innovation_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates the innovation and its covariance for a GNSS position measurement

        Args:
            x_nominal (np.ndarray): The nominal state to calculate the innovation from, shape (16,)
            P (np.ndarray): The error state covariance to calculate the innovation covariance from, shape (15, 15)
            z_GNSS_position (np.ndarray): The measured 3D position, shape (3,)
            R_GNSS (np.ndarray): The estimated covariance matrix of the measurement, shape (3, 3)
            lever_arm (np.ndarray, optional): The position of the GNSS receiver from the IMU reference. Defaults to np.zeros(3).

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            Tuple[ np.ndarray, np.ndarray ]: Innovation Tuple(v, S):
                v: innovation, shape (3,)
                S: innovation covariance, shape (3, 3)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.innovation_GNSS: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.innovation_GNSS: P shape incorrect {P.shape}"

        assert z_GNSS_position.shape == (
            3,
        ), f"ESKF.innovation_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
        assert R_GNSS.shape == (
            3,
            3,
        ), f"ESKF.innovation_GNSS: R_GNSS shape incorrect {R_GNSS.shape}"
        assert lever_arm.shape == (
            3,
        ), f"ESKF.innovation_GNSS: lever_arm shape incorrect {lever_arm.shape}"

        H = np.block([np.eye(3),np.zeros((3,13))])
      
         # TODO: measurement matrix
        
        v = z_GNSS_position - x_nominal[POS_IDX] # TODO: innovation

        # leverarm compensation
        if not np.allclose(lever_arm, 0):
            R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)
            H[:, ERR_ATT_IDX] = -R @ cross_product_matrix(lever_arm, debug=self.debug)
            v -= R @ lever_arm
        
        #H = np.block([np.eye(3), np.zeros((3,13))])@X_delta_x # TODO: measurement matrix
        H = np.eye(3,15)
        S = H@P@H.T + R_GNSS  # TODO: innovation covariance

        assert v.shape == (3,), f"ESKF.innovation_GNSS: v shape incorrect {v.shape}"
        assert S.shape == (3, 3), f"ESKF.innovation_GNSS: S shape incorrect {S.shape}"
        return v, S

    def update_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates the state and covariance from a GNSS position measurement

        Args:
            x_nominal (np.ndarray): The nominal state to update, shape (16,)
            P (np.ndarray): The error state covariance to update, shape (15, 15)
            z_GNSS_position (np.ndarray): The measured 3D position, shape (3,)
            R_GNSS (np.ndarray): The estimated covariance matrix of the measurement, shape (3, 3)
            lever_arm (np.ndarray, optional): The position of the GNSS receiver from the IMU reference, shape (3,). Defaults to np.zeros(3), shape (3,).

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated Tuple(x_injected, P_injected):
                x_injected: The nominal state after injection of updated error state, shape (16,)
                P_injected: The error state covariance after error state update and injection, shape (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.update_GNSS: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.update_GNSS: P shape incorrect {P.shape}"
        assert z_GNSS_position.shape == (
            3,
        ), f"ESKF.update_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
        assert R_GNSS.shape == (
            3,
            3,
        ), f"ESKF.update_GNSS: R_GNSS shape incorrect {R_GNSS.shape}"
        assert lever_arm.shape == (
            3,
        ), f"ESKF.update_GNSS: lever_arm shape incorrect {lever_arm.shape}"

        I = np.eye(*P.shape)

        innovation, S = self.innovation_GNSS_position(
            x_nominal, P, z_GNSS_position, R_GNSS, lever_arm
        )
        
        H = np.eye(3,15)# TODO: measurement matrix
        # in case of a specified lever arm
        if not np.allclose(lever_arm, 0):
            R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)
            H[:, ERR_ATT_IDX] = -R @ cross_product_matrix(lever_arm, debug=self.debug)


        # KF error state update

        W = P @ la.solve(S,H).T  # TODO: Kalman gain
        delta_x = W@innovation  # TODO: delta x

        Jo = I - W @ H  # for Joseph form

        P_update = Jo@P@Jo.T+W@R_GNSS@W.T  # TODO: P update

        # error state injection
        x_injected, P_injected = self.inject(x_nominal, delta_x, P_update)

        assert x_injected.shape == (
            16,
        ), f"ESKF.update_GNSS: x_injected shape incorrect {x_injected.shape}"
        assert P_injected.shape == (
            15,
            15,
        ), f"ESKF.update_GNSS: P_injected shape incorrect {P_injected.shape}"

        return x_injected, P_injected

    def NIS_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> float:
        """Calculates the NIS for a GNSS position measurement

        Args:
            x_nominal (np.ndarray): The nominal state to calculate the innovation from, shape (16,)
            P (np.ndarray): The error state covariance to calculate the innovation covariance from, shape (15, 15)
            z_GNSS_position (np.ndarray): The measured 3D position, shape (3,)
            R_GNSS (np.ndarray): The estimated covariance matrix of the measurement, shape (3,)
            lever_arm (np.ndarray, optional): The position of the GNSS receiver from the IMU reference, shape (3,). Defaults to np.zeros(3).

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            float: The normalized innovations squared (NIS)
        """

        assert x_nominal.shape == (
            16,
        ), "ESKF.NIS_GNSS: x_nominal shape incorrect " + str(x_nominal.shape)
        assert P.shape == (15, 15), "ESKF.NIS_GNSS: P shape incorrect " + str(P.shape)
        assert z_GNSS_position.shape == (
            3,
        ), "ESKF.NIS_GNSS: z_GNSS_position shape incorrect " + str(
            z_GNSS_position.shape
        )
        assert R_GNSS.shape == (3, 3), "ESKF.NIS_GNSS: R_GNSS shape incorrect " + str(
            R_GNSS.shape
        )
        assert lever_arm.shape == (
            3,
        ), "ESKF.NIS_GNSS: lever_arm shape incorrect " + str(lever_arm.shape)

        v, S = self.innovation_GNSS_position(
            x_nominal, P, z_GNSS_position, R_GNSS, lever_arm
        )

        NIS = v.T @ np.linalg.inv(S) @ v  # TODO: Calculate NIS

        assert NIS >= 0, "EKSF.NIS_GNSS_positionNIS: NIS not positive"

        return NIS

    @classmethod
    def delta_x(cls, x_nominal: np.ndarray, x_true: np.ndarray,) -> np.ndarray:
        """Calculates the error state between x_nominal and x_true

        Args:
            x_nominal (np.ndarray): The nominal estimated state, shape (16,)
            x_true (np.ndarray): The true state, shape (16,)

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            np.ndarray: The state difference in error state, shape (15,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.delta_x: x_nominal shape incorrect {x_nominal.shape}"
        assert x_true.shape == (
            16,
        ), f"ESKF.delta_x: x_true shape incorrect {x_true.shape}"

        delta_position = x_nominal[POS_IDX] - x_true[POS_IDX]  # TODO: Delta position
        delta_velocity = x_nominal[VEL_IDX] - x_true[VEL_IDX]  # TODO: Delta velocity

        quaternion_conj = -x_nominal[ATT_IDX]  # TODO: Conjugate of quaternion

        delta_quaternion = quaternion_product(quaternion_conj, x_true[ATT_IDX])  # TODO: Error quaternion
        delta_theta = 2*np.imag(delta_quaternion[1:4])
        

        # Concatenation of bias indices
        BIAS_IDX = ACC_BIAS_IDX + GYRO_BIAS_IDX
        delta_bias = x_nominal[BIAS_IDX] - x_true[BIAS_IDX]  # TODO: Error biases

        d_x = np.concatenate((delta_position, delta_velocity, delta_theta, delta_bias))

        assert d_x.shape == (15,), f"ESKF.delta_x: d_x shape incorrect {d_x.shape}"

        return d_x

    @classmethod
    def NEESes(
        cls, x_nominal: np.ndarray, P: np.ndarray, x_true: np.ndarray,
    ) -> np.ndarray:
        """Calculates the total NEES and the NEES for the substates

        Args:
            x_nominal (np.ndarray): The nominal estimate
            P (np.ndarray): The error state covariance
            x_true (np.ndarray): The true state

        Raises:
            AssertionError: If any input is of the wrong shape, and if debug mode is on, certain numeric properties

        Returns:
            np.ndarray: NEES for [all, position, velocity, attitude, acceleration_bias, gyroscope_bias], shape (6,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.NEES: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.NEES: P shape incorrect {P.shape}"
        assert x_true.shape == (
            16,
        ), f"ESKF.NEES: x_true shape incorrect {x_true.shape}"

        d_x = cls.delta_x(x_nominal, x_true)
        
        NEES_all = d_x.T @ np.linalg.inv(P) @ d_x  # TODO: NEES all
        NEES_pos = d_x[POS_IDX].T @ np.linalg.inv(P[POS_IDX*POS_IDX]) @ d_x[POS_IDX]  # TODO: NEES position
        NEES_vel = d_x[VEL_IDX].T @ np.linalg.inv(P[VEL_IDX*VEL_IDX]) @ d_x[VEL_IDX]  # TODO: NEES velocity
        NEES_att = d_x[ERR_ATT_IDX].T @ np.linalg.inv(P[ERR_ATT_IDX*ERR_ATT_IDX]) @ d_x[ERR_ATT_IDX]  # TODO: NEES attitude
        print(P.shape)
        NEES_accbias = d_x[ERR_ACC_BIAS_IDX].T @ np.linalg.inv(P[ERR_ACC_BIAS_IDX*ERR_ACC_BIAS_IDX]) @ d_x[ERR_ACC_BIAS_IDX]   # TODO: NEES accelerometer bias
        NEES_gyrobias = d_x[ERR_GYRO_BIAS_IDX].T @ np.linalg.inv(P[ERR_GYRO_BIAS_IDX*ERR_GYRO_BIAS_IDX]) @ d_x[ERR_GYRO_BIAS_IDX]  # TODO: NEES gyroscope bias
        NEESes = np.array(
            [NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias, NEES_gyrobias]
        )
        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        return NEESes

    @classmethod
    def _NEES(cls, diff, P):
        NEES = diff.T @ P @ diff  # TODO: NEES
        assert NEES >= 0, "ESKF._NEES: negative NEES"
        return NEES


# %%
