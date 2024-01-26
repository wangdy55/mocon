import numpy as np
from scipy.spatial.transform import Rotation as R

class QuatHelper:
    @staticmethod
    def alignQuat(qt: np.ndarray, inplace: bool):
        ''' 
        Make q_n and q_n+1 in the same semisphere
        The first dim. of qt should be time
        '''
        qt = np.asarray(qt)
        if qt.shape[-1] != 4:
            raise ValueError('qt has to be an array of quats')

        if not inplace:
            qt = qt.copy()

        if qt.size == 4:  # do nothing since there is only one quation
            return qt

        sign = np.sum(qt[:-1] * qt[1:], axis=-1)
        sign[sign < 0] = -1
        sign[sign >= 0] = 1
        sign = np.cumprod(sign, axis=0, )

        qt[1:][sign < 0] *= -1
        return qt

    @staticmethod
    def quatToAvel(rot, dt):
        '''
        Calculate angular velocity from quaternion using finite difference
        The first dim. of rot should be time
        '''
        rot = QuatHelper.alignQuat(rot, inplace=False) # hemisphere alignment
        quat_diff = (rot[1:] - rot[:-1]) / dt # finite difference
        # Fix real part w of quat: quat's norm should be 1
        quat_diff[...,-1] = (1 - np.sum(quat_diff[...,:-1]**2, axis=-1)).clip(min = 0)**0.5
        quat_tmp = rot[:-1].copy()
        quat_tmp[...,:3] *= -1
        shape = quat_diff.shape[:-1]
        rot_tmp = R.from_quat(quat_tmp.reshape(-1, 4)) * R.from_quat(quat_diff.reshape(-1, 4))
        return 2 * rot_tmp.as_quat().reshape( shape + (4, ) )[...,:3]

    @staticmethod
    def quatMul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Multiply 2 quat. (p.shape == q.shape)
        """
        assert p.shape == q.shape, "p and q should have the same shape"
        assert p.shape[-1] == 4, "p and q should have the last dimension of 4"

        xyz: np.ndarray = p[..., None, 3] * q[..., :3] + q[..., None, 3] * p[..., :3] + np.cross(p[..., :3], q[..., :3], axis=-1)
        w: np.ndarray = p[..., 3:4] * q[..., 3:4] - np.sum(p[..., :3] * q[..., :3], axis=-1, keepdims=True)

        return np.concatenate([xyz, w], axis=-1)

    @staticmethod
    def quatIntegrate(q: np.ndarray, omega: np.ndarray, dt: float):
        """
        Integrate quat. with angular vel.: q_{t+1} = normalize(q_{t} + 0.5 * w * q_{t})
        """
        assert q.shape[-1] == 4, "q should have the last dimension of 4"
        assert 3 == omega.shape[-1], "omega should have the last dimension of 3"

        initQShape = q.shape
        if q.shape[-1] == 1 and omega.shape[-1] == 1:
            q = q.reshape(q.shape[:-1])
            omega = omega.reshape(omega.shape[:-1])

        omega = np.concatenate([omega, np.zeros(omega.shape[:-1] + (1,))], axis=-1)

        dq = 0.5 * dt * QuatHelper.quatMul(omega, q)
        res = q + dq
        res /= np.linalg.norm(res, axis=-1, keepdims=True)
        return res.reshape(initQShape)
    
    @staticmethod
    def getQuatBetween(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Get quat. from vector a to vector b

        @Parameters:
            a: (n, 3) vector
            b: (n, 3) vector

        @Returns:
            q: (n, 4) quaternion
        """
        try:
            cross_res = np.cross(a, b, axis=-1)
        except:
            pass

        w = np.sqrt((a ** 2).sum(axis=-1) * (b ** 2).sum(axis=-1)) + (a * b).sum(axis=-1)
        q = np.concatenate([cross_res, w[..., np.newaxis]], axis=-1)
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    @staticmethod
    def decomposeQuat(q: np.ndarray, vb: np.ndarray):
        if vb.size < q.size:
            vb = np.ascontiguousarray(np.broadcast_to(vb, q.shape[:-1] + (3,)), dtype=np.float64)
        q = np.ascontiguousarray(q, np.float64)
        va = R.from_quat(q).apply(vb)
        va /= np.linalg.norm(va, axis=-1, keepdims=True)
        tmp = QuatHelper.getQuatBetween(va, vb)
        res = (R.from_quat(tmp) * R.from_quat(q)).as_quat()
        return res / np.linalg.norm(res, axis=-1, keepdims=True)

    @staticmethod
    def flipVectorByDot(x: np.ndarray, inplace: bool = False) -> np.ndarray:
        """
        make sure x[i] * x[i+1] >= 0
        """
        if x.ndim == 1:
            return x

        sign: np.ndarray = np.sum(x[:-1] * x[1:], axis=-1)
        sign[sign < 0] = -1
        sign[sign >= 0] = 1
        sign = np.cumprod(sign, axis=0, )

        x_res = x.copy() if not inplace else x
        x_res[1:][sign < 0] *= -1

        return x_res

    @staticmethod
    def flipQuatByDot(q: np.ndarray, inplace: bool = False) -> np.ndarray:
        if q.shape[-1] != 4:
            raise ValueError

        return QuatHelper.flipVectorByDot(q, inplace)

    @staticmethod
    def axisDecompose(q: np.ndarray, axis: np.ndarray):
        """
        Decompose a quat. into two quats along an axis
        
        @Returns:
            qa: quat. along axis
            qb: quat. perpendicular to axis
        """
        assert axis.ndim == 1 and axis.shape[0] == 3

        qa = QuatHelper.decomposeQuat(q, np.asarray(axis))
        qb = (R(qa, copy=False, normalize=False).inv() * R(q, copy=False, normalize=False)).as_quat()
        qb = QuatHelper.flipQuatByDot(qb)
        qa[np.abs(qa) < 1e-14] = 0
        qb[np.abs(qb) < 1e-14] = 0
        qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
        qb /= np.linalg.norm(qb, axis=-1, keepdims=True)
        return qa, qb

    @staticmethod
    def yDecompose(q: np.ndarray) -> tuple:
        return QuatHelper.axisDecompose(
            q, np.array([0.0, 1.0, 0.0])
        )

    @staticmethod
    def yExtractRad(q):
        '''
        Extract y-axis of quat. and return its angle in rad

        @Parameters:
            q: R.quat, shape = (N, 4)

        @Returns:
            y_rad: ndarray, shape = (N, 1)
        '''
        qy, _ = QuatHelper.yDecompose(q)
        return R(qy).as_rotvec()[:, 1]
