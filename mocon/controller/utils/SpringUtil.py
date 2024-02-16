from scipy.spatial.transform import Rotation as R
import math

def from_euler(e):
    return R.from_euler('XYZ', e, degrees=True)

class SpringUtil:
    @staticmethod
    def lerp(a, b, t):
        return a + (b - a) * t
    
    @staticmethod
    def halflife2dampling(halflife):
        return 4 * math.log(2) / halflife

    @staticmethod
    def update_spring_rot(rot, avel, target_rot, halflife, dt):
        # to target rotation (R_target * R^-1)
        j0 = R.from_quat(rot) * R.from_quat(target_rot).inv()
        j0 = j0.as_rotvec()

        d = SpringUtil.halflife2dampling(halflife) / 2
        j1 = avel + d * j0
        
        eydt = math.exp(-d * dt) # exponential decay factor
        tmp1 = eydt * (j0 + j1 * dt)

        rot = R.from_rotvec(tmp1) * R.from_quat(target_rot)
        rot = rot.as_quat()
        avel = eydt * (avel - j1 * dt * d)
        return rot, avel

    def update_spring_pos(pos, vel, acc, target_vel, halflife, dt):
        d = SpringUtil.halflife2dampling(halflife) / 2
        j0 = vel - target_vel
        j1 = acc + d * j0

        eydt = math.exp(-d * dt)
        pos_prev = pos

        tmp1 = j0 + j1 * dt
        tmp2 = j1 / (d * d)
        pos = eydt * ( -tmp2 - tmp1/d ) + tmp2 + j0/d + target_vel*dt + pos_prev
        vel = eydt*tmp1 + target_vel
        acc = eydt * (acc - j1*d*dt)
        return pos, vel, acc
    
    @staticmethod
    def decay_spring_implicit_damping_rot(rot, avel, halflife, dt):
        d = SpringUtil.halflife2dampling(halflife) / 2
        j0 = from_euler(rot).as_rotvec()
        j1 = avel + d * j0
        eydt = math.exp(-d * dt)
        a1 = eydt * (j0 + j1*dt)
       
        rot_res = R.from_rotvec(a1).as_euler('XYZ', degrees=True)
        avel_res = eydt * (avel - j1 * dt * d)
        return rot_res, avel_res
    
    @staticmethod
    def decay_spring_implicit_damping_pos(pos, vel, halflife, dt):
        d = SpringUtil.halflife2dampling(halflife) / 2
        j1 = vel + d * pos
        eydt = math.exp(-d * dt)
        pos = eydt * (pos + j1*dt)
        vel = eydt * (vel - j1 * dt * d)
        return pos, vel
    
    @staticmethod
    def inertialize_transition_rot(prev_off_rot, prev_off_avel, src_rot, src_avel, dst_rot, dst_avel):
        prev_off_rot, prev_off_avel = SpringUtil.decay_spring_implicit_damping_rot(prev_off_rot, prev_off_avel, 1/20, 1/60)
        off_rot = from_euler(prev_off_rot) * from_euler(src_rot) * from_euler(dst_rot).inv()
        off_avel = prev_off_avel + src_avel - dst_avel
        # off_rot = from_euler(src_rot) * from_euler(dst_rot).inv()
        # off_avel = src_avel - dst_avel
        return off_rot.as_euler('XYZ', degrees=True), off_avel
    
    @staticmethod
    def inertialize_update_rot(prev_off_rot, prev_off_avel, rot, avel, halflife, dt):
        off_rot , off_avel = SpringUtil.decay_spring_implicit_damping_rot(prev_off_rot, prev_off_avel, halflife, dt)
        rot = from_euler(off_rot) * from_euler(rot)
        avel = off_avel + avel
        return rot.as_euler('XYZ', degrees=True), avel, off_rot, off_avel
    
    @staticmethod
    def inertialize_transition_pos(prev_off_pos, prev_off_vel, src_pos, src_vel, dst_pos, dst_vel):
        prev_off_pos, prev_off_vel = SpringUtil.decay_spring_implicit_damping_pos(prev_off_pos, prev_off_vel, 1/20, 1/60)
        off_pos = prev_off_pos + src_pos - dst_pos
        off_vel = prev_off_vel + src_vel - dst_vel
        return off_pos, off_vel
    
    @staticmethod
    def inertialize_update_pos(prev_off_pos, prev_off_vel, pos, vel, halflife, dt):
        off_pos , off_vel = SpringUtil.decay_spring_implicit_damping_pos(prev_off_pos, prev_off_vel, halflife, dt)
        pos = off_pos + pos
        vel = off_vel + vel
        return pos, vel, off_pos, off_vel
    