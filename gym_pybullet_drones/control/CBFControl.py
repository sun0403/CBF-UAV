import math
import numpy as np
import pybullet as p
import nlopt
from scipy.spatial.transform import Rotation
from transforms3d.quaternions import rotate_vector, qconjugate, mat2quat, qmult
from transforms3d.utils import normalized_vector

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel
from GaussianProcess import GaussianProcessCBF
import torch

class NloptControl(BaseControl):
    
    

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8,
                 obstacle_positions=None,
                 obstacle_radius = 0.2
                 ):
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.CF2X and self.DRONE_MODEL != DroneModel.CF2P:
            print("[ERROR] in DSLPIDControl.__init__(), DSLPIDControl requires DroneModel.CF2X or DroneModel.CF2P")
            exit()
        

        self.P_COEFF_FOR = np.array([.4, .4, 1.25])
        self.I_COEFF_FOR = np.array([.05, .05, .05])
        self.D_COEFF_FOR = np.array([.2, .2, .5])
        self.P_COEFF_TOR = np.array([70000., 70000., 60000.])
        self.I_COEFF_TOR = np.array([.0, .0, 500.])
        self.D_COEFF_TOR = np.array([20000., 20000., 12000.])
        self.PWM2RPM_SCALE = 0.2685
        self.PWM2RPM_CONST = 4070.3
        self.MIN_PWM = 20000
        self.MAX_PWM = 65535
        self.obstacle_positions = obstacle_positions
        self.gp_model = GaussianProcessCBF(input_dim=6, debug=True)

        self.obstacle_radius = 0.1
        self.theta = 1.0
        


        self.safe_distance = 0.05
        self.lambda1 =10.0
        self.lambda2 = 10.0
        self._initialize_gp_samples()
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MIXER_MATRIX = np.array([ 
                                    [-.5, -.5, -1],
                                    [-.5,  .5,  1],
                                    [.5, .5, -1],
                                    [.5, -.5,  1]
                                    ])
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MIXER_MATRIX = np.array([
                                    [0, -1,  -1],
                                    [+1, 0, 1],
                                    [0,  1,  -1],
                                    [-1, 0, 1]
                                    ])
        self.reset()

    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Store the last roll, pitch, and yaw ###################
        self.last_rpy = np.zeros(3)
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

    
    
    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        self.control_counter += 1
        thrust, computed_target_rpy, pos_e = self._dslPIDPositionControl(control_timestep,
                                                                         cur_pos,
                                                                         cur_quat,
                                                                         cur_vel,
                                                                         target_pos,
                                                                         target_rpy,
                                                                         target_vel
                                                                         )
        rpm = self._dslPIDAttitudeControl(control_timestep,
                                          thrust,
                                          cur_quat,
                                          cur_ang_vel,
                                          computed_target_rpy,
                                          target_rpy_rates
                                          )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]
    
    ################################################################################

    def _dslPIDPositionControl(self,
                               control_timestep,
                               cur_pos,
                               cur_quat,
                               cur_vel,
                               target_pos,
                               target_rpy,
                               target_vel
                               ):
        
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        pos_e = target_pos - cur_pos
        vel_e = target_vel - cur_vel
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        self.integral_pos_e = np.clip(self.integral_pos_e, -2., 2.)
        self.integral_pos_e[2] = np.clip(self.integral_pos_e[2], -0.15, .15)


        #### PID target thrust #####################################
        target_thrust = np.multiply(self.P_COEFF_FOR, pos_e) \
                        + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                        + np.multiply(self.D_COEFF_FOR, vel_e) + np.array([0, 0, self.GRAVITY])
        
    
        u_opt = self._optimize_control(target_thrust, cur_pos, cur_vel, cur_quat, target_pos)

        
        
        scalar_thrust = max(0., np.dot(u_opt, cur_rotation[:,2]))
        thrust = (math.sqrt(scalar_thrust / (4*self.KF)) - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
    
        target_z_ax = u_opt / np.linalg.norm(u_opt)
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()
        #### Target rotation #######################################
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False)
        if np.any(np.abs(target_euler) > math.pi):
            print("\n[ERROR] ctrl it", self.control_counter, "in Control._dslPIDPositionControl(), values outside range [-pi,pi]")
        return thrust, target_euler, pos_e
    
    ################################################################################

    def _dslPIDAttitudeControl(self,
                               control_timestep,
                               thrust,
                               cur_quat,
                               cur_ang_vel,
                               target_euler,
                               target_rpy_rates
                               ):
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        cur_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        print(f"目标欧拉{target_euler},当前角速度{cur_ang_vel},当前欧拉{cur_rpy}")
        target_euler_cbf = self._optimize_control_attitude(target_euler, cur_ang_vel, cur_rpy)
        target_quat = (Rotation.from_euler('XYZ', target_euler_cbf, degrees=False)).as_quat()
        w,x,y,z = target_quat
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()),cur_rotation) - np.dot(cur_rotation.transpose(),target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) 
        rpy_rates_e = target_rpy_rates - (cur_rpy - self.last_rpy)/control_timestep
        self.last_rpy = cur_rpy
        self.integral_rpy_e = self.integral_rpy_e - rot_e*control_timestep
        self.integral_rpy_e = np.clip(self.integral_rpy_e, -1500., 1500.)
        self.integral_rpy_e[0:2] = np.clip(self.integral_rpy_e[0:2], -1., 1.)
        #### PID target torques ####################################
        target_torques = - np.multiply(self.P_COEFF_TOR, rot_e) \
                         + np.multiply(self.D_COEFF_TOR, rpy_rates_e) \
                         + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e)
        target_torques = np.clip(target_torques, -3200, 3200)
        pwm = thrust + np.dot(self.MIXER_MATRIX, target_torques)
        pwm = np.clip(pwm, self.MIN_PWM, self.MAX_PWM)
        return self.PWM2RPM_SCALE * pwm + self.PWM2RPM_CONST
    
    
    def _one23DInterface(self,
                         thrust
                         ):
        """Utility function interfacing 1, 2, or 3D thrust input use cases.

        Parameters
        ----------
        thrust : ndarray
            Array of floats of length 1, 2, or 4 containing a desired thrust input.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.

        """
        DIM = len(np.array(thrust))
        pwm = np.clip((np.sqrt(np.array(thrust)/(self.KF*(4/DIM)))-self.PWM2RPM_CONST)/self.PWM2RPM_SCALE, self.MIN_PWM, self.MAX_PWM)
        if DIM in [1, 4]:
            return np.repeat(pwm, 4/DIM)
        elif DIM==2:
            return np.hstack([pwm, np.flip(pwm)])
        else:
            print("[ERROR] in DSLPIDControl._one23DInterface()")
            exit()
    
    def _optimize_control(self, u_nominal, cur_pos, cur_vel, cur_quat, target_pos):
        opt = nlopt.opt(nlopt.LD_SLSQP, 3)  
        opt.set_min_objective(lambda u, grad: self._objective(u, u_nominal, grad))
        opt.set_lower_bounds([-50.0, -50.0, -50.0])  
        opt.set_upper_bounds([50.0, 50, 50.0])
        opt.add_inequality_constraint(lambda u, grad: self._cbf_constraint(u, cur_pos, cur_vel, cur_quat, grad), 1e-6)
        opt.set_maxeval(50)
        opt.set_xtol_rel(1e-6)


        u_opt = opt.optimize(u_nominal)  
        self.gp_model.update(
            cur_pos, cur_vel, u_opt, target_pos,
            obstacle_positions=self.obstacle_positions,
            GRAVITY=self.GRAVITY,
            safe_distance=self.safe_distance,
            lambda1=self.lambda1,
            lambda2=self.lambda2
        )



        return u_opt
    def _objective(self, u, u_nominal, grad):
        
       
        if grad.size > 0:
            grad[:] = 2 * (u - u_nominal)  
        return np.linalg.norm(u - u_nominal) ** 2

    def _initialize_gp_samples(self, path_points=None, num_samples=1, vel_range=0.5):
        
        print("[INFO] Initializing GP dataset...")
        for i in range(num_samples):
            # 若给定路径点，则在路径上采样，否则用默认中心
            if path_points is not None:
                pos_center = path_points[i % len(path_points)]
            else:
                pos_center = np.array([0.0, 0.0, 0.5])

            pos_noise = np.random.normal(0, 0.05, size=3)
            vel_sample = np.random.uniform(-vel_range, vel_range, size=3)
            x_sample = np.concatenate([pos_center + pos_noise, vel_sample])

            
            if self.obstacle_positions and len(self.obstacle_positions) > 0:
                h = np.min([np.linalg.norm(x_sample[:3] - obs) for obs in self.obstacle_positions]) - self.safe_distance
            

            self.gp_model.add_sample(x_sample, h)

        print(f"[INFO] GP initialization complete. Dataset size: {self.gp_model.X_train.size(0)}")


    def _cbf_constraint(self, u, cur_pos, cur_vel, cur_quat, grad):
        
        cbf_values = []
        

        for i, obstacle_pos in enumerate(self.obstacle_positions):

            obstacle_distance = np.linalg.norm(cur_pos - obstacle_pos)  
        
        

        
            gp_input = np.concatenate([cur_pos, cur_vel])
            
            sigma2 = self.gp_model.predict_variance(gp_input)
            
            sigma2_grad, sigma2_hessian = self.gp_model.compute_variance_gradient_and_hessian(gp_input)
            sigma2_grad = sigma2_grad.flatten()
            grad_pos = sigma2_grad[:3]
            grad_vel = sigma2_grad[3:]
            
            h_value = obstacle_distance - self.safe_distance - self.theta*sigma2
            
            

            
            direction_vector = (cur_pos-obstacle_pos) / (obstacle_distance + 1e-6)

            
            h_dot = np.dot(direction_vector, cur_vel)  
            sigma2_dot = np.dot(grad_pos, cur_vel)
            
            h_dot_s = h_dot - self.theta*sigma2_dot
            

            acceleration = (u / self.GRAVITY) - np.array([0, 0, 9.8]) 

            h_ddot = np.dot(direction_vector, acceleration)
            sigma2_ddot = np.dot(grad_vel, acceleration) + np.dot(acceleration, sigma2_hessian[3:,3:] @ acceleration)
    

    
            h_ddot_s = h_ddot - self.theta*sigma2_ddot
            cbf_value = -(h_ddot_s + self.lambda1 * h_dot_s + self.lambda2 * h_value)


            
        
        if grad.size > 0:
            grad[:] = -direction_vector / self.GRAVITY  
        return cbf_value
    
    def _euler_cbf_constraint_single_deg(self, u_deg, ang_vel_deg, rpy_deg, grad, idx):
        max_angle = 15.0  # 最大角度，单位度
        lambda_att = 10000.0

        phi = u_deg[idx]  # u_deg 是优化变量
        phi_dot = ang_vel_deg[idx]  
        h = max_angle**2 - phi**2
        h_dot = -2 * phi * phi_dot
        cbf = -(h_dot + lambda_att * h)

        if grad.size > 0:
            
            grad[:] = np.zeros_like(u_deg)  # 确保其他维度是0
            grad[idx] = 2.0 * phi_dot + lambda_att * 2.0 * phi

        return cbf
    def _optimize_control_attitude(self, target_euler, cur_ang_vel, cur_rpy):
        target_deg = np.degrees(target_euler)
        cur_rpy_deg = np.degrees(cur_rpy)
        cur_ang_vel_deg = np.degrees(cur_ang_vel)
        


        def cbf_wrapper(u_deg, grad):
            u_rad = np.radians(u_deg)
            return self._objective(u_rad, target_euler, grad)
        
        opt = nlopt.opt(nlopt.LD_SLSQP, 3)  
        opt.set_min_objective(cbf_wrapper)
        opt.set_lower_bounds([-180.0, -180.0, -180.0])
        opt.set_upper_bounds([180.0, 180.0, 180.0])
        for i in range(3):
            opt.add_inequality_constraint(lambda u_deg, grad, i=i: self._euler_cbf_constraint_single_deg(u_deg, cur_ang_vel_deg, cur_rpy_deg, grad, i), 1e-6)
        opt.set_xtol_rel(1e-6)
        opt.set_maxeval(50)


        
        try:
            u_opt_deg = opt.optimize(np.degrees(target_euler))
        except Exception as e:
            print(f"[WARNING] NLOPT exception: {e}")
            print(f"[⚠️ DEBUG] Final CBF values before failure (deg): cur_rpy={cur_rpy_deg}, ang_vel={cur_ang_vel_deg}, target={target_deg}")
            u_opt_deg = target_deg
        return np.radians(u_opt_deg)
    
    def _objective(self, u, u_nominal, grad):
        weights = np.array([1.0, 1.0, 1.0])  

        diff = u - u_nominal
        obj = np.sum(weights * diff**2)

        if grad.size > 0:
            grad[:] = 2.0 * weights * diff

        return obj
    


        

   

     