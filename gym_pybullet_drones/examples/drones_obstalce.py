import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl  
from gym_pybullet_drones.utils.Logger import Logger  
from gym_pybullet_drones.control.CBFControl import NloptControl
from gym_pybullet_drones.control.CTBRControl import CTBRControl
import os

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.animation as animation
os.environ["PYBULLET_EGL"] = "1"
import sys


class Log(object):
    def __init__(self, filename="debug_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Log()

SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 10
PLOT_RESULTS = True  
DEFAULT_NUM_DRONES = 2


def generate_target_path(start_pos, end_pos, steps=200):
    
    return np.linspace(start_pos, end_pos, steps)
def add_moving_obstacles(client_id, num_obstacles=30, x_range=(0.5, 1.0), y_range=(0.5, 1.0), z_range=(0.7, 1.5), velocity_range=(-0.3, 0.3)):
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    obstacle_ids = []
    obstacle_positions = []
    obstacle_velocities = []

    for _ in range(num_obstacles):
       
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)

        obstacle_position = np.array([x, y, z])

        
        velocity = np.random.uniform(*velocity_range, size=3)

        print(f"[INFO] Adding moving obstacle at {obstacle_position} with velocity {velocity}")

        # **加载 URDF**
        obstacle_id = p.loadURDF("sphere_small.urdf", obstacle_position, useFixedBase=False, physicsClientId=client_id)

        # **存储数据**
        obstacle_ids.append(obstacle_id)
        obstacle_positions.append(obstacle_position)
        obstacle_velocities.append(velocity)

    return obstacle_ids, obstacle_positions, obstacle_velocities, x_range, y_range, z_range

def add_obstacles_on_path(client_id, num_obstacles=20,x_range=(0.5, 1.0), y_range=(0.5, 1.0), z_range=(0.7, 1.5)):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    obstacle_ids = []
    obstacle_positions = []

    for _ in range(num_obstacles):
        
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])

        obstacle_position = np.array([x, y, z])
        
        print(f"[INFO] Adding obstacle at {obstacle_position}")

        obstacle_id = p.loadURDF("sphere_small.urdf", obstacle_position, useFixedBase=True, physicsClientId=client_id)
        
        obstacle_ids.append(obstacle_id)
        obstacle_positions.append(obstacle_position)

    return obstacle_ids, obstacle_positions
initial_xyzs = np.array([
    [0, 0, 0.5],      # 主控无人机
    [0.5, 0.5, 1.0]
])




def run_simulation():
    
    ground_truth_log = []
    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=DEFAULT_NUM_DRONES,
        neighbourhood_radius=np.inf,
        initial_xyzs=initial_xyzs,
 
        physics=Physics.PYB_DW,  
        pyb_freq=SIMULATION_FREQ_HZ,
        ctrl_freq=CONTROL_FREQ_HZ,
        gui=True,
        obstacles=False
    )

    
    logger = Logger(
        logging_freq_hz=CONTROL_FREQ_HZ,
        num_drones=DEFAULT_NUM_DRONES,
        duration_sec=DURATION_SEC
    )
        
    static_controllers = []
    for i in range(1, DEFAULT_NUM_DRONES):
        static_controllers.append(DSLPIDControl(drone_model=DroneModel.CF2X))
    obstacle_positions = [initial_xyzs[i] for i in range(1, DEFAULT_NUM_DRONES)]
    
    target_positions = generate_target_path(np.array([0, 0, 0.5]), np.array([1, 1, 1.5]), steps=DURATION_SEC * CONTROL_FREQ_HZ)
    client_id = env.getPyBulletClient()
   

    controller =  NloptControl (drone_model=DroneModel.CF2X,obstacle_positions=obstacle_positions)
    # controller = DSLPIDControl(drone_model=DroneModel.CF2X)
    controller._initialize_gp_samples(path_points=target_positions)

    controller.obstacle_radius = 0.2
    
   
    drone_positions = []
    timestamps = []
    sigma2_frames = []
    sigma2_vertical_profiles = []
    z_heights = np.linspace(0.3, 2.0, 50)
    obstacle_center = obstacle_positions[0] 

    # 仿真开始
    action = np.zeros((DEFAULT_NUM_DRONES, 4))  
    start_time = time.time()
    

    for step in range(DURATION_SEC * CONTROL_FREQ_HZ):
        
        obs, _, _, _, _ = env.step(action)
        
        
        cur_pos = obs[0][:3]   
        cur_quat = obs[0][3:7] 
        cur_vel = obs[0][10:13]
        cur_ang_vel = obs[0][13:16] 
        
        if np.linalg.norm(cur_quat) < 1e-6:
            print("[WARNING] cur_quat is invalid, setting to [1, 0, 0, 0]")
            cur_quat = np.array([1, 0, 0, 0])

        
        target_pos = target_positions[step]

        
        controller.obstacle_positions = obstacle_positions
        output = controller.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos
        )
        action[0, :] = output[0]
    
    
        for i in range(1, DEFAULT_NUM_DRONES):
            target_pos = initial_xyzs[i]
            action[i, :], _, _ = static_controllers[i - 1].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[i],
                target_pos=target_pos)

        
        drone_positions.append(cur_pos)
        timestamps.append(step / CONTROL_FREQ_HZ)

        
        logger.log(
            drone=0,
            timestamp=step / CONTROL_FREQ_HZ,
            state=obs[0],
            control=np.hstack([action[0], np.zeros(8)])  
        )

        
        env.render()
        vertical_sigma2 = []
        for z in z_heights:
            pos = np.array([obstacle_center[0], obstacle_center[1], z])
            x_input = np.concatenate([pos, np.zeros(3)])
            sigma2 = controller.gp_model.predict_variance(x_input)
            vertical_sigma2.append(sigma2)
        sigma2_vertical_profiles.append(vertical_sigma2)
        sigma2_frames.append(sigma2_vertical_profiles) 
        time.sleep(env.CTRL_TIMESTEP)
    
    # 关闭仿真环境
    env.close()
    # import pickle
    # with open("ground_truth_pid.pkl", "wb") as f:
    #     pickle.dump(ground_truth_log, f)
    # print("Ground truth 数据已保存为 ground_truth_pid.pkl")

    
    if len(timestamps) > 0:
        logger.save()
        logger.save_as_csv("pid_nlopt")
    else:
        print("[WARNING] No valid data recorded, skipping CSV save.")

    
    if PLOT_RESULTS:
        logger.plot()
        plot_trajectory(drone_positions, obstacle_positions)
        (x0, y0, z0), (x1, y1, z1) = auto_range(drone_positions)
        

        visualize_gp_samples(
    gp_model=controller.gp_model,
    obstacle_positions=obstacle_positions,
    drone_positions=drone_positions
        )
        plot_sigma2_vertical_over_time(sigma2_vertical_profiles, z_heights)
# =======================
# 绘制轨迹
# =======================

def auto_range(drone_positions, margin=1.0):
    drone_positions = np.array(drone_positions)
    min_pos = np.min(drone_positions, axis=0) - margin
    max_pos = np.max(drone_positions, axis=0) + margin
    return tuple(min_pos), tuple(max_pos)
def plot_trajectory(drone_positions, obstacle_positions):

    drone_positions = np.array(drone_positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], label='Drone Path', color='blue')

    first_label_added = False
    for obs_pos in obstacle_positions:
        if not first_label_added:
            ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], label="Obstacle", color='red', s=100)
            first_label_added = True
        else:
            ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], color='red', s=100)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Drone Trajectory with Multiple Obstacles")
    ax.legend()
    
    plt.show()


def visualize_gp_samples(gp_model, obstacle_positions=None, drone_positions=None):
    if hasattr(gp_model, 'X_all') and len(gp_model.X_all) > 0:
        pos = np.array(gp_model.X_all)
    elif gp_model.X_train.size(0) > 0:
        pos = gp_model.X_train.cpu().numpy()[:, :3]
    else:
        print("[GP] 当前无训练样本")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='b', s=20, label='GP Samples')

    if obstacle_positions is not None:
        for idx, obs_pos in enumerate(obstacle_positions):
            if idx == 0:
                ax.scatter(*obs_pos, c='r', s=80, label="Obstacle")
            else:
                ax.scatter(*obs_pos, c='r', s=80)

    if drone_positions is not None:
        drone_positions = np.array(drone_positions)
        ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], color='g', label='Drone Path')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("All GP Sample Positions (History)")
    ax.legend()
    plt.tight_layout()
    plt.show()

def get_sigma2_vertical_curve(gp_model, center, z_range=(0.0, 2.0), num_points=100):
    x0, y0, _ = center
    zs = np.linspace(*z_range, num_points)
    sigma2_vals = []
    for z in zs:
        pos = np.array([x0, y0, z])
        vel = np.zeros(3)
        x_input = np.concatenate([pos, vel])
        sigma2_vals.append(gp_model.predict_variance(x_input))
    return sigma2_vals

def plot_sigma2_vertical_over_time(sigma2_profiles, z_heights):
    sigma2_profiles = np.array(sigma2_profiles).T  # shape: [z, time]

    plt.figure(figsize=(8, 6))
    plt.imshow(sigma2_profiles, aspect='auto', cmap='viridis',
               extent=[0, sigma2_profiles.shape[1], z_heights[0], z_heights[-1]],
               origin='lower')
    plt.colorbar(label='σ²(x)')
    plt.xlabel("Time step")
    plt.ylabel("Height (Z)")
    plt.title("Vertical σ² Distribution Below Obstacle Over Time")
    plt.tight_layout()
    plt.show()


# =======================
# 运行主函数
# =======================
if __name__ == "__main__":
    run_simulation()
