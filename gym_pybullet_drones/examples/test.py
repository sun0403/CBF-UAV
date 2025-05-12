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
os.environ["PYBULLET_EGL"] = "1"

SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 10
PLOT_RESULTS = True  
DEFAULT_NUM_DRONES = 1


def generate_target_path(start_pos, end_pos, steps=100):
    
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

        # **随机生成速度**
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


def run_simulation():
    

    
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=DEFAULT_NUM_DRONES,
        neighbourhood_radius=np.inf,
        initial_xyzs=np.array([[0, 0, 0.5]]), 
        physics=Physics.PYB,  # 物理仿真
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
    
    target_positions = generate_target_path(np.array([0, 0, 0.5]), np.array([1, 1, 1.5]), steps=DURATION_SEC * CONTROL_FREQ_HZ)
    client_id = env.getPyBulletClient()
    obstacle_ids, obstacle_positions, obstacle_velocities, x_range, y_range, z_range = add_moving_obstacles(client_id, num_obstacles=15)

    controller =  NloptControl (drone_model=DroneModel.CF2X,obstacle_positions=obstacle_positions)
    controller.obstacle_position = obstacle_positions
    controller.obstacle_radius = 0.05

   
    drone_positions = []
    timestamps = []

    # 仿真开始
    action = np.zeros((DEFAULT_NUM_DRONES, 4))  
    start_time = time.time()
    

    for step in range(DURATION_SEC * CONTROL_FREQ_HZ):
        
        obs, _, _, _, _ = env.step(action)
        cur_pos = obs[0][:3]   
        cur_quat = obs[0][3:7] 
        cur_vel = obs[0][10:13]
        cur_ang_vel = obs[0][13:16] 
        print(f"角速度是{cur_ang_vel}")
        if np.linalg.norm(cur_quat) < 1e-6:
            print("[WARNING] cur_quat is invalid, setting to [1, 0, 0, 0]")
            cur_quat = np.array([1, 0, 0, 0])

        
        target_pos = target_positions[step]

        for i, obs_id in enumerate(obstacle_ids):
            new_position = obstacle_positions[i] + obstacle_velocities[i] * env.CTRL_TIMESTEP

            # **检查边界，并反转速度**
            if new_position[0] < x_range[0] or new_position[0] > x_range[1]:  
                obstacle_velocities[i][0] *= -1  # 反转 X 轴速度
            if new_position[1] < y_range[0] or new_position[1] > y_range[1]:  
                obstacle_velocities[i][1] *= -1  # 反转 Y 轴速度
            if new_position[2] < z_range[0] or new_position[2] > z_range[1]:  
                obstacle_velocities[i][2] *= -1  # 反转 Z 轴速度

            # **确保位置不会超出边界**
            new_position = np.clip(new_position, [x_range[0], y_range[0], z_range[0]], [x_range[1], y_range[1], z_range[1]])
            obstacle_positions[i] = new_position
            p.resetBasePositionAndOrientation(obs_id, new_position, [0, 0, 0, 1])
        controller.obstacle_positions = obstacle_positions
        action[0, :] = controller.computeControl(
            control_timestep=env.CTRL_TIMESTEP,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos
        )[0]

        
        drone_positions.append(cur_pos)
        timestamps.append(step / CONTROL_FREQ_HZ)

        
        logger.log(
            drone=0,
            timestamp=step / CONTROL_FREQ_HZ,
            state=obs[0],
            control=np.hstack([action[0], np.zeros(8)])  
        )

        
        env.render()

       
        time.sleep(env.CTRL_TIMESTEP)
    
    # 关闭仿真环境
    env.close()

    
    if len(timestamps) > 0:
        logger.save()
        logger.save_as_csv("pid_nlopt")
    else:
        print("[WARNING] No valid data recorded, skipping CSV save.")

    
    if PLOT_RESULTS:
        logger.plot()
        plot_trajectory(drone_positions, obstacle_positions)

# =======================
# 绘制轨迹
# =======================
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


# =======================
# 运行主函数
# =======================
if __name__ == "__main__":
    run_simulation()
