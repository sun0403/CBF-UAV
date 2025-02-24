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
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 10
PLOT_RESULTS = True  
DEFAULT_NUM_DRONES = 1


def generate_target_path(start_pos, end_pos, steps=100):
    
    return np.linspace(start_pos, end_pos, steps)

def add_obstacle_on_path(client_id, path_points,place_at_ratio=0.4):
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    
    obstacle_idx = int(len(path_points) * place_at_ratio)
    obstacle_position = path_points[obstacle_idx].copy()
    
    
    obstacle_position[2] += 0.05 

    print(f"[INFO] Adding obstacle at {obstacle_position}")

    obstacle_id = p.loadURDF("sphere_small.urdf", obstacle_position, useFixedBase=True, physicsClientId=client_id)
    return obstacle_id, obstacle_position

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
    obstacle_id, obstacle_position = add_obstacle_on_path(client_id, target_positions,place_at_ratio=0.4)

    controller =  NloptControl (drone_model=DroneModel.CF2X)
    controller.obstacle_position = obstacle_position
    controller.obstacle_radius = 0.2

   
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

        if np.linalg.norm(cur_quat) < 1e-6:
            print("[WARNING] cur_quat is invalid, setting to [1, 0, 0, 0]")
            cur_quat = np.array([1, 0, 0, 0])

        
        target_pos = target_positions[step]

        
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
        plot_trajectory(drone_positions, obstacle_position)

# =======================
# 绘制轨迹
# =======================
def plot_trajectory(drone_positions, obstacle_position):
    """绘制无人机轨迹"""
    drone_positions = np.array(drone_positions)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制无人机轨迹
    ax.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2], label='Drone Path', color='blue')
    ax.scatter(obstacle_position[0], obstacle_position[1], obstacle_position[2], label="Obstacle", color='red', s=100)


    # 标注
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("Drone Trajectory with PID Control")
    ax.legend()
    
    plt.show()

# =======================
# 运行主函数
# =======================
if __name__ == "__main__":
    run_simulation()
