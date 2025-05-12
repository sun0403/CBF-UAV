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

def plot_gp_variance_3d(gp_model, x_range, y_range, z_range, num_points=20):


    xs = np.linspace(*x_range, num_points)
    ys = np.linspace(*y_range, num_points)
    zs = np.linspace(*z_range, num_points)

    X, Y, Z = np.meshgrid(xs, ys, zs)
    variances = np.zeros_like(X)

    for i in range(num_points):
        for j in range(num_points):
            for k in range(num_points):
                pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                vel = np.zeros(3)  # 速度设为 0
                input_x = np.concatenate([pos, vel])
                sigma2 = gp_model.predict_variance(input_x)
                variances[i, j, k] = sigma2

    # 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 阈值，避免太小的方差看不出来
    threshold = np.percentile(variances.flatten(), 95)

    ax.scatter(
        X[variances > threshold],
        Y[variances > threshold],
        Z[variances > threshold],
        c=variances[variances > threshold],
        cmap='viridis',
        marker='o'
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('GP Predicted Variance (sigma²) Distribution')
    plt.show()




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
        print(f"角速度是{cur_ang_vel}")
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
    
    #     ground_truth_log.append({
    #     "time": step / CONTROL_FREQ_HZ,
    #     "cur_pos": cur_pos.copy(),
    #     "cur_vel": cur_vel.copy(),
    #     "cur_quat": cur_quat.copy(),
    #     "cur_ang_vel": cur_ang_vel.copy(),
    #     "obstacles": obstacle_positions.copy()
    # })
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

       
        time.sleep(env.CTRL_TIMESTEP)
    
    # 关闭仿真环境
    env.close()
    # import pickle
    # with open("ground_truth_pid.pkl", "wb") as f:
    #     pickle.dump(ground_truth_log, f)
    # print("Ground truth 数据已保存为 ground_truth_pid.pkl")

    if hasattr(controller, "gp_model"):
        controller.gp_model.save("gp_model_sigma.pt")
    if len(timestamps) > 0:
        logger.save()
        logger.save_as_csv("pid_nlopt")
    else:
        print("[WARNING] No valid data recorded, skipping CSV save.")

    
    if PLOT_RESULTS:
        logger.plot()
        plot_trajectory(drone_positions, obstacle_positions)

        plot_gp_variance_and_trajectory(
            drone_positions,
            obstacle_positions,
            controller.gp_model,
            x_range=(0.0, 2.0),
            y_range=(0.0, 1.2),
            z_range=(0.5, 2.0)
        )
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

def plot_gp_variance_and_trajectory(drone_positions, obstacle_positions, gp_model, x_range, y_range, z_range, num_points=20):

    xs = np.linspace(*x_range, num_points)
    ys = np.linspace(*y_range, num_points)
    zs = np.linspace(*z_range, num_points)

    X, Y, Z = np.meshgrid(xs, ys, zs)
    variances = np.zeros_like(X)

    for i in range(num_points):
        for j in range(num_points):
            for k in range(num_points):
                pos = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                vel = np.zeros(3)  # 假设速度为0
                input_x = np.concatenate([pos, vel])
                sigma2 = gp_model.predict_variance(input_x)
                variances[i, j, k] = sigma2

    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 方差散点
    threshold = np.percentile(variances.flatten(), 95)  # 只画方差最大的 5%
    ax.scatter(
        X.flatten(),
        Y.flatten(),
        Z.flatten(),
        c=variances.flatten(),
        cmap='viridis',
        marker='o',
        s=15,
        alpha=0.5,
        label="Variance (sigma²)"
    )

    # 无人机飞行轨迹
    drone_positions = np.array(drone_positions)
    ax.plot(
        drone_positions[:, 0],
        drone_positions[:, 1],
        drone_positions[:, 2],
        color='blue',
        label='Drone Path'
    )

    # 障碍物
    for idx, obs_pos in enumerate(obstacle_positions):
        if idx == 0:
            ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], color='red', s=100, label='Obstacle')
        else:
            ax.scatter(obs_pos[0], obs_pos[1], obs_pos[2], color='red', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Drone Path and GP Variance Distribution')
    ax.legend()
    plt.show()


# =======================
# 运行主函数
# =======================
if __name__ == "__main__":
    run_simulation()
