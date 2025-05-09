import rospy
import nest
import nest.voltage_trace
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist, PoseStamped
import time
from scipy.spatial import distance
from scipy.interpolate import splprep, splev
import math

# 初始化 ROS 节点
rospy.init_node('snakebot_snn_controller')
bridge = CvBridge()

# 仿真参数
img_size = (16, 16)  # DVS下采样尺寸
n_input = img_size[0] * img_size[1]
n_hidden = 100
n_output = 4
sim_time = 50.0

# 轨迹规划参数
class TrajectoryPlanner:
    def __init__(self):
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 0.5  # 到达路径点的距离阈值
        self.smooth_trajectory = None
        self.trajectory_length = 0
        
    def add_waypoint(self, x, y):
        self.waypoints.append([x, y])
        self._update_smooth_trajectory()
        
    def _update_smooth_trajectory(self):
        if len(self.waypoints) >= 2:
            waypoints = np.array(self.waypoints)
            tck, u = splprep(waypoints.T, u=None, s=0.0, per=0)
            u_new = np.linspace(u.min(), u.max(), 100)
            self.smooth_trajectory = np.column_stack(splev(u_new, tck))
            self.trajectory_length = self._calculate_trajectory_length()
            
    def _calculate_trajectory_length(self):
        if self.smooth_trajectory is None:
            return 0
        return np.sum(np.sqrt(np.sum(np.diff(self.smooth_trajectory, axis=0)**2, axis=1)))
    
    def get_next_waypoint(self):
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]
        return None
    
    def update_waypoint(self, current_pos):
        if self.current_waypoint_idx < len(self.waypoints):
            dist = distance.euclidean(current_pos, self.waypoints[self.current_waypoint_idx])
            if dist < self.waypoint_threshold:
                self.current_waypoint_idx += 1
                return True
        return False
    
    def get_trajectory_progress(self, current_pos):
        if self.smooth_trajectory is None:
            return 0.0
        # 找到最近的点
        distances = np.sqrt(np.sum((self.smooth_trajectory - current_pos)**2, axis=1))
        closest_idx = np.argmin(distances)
        # 计算已完成的轨迹长度
        completed_length = np.sum(np.sqrt(np.sum(np.diff(self.smooth_trajectory[:closest_idx+1], axis=0)**2, axis=1)))
        return completed_length / self.trajectory_length if self.trajectory_length > 0 else 0.0

# 性能评估类
class PerformanceEvaluator:
    def __init__(self):
        self.trajectory_history = []
        self.action_history = []
        self.reward_history = []
        self.start_time = time.time()
        
    def record_step(self, position, action, reward):
        self.trajectory_history.append(position)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
    def calculate_metrics(self):
        if len(self.trajectory_history) < 2:
            return {}
            
        # 计算平均速度
        positions = np.array(self.trajectory_history)
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_time = time.time() - self.start_time
        avg_speed = np.mean(distances) / total_time if total_time > 0 else 0
        
        # 计算动作平滑度
        action_changes = np.diff(self.action_history)
        smoothness = 1.0 / (1.0 + np.mean(np.abs(action_changes)))
        
        # 计算奖励统计
        reward_stats = {
            'mean': np.mean(self.reward_history),
            'std': np.std(self.reward_history),
            'max': np.max(self.reward_history)
        }
        
        return {
            'avg_speed': avg_speed,
            'smoothness': smoothness,
            'reward_stats': reward_stats,
            'total_distance': np.sum(distances),
            'total_time': total_time
        }

# 初始化轨迹规划器和性能评估器
trajectory_planner = TrajectoryPlanner()
performance_evaluator = PerformanceEvaluator()

# === NEST ===
nest.ResetKernel()
nest.SetKernelStatus({"resolution": 0.1})  

input_generators = nest.Create('poisson_generator', n_input)
hidden_neurons = nest.Create('iaf_psc_alpha', n_hidden, params={
    "I_e": 70.0,  
    "tau_m": 10.0,  
    "V_th": -55.0,  
    "V_reset": -70.0,  
    "t_ref": 2.0  
})
output_neurons = nest.Create('iaf_psc_alpha', n_output, params={
    "I_e": 0.0,
    "tau_m": 20.0,  
    "V_th": -55.0,
    "V_reset": -70.0,
    "t_ref": 2.0
})


spike_detectors = nest.Create("spike_detector", n_output, params={"withgid": True, "withtime": True})
voltmeter = nest.Create("voltmeter", params={"withgid": True, "withtime": True, "interval": 0.1})


for i in range(n_output):
    nest.Connect([output_neurons[i]], [spike_detectors[i]])
nest.Connect(voltmeter, output_neurons)

# R-STDP 
input_to_hidden_syn = {
    "model": "stdp_synapse",
    "weight": {"distribution": "normal", "mu": 1.0, "sigma": 0.5},
    "delay": 1.0,
    "alpha": 1.0,  # lr
    "lambda": 0.01,  
    "mu_plus": 1.0,  
    "mu_minus": 1.0,  
    "Wmax": 100.0,  
    "Wmin": 0.0  
}


hidden_to_output_syn = {
    "model": "stdp_synapse",
    "weight": {"distribution": "normal", "mu": 2.0, "sigma": 0.8},
    "delay": 1.0,
    "alpha": 0.5,
    "lambda": 0.01,
    "mu_plus": 1.0,
    "mu_minus": 1.0,
    "Wmax": 100.0,
    "Wmin": 0.0
}


input_to_hidden_conn = nest.Connect(input_generators, hidden_neurons, 
                                    conn_spec={"rule": "all_to_all"}, 
                                    syn_spec=input_to_hidden_syn)

hidden_to_output_conn = nest.Connect(hidden_neurons, output_neurons, 
                                    conn_spec={"rule": "all_to_all"}, 
                                    syn_spec=hidden_to_output_syn)


lateral_inhibition = {
    "model": "static_synapse",
    "weight": -10.0,  
    "delay": 1.0
}

for i in range(n_output):
    for j in range(n_output):
        if i != j:  # 避免自连接
            nest.Connect([output_neurons[i]], [output_neurons[j]], syn_spec=lateral_inhibition)

# 发布话题
motor_pub = []
for i in range(4):  
    motor_pub.append(rospy.Publisher(f'/snake_like_robo/joint{i+1}_position_controller/command', 
                                    Float64, queue_size=1))

# 全局变量
last_reward = 0.0
cumulative_reward = 0.0
spike_history = np.zeros((n_output, 10))  # 记录最近10帧的脉冲数
target_trajectory = None  # 目标轨迹
current_position = np.array([0, 0])  # 当前位置
last_action = -1  # 上一次的动作

# 轨迹跟踪变量
trajectory_history = []
last_update_time = time.time()

def encode_image(img):
    img = cv2.equalizeHist(img)
    
    # DVS编码
    if hasattr(encode_image, 'prev_img'):
        diff = cv2.absdiff(img, encode_image.prev_img)
        thresh = 20  # 变化阈值
        mask = diff > thresh
        rates = np.zeros_like(img, dtype=float)
        rates[mask] = 255
    else:
        rates = img.copy().astype(float)
    
    encode_image.prev_img = img.copy()
    
    # 降采样并转换为发放率
    rates = cv2.resize(rates, img_size)
    rates = (rates.flatten() / 255.0 * 200.0).tolist()  # 映射到0-200Hz
    
    # 设置Poisson生成器的发放率
    for gid, rate in zip(input_generators, rates):
        nest.SetStatus([gid], {'rate': rate})

# 初始化前一帧
encode_image.prev_img = np.zeros(img_size, dtype=np.uint8)

# 输出电压 - 关节函数
def snn_output_to_angle():
    potentials = nest.GetStatus(output_neurons, keys="V_m")
    
    # 将膜电位转换为关节角度，不同输出控制不同的行为模式
    base_angles = np.zeros(4)
    
    # 根据当前最活跃的输出神经元确定行为
    max_idx = np.argmax(potentials)
    
    if max_idx == 0:  # 左转
        base_angles = np.array([0.3, -0.3, 0.3, -0.3])
    elif max_idx == 1:  # 右转
        base_angles = np.array([-0.3, 0.3, -0.3, 0.3])
    elif max_idx == 2:  # 直行
        base_angles = np.array([0.2, -0.2, 0.2, -0.2])
    else:  # 停止
        base_angles = np.array([0.0, 0.0, 0.0, 0.0])
    
    # 映射电位变化到微调角度
    fine_tune = [(v + 70.0) / 140.0 * np.pi / 8 for v in potentials]
    angles = base_angles + np.array(fine_tune) * 0.5
    
    angles = np.clip(angles, -np.pi/4, np.pi/4)
    
    return angles, max_idx

# R-STDP 
def apply_reward(conns, reward_signal):
    global last_reward
    
    reward_gradient = reward_signal - last_reward
    last_reward = reward_signal
    
    # 仅当奖励有显著变化时更新
    if abs(reward_gradient) < 0.05:
        return
    
    # 获取当前权重
    weights = nest.GetStatus(conns, keys="weight")
    
    # 计算新权重
    if reward_gradient > 0:  
        new_weights = [w * (1.0 + 0.05 * reward_gradient) for w in weights]
    else:  
        new_weights = [w * (1.0 + 0.02 * reward_gradient) for w in weights]
    
    for conn, w in zip(conns, new_weights):
        w_bounded = max(0.0, min(w, 100.0))
        nest.SetStatus([conn], {"weight": w_bounded})

# 轨迹奖励函数
def compute_trajectory_reward(img, current_action, current_position):
    global cumulative_reward
    
    # 路径跟踪奖励
    waypoint_reward = 0.0
    next_waypoint = trajectory_planner.get_next_waypoint()
    if next_waypoint is not None:
        dist_to_waypoint = distance.euclidean(current_position, next_waypoint)
        waypoint_reward = max(0.0, 1.0 - dist_to_waypoint / 5.0)  # 5.0是最大距离阈值
        
    # 轨迹进度奖励
    progress_reward = trajectory_planner.get_trajectory_progress(current_position)
    
    # 动作连续性奖励
    action_consistency_reward = 0.0
    if hasattr(compute_trajectory_reward, 'last_action') and compute_trajectory_reward.last_action == current_action:
        action_consistency_reward = 0.1
    compute_trajectory_reward.last_action = current_action
    
    # 光流分析奖励
    flow_reward = compute_flow_reward(img)
    
    # 障碍物避免奖励
    obstacle_reward = 1.0 - min(1.0, np.sum(img > 200) / (img_size[0] * img_size[1] * 0.3))
    
    # 能量效率奖励（基于动作幅度）
    energy_efficiency = 1.0 - min(1.0, np.abs(current_action) / 4.0)
    
    # 综合奖励计算
    reward = (
        0.3 * waypoint_reward +          # 路径点跟踪
        0.2 * progress_reward +          # 轨迹进度
        0.1 * action_consistency_reward + # 动作连续性
        0.15 * flow_reward +             # 光流
        0.1 * obstacle_reward +          # 障碍物避免
        0.15 * energy_efficiency         # 能量效率
    )
    
    # 更新累积奖励
    cumulative_reward = 0.9 * cumulative_reward + 0.1 * reward
    
    # 记录性能指标
    performance_evaluator.record_step(current_position, current_action, reward)
    
    return reward

# 光流奖励计算
def compute_flow_reward(img):
    if not hasattr(compute_flow_reward, 'prev_img'):
        compute_flow_reward.prev_img = img.copy()
        return 0.0
        
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(
        compute_flow_reward.prev_img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # 计算光流方向和幅度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 计算期望运动方向（向前）
    expected_angle = np.pi/2  # 90度，向上运动
    angle_diff = np.abs(angle - expected_angle)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    
    # 计算方向一致性奖励
    direction_reward = np.mean(np.cos(angle_diff))
    
    # 计算运动幅度奖励
    magnitude_reward = np.mean(magnitude) / 10.0  # 归一化
    
    compute_flow_reward.prev_img = img.copy()
    
    return 0.7 * direction_reward + 0.3 * magnitude_reward

# Spike计数和行为检测
def detect_behavior(spike_detectors):
    global spike_history
    
    # 获取自上次检测以来的新脉冲数量
    new_spikes = np.array([nest.GetStatus([sd], 'n_events')[0] for sd in spike_detectors])
    
    # 重置检测器
    for sd in spike_detectors:
        nest.SetStatus([sd], {'n_events': 0})
    
    # 更新脉冲历史
    spike_history = np.roll(spike_history, 1, axis=1)
    spike_history[:, 0] = new_spikes
    
    # 计算最近历史中的累积脉冲
    cumulative_spikes = np.sum(spike_history, axis=1)
    
    # 选择脉冲最多的动作
    if np.sum(cumulative_spikes) == 0:
        action = 3  # 无脉冲时默认停止
    else:
        action = np.argmax(cumulative_spikes)
    
    # 输出行为信息
    actions = ["左转", "右转", "直行", "停止"]
    rospy.loginfo(f"[SNN行为检测] 当前行为: {actions[action]}, 脉冲计数: {cumulative_spikes}")
    
    return action, cumulative_spikes

# 记录和可视化膜电位
def record_membrane_potentials():
    global last_update_time
    current_time = time.time()
    
    # 每秒最多更新一次图表
    if current_time - last_update_time < 1.0:
        return
    
    last_update_time = current_time
    
    # 获取电压记录数据
    events = nest.GetStatus(voltmeter)[0]['events']
    times = events['times']
    senders = events['senders']
    potentials = events['V_m']
    
    # 清空电压记录器的事件
    nest.SetStatus(voltmeter, {'n_events': 0})
    
    # 如果没有足够的数据，不绘图
    if len(times) < 10:
        return
    
    # 可视化每个输出神经元的膜电位
    plt.figure(figsize=(10, 6))
    for i in range(n_output):
        idx = np.where(senders == output_neurons[i])[0]
        if len(idx) > 0:
            plt.plot(times[idx], potentials[idx], label=f"神经元 {i}")
    
    plt.axhline(y=-55.0, color='r', linestyle='--', label="阈值")
    plt.xlabel('时间 (ms)')
    plt.ylabel('膜电位 (mV)')
    plt.title('输出神经元膜电位')
    plt.legend()
    plt.ylim(-80, -50)
    
    plt.savefig('/tmp/membrane_potentials.png')
    plt.close()

# 图像回调函数
def dvs_callback(msg):
    global last_action
    
    # 转换ROS图像消息为OpenCV格式
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
    
    encode_image(cv_img)
    
    nest.Simulate(sim_time)
    
    # 获取输出角度和当前行为
    angles, action_idx = snn_output_to_angle()
    
    # 发布关节指令
    for i in range(min(len(angles), len(motor_pub))):
        motor_pub[i].publish(Float64(angles[i]))
    
    # 行为检测
    action, spike_counts = detect_behavior(spike_detectors)
    last_action = action
    
    # 计算奖励
    reward = compute_trajectory_reward(cv_img, action, current_position)
    
    # R-STDP
    apply_reward(input_to_hidden_conn, reward)
    apply_reward(hidden_to_output_conn, reward)
  
    record_membrane_potentials()
    
    rospy.loginfo(f"[SNN控制器] 当前奖励: {reward:.4f}, 累积奖励: {cumulative_reward:.4f}")

# 完善主循环
def main():
    # 设置初始路径点
    trajectory_planner.add_waypoint(0, 0)
    trajectory_planner.add_waypoint(5, 5)
    trajectory_planner.add_waypoint(10, 0)
    trajectory_planner.add_waypoint(15, 5)
    
    # 订阅DVS图像话题
    rospy.Subscriber('/dvs/image_raw', Image, dvs_callback)
    
    # 主循环
    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        # 获取当前位置（需要根据实际机器人状态更新）
        current_position = np.array([0, 0])  # 实际位置
        
        # 更新路径点
        trajectory_planner.update_waypoint(current_position)
        
        # 运行SNN仿真
        nest.Simulate(sim_time)
        
        # 获取输出并转换为关节角度
        angles, action = snn_output_to_angle()
        
        # 发布关节角度
        for i, angle in enumerate(angles):
            motor_pub[i].publish(angle)
        
        # 计算并应用奖励
        reward = compute_trajectory_reward(encode_image.prev_img, action, current_position)
        apply_reward(input_to_hidden_conn, reward)
        apply_reward(hidden_to_output_conn, reward)
        
        # 定期输出性能指标
        if rospy.get_time() % 5.0 < 0.1:  # 每5秒输出一次
            metrics = performance_evaluator.calculate_metrics()
            rospy.loginfo(f"Performance Metrics: {metrics}")
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass 
