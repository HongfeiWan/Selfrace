import torch
from typing import Dict
from randomize_components import RewardParameterSampler

class RewardCalculator:
    """
    该模块的设计可配置，并遵循论文中描述的奖励结构。
    输入状态张量 agents_state 的完整结构 (B, M, N):
    - B: 批次大小 (batch size)
    - M: 智能体数量 (number of agents)
    - N: 状态维度，需要包含以下字段 (按索引顺序):
    
    索引 0-1: 位置信息
        [0]: x - 智能体在全局坐标系中的 x 坐标
        [1]: y - 智能体在全局坐标系中的 y 坐标
    索引 2-3: 运动学信息
        [2]: heading - 智能体的朝向角度 (弧度)
        [3]: speed - 智能体的速度大小
    索引 4-7: 动力学信息
        [4]: along - 纵向加速度
        [5]: alat - 横向加速度
        [6]: along_jerk - 纵向加加速度 (jerk)
        [7]: alat_jerk - 横向加加速度 (jerk)
    索引 8-9: Frenet 坐标系信息
        [8]: theta_f - Frenet 坐标系中的角度误差 (车道角度误差)
        [9]: d - Frenet 坐标系中的横向距离 (相对于车道中心的横向偏移)
    
    奖励系统包含以下组件:
    1. 碰撞惩罚: Rcollision = -(αcollision + 0.1|v|)1collision
    2. 离路惩罚: Roff-road = -αboundary1boundary  
    3. 舒适度惩罚: Rcomfort = -αcomfort * (1|along|>3 + 1|alat|>3 + 1|along_jerk|>5 ∨ |alat_jerk|>5)
    4. 车道对齐奖励: R_{l-align} = α_{l-align} * Δt * (min(cos(θ_f), 0) + α_{vel-align} * min(cos(θ_f) * v, 0) + 0.0025 * (1 - |θ_f|/(π/2)))
    5. 车道中心对齐奖励: R_{l-center} = -α_{l-center} * Δt * (1_{cos(θ_f) > 0.5} * |d - α_{center-bias}| - 0.05/exp(|d - α_{center-bias}| - 0.5))
    6. 速度奖励: Rvelocity = αvelocity * Δt * max(cos(θ_f), 0.0) * 1_{|v| > 2.5}
    7. 倒车惩罚: Rreverse = -αreverse * Δt * 1_{v < 0}
    8. 停止线违规惩罚: Rstop-line = -αstop-line * 1stop-line-violation
    9. 时间步惩罚: Rtimestep = -(αtimestep * Δt) * 1_{|v| > 0 ∨ |a| > 0}
    10. 目标奖励: Rgoal = 1 if (||x-g|| < δgoal ∧ (waypoint_reached ∨ |v| < vgoal)) else 0
    """

    def __init__(self, config: Dict, device: torch.device):
        """
        初始化奖励计算器。
        Args:
            config (Dict): 包含奖励参数的配置字典。
            device (torch.device): 计算设备。
        """
        self.device = device
        self.config = config
        sim_cfg = config.get('simulator')
        self.reward_config = sim_cfg.get('reward')
        # 初始化参数采样器
        self.parameter_sampler = RewardParameterSampler(self.reward_config, device)
        # 从配置中读取默认的批次尺寸 (B, M)，若不存在则回退到1
    
        self.B = sim_cfg.get('num_envs')
        self.M = sim_cfg.get('max_agents_num')

        # 从配置中加载固定参数
        self.v_goal = self.reward_config.get('v_goal', 3.0)
        self.goal_reward = self.reward_config.get('goal_reward', 1.0)
        self.collision_speed_mult = self.reward_config.get('collision_speed_mult', 0.1)
        # 初始化所有随机参数
        self.sampled_params = self.parameter_sampler.sample_all_parameters(self.B, self.M)
        self.last_reward_components = {}
        
        # 参数名到sampled_params索引映射
        self._param_name_to_idx = {
            'delta_goal': 0,
            'collision_alpha': 1,
            'boundary_alpha': 2,
            'comfort_alpha': 3,
            'l_align_alpha': 4,
            'vel_align_alpha': 5,
            'l_center_alpha': 6,
            'center_bias_alpha': 7,
            'velocity_alpha': 8,
            'reverse_alpha': 9,
            'stop_line_alpha': 10,
            'timestep_alpha': 11,
        }
        
    def reset_episode(self):
        """
        重置episode相关的随机参数。
        在每个新episode开始时调用。
        """
        # 重新采样所有参数
        self.sampled_params = self.parameter_sampler.sample_all_parameters(self.B, self.M)
        self.last_reward_components = {}

    def reset_worlds(self, world_indices: torch.Tensor):
        """Resample reward conditioning only for the selected vector worlds."""
        world_indices = world_indices.to(device=self.device, dtype=torch.long).reshape(-1)
        if world_indices.numel() == 0:
            return
        sampled = self.parameter_sampler.sample_all_parameters(
            int(world_indices.numel()), self.M
        )
        self.sampled_params[world_indices] = sampled
        self.last_reward_components = {}
    
    def calculate_goal_reward(self, 
                             agent_positions: torch.Tensor,
                             goal_positions: torch.Tensor,
                             speeds: torch.Tensor,
                             waypoint_reached: torch.Tensor) -> torch.Tensor:
        """
        计算目标奖励 Rgoal。
        Rgoal = 1 if (||x-g|| < δgoal ∧ (waypoint_reached ∨ |v| < vgoal)) else 0
        Args:
            agent_positions (torch.Tensor): 智能体位置 (B, M, 2) - (x, y)
            goal_positions (torch.Tensor): 目标位置 (B, M, 2) - (x, y)
            speeds (torch.Tensor): 智能体速度 (B, M)
            waypoint_reached (torch.Tensor): 是否到达路点 (B, M) - 布尔值
        Returns:
            torch.Tensor: 目标奖励 (B, M)
        """
        B, M, _ = agent_positions.shape
        
        # 检查并修复goal_positions的形状
        
        if goal_positions.dim() == 2 and goal_positions.shape[1] == 2:
            # 如果goal_positions是 (B*M, 2)，恢复为 (B, M, 2)
            goal_positions = goal_positions.view(B, M, 2)


        # 计算距离 ||x-g||
        distances = torch.norm(agent_positions - goal_positions, dim=-1)  # (B, M)
        # 逐元素 δgoal 阈值 (B, M)
        delta_goal_tensor = self.sampled_params[..., self._param_name_to_idx['delta_goal']]
        try:
            delta_goal_tensor = delta_goal_tensor.expand_as(distances)
        except Exception:
            pass

        # 距离条件: ||x-g|| < δgoal(B,M)

        distance_condition = distances < delta_goal_tensor
        # 速度条件: |v| < vgoal
        speed_condition = torch.abs(speeds) < self.v_goal
        
        # 路点条件: waypoint_reached
        waypoint_condition = waypoint_reached
        
        # 综合条件: (||x-g|| < δgoal) ∧ (waypoint_reached ∨ |v| < vgoal)
        goal_condition = distance_condition & (waypoint_condition | speed_condition)
        
        # 计算奖励
        goal_rewards = torch.where(goal_condition, 
                                  torch.full_like(goal_condition, self.goal_reward, dtype=torch.float32),
                                  torch.zeros_like(goal_condition, dtype=torch.float32))

        return goal_rewards

    def calculate_collision_penalty(self, 
                                    agents_state: torch.Tensor,
                                    all_collisions: torch.Tensor) -> torch.Tensor:
            """
            计算碰撞惩罚 Rcollision。
            Rcollision = -(αcollision + 0.1|v|)1collision
            Args:
                agents_state (torch.Tensor): 智能体状态张量 (B, M, N)
                all_collisions (torch.Tensor): 碰撞掩码 (B, M)，布尔值张量
            Returns:
                torch.Tensor: 碰撞惩罚 (B, M)
            """
            B, M, _ = agents_state.shape
            # 提取速度信息
            speeds = agents_state[..., 3]  # 速度
            # 计算碰撞惩罚：-(αcollision + 0.1|v|)·1collision
            collision_penalty = torch.zeros((B, M), device=self.device)
            alpha_collision = self.sampled_params[..., self._param_name_to_idx['collision_alpha']]
            # 只对发生碰撞的智能体计算惩罚
            collision_penalty[all_collisions] = -(
                alpha_collision[all_collisions] + 
                self.collision_speed_mult * torch.abs(speeds[all_collisions])
            )
            return collision_penalty

    def calculate_offroad_penalty(self, 
                                 offroad_mask: torch.Tensor) -> torch.Tensor:
        """
        计算离路惩罚 Roff-road。
        Roff-road = -αboundary1boundary
        
        Args:
            offroad_mask (torch.Tensor): 离路掩码 (B, M)，布尔值张量
            
        Returns:
            torch.Tensor: 离路惩罚 (B, M)
        """
        # 计算离路惩罚：-αboundary·1boundary
        offroad_penalty = torch.zeros_like(offroad_mask, dtype=torch.float32, device=self.device)
        alpha_boundary = self.sampled_params[..., self._param_name_to_idx['boundary_alpha']]
        # 只对离路的智能体计算惩罚
        offroad_penalty[offroad_mask] = -alpha_boundary[offroad_mask]
        
        return offroad_penalty

    def calculate_comfort_penalty(self, 
                                 along: torch.Tensor,
                                 alat: torch.Tensor,
                                 along_jerk: torch.Tensor,
                                 alat_jerk: torch.Tensor) -> torch.Tensor:
        """
        计算舒适度惩罚 Rcomfort。
        Rcomfort = -αcomfort * (1|along|>3 + 1|alat|>3 + 1|along_jerk|>5 ∨ |alat_jerk|>5)
        
        Args:
            along (torch.Tensor): 纵向加速度 (B, M)
            alat (torch.Tensor): 横向加速度 (B, M)
            along_jerk (torch.Tensor): 纵向加加速度 (B, M)
            alat_jerk (torch.Tensor): 横向加加速度 (B, M)
        Returns:
            torch.Tensor: 舒适度惩罚 (B, M)
        """

        # 计算各项指标
        along_violation = (torch.abs(along) > 3.0).float()  # 1|along|>3
        alat_violation = (torch.abs(alat) > 3.0).float()    # 1|alat|>3
        
        # 加加速度违规: 1|along_jerk|>5 ∨ |alat_jerk|>5
        along_jerk_violation = (torch.abs(along_jerk) > 5.0).float()
        alat_jerk_violation = (torch.abs(alat_jerk) > 5.0).float()
        jerk_violation = torch.max(along_jerk_violation, alat_jerk_violation)  # ∨ 操作
        
        # 总违规次数
        total_violations = along_violation + alat_violation + jerk_violation
        
        # 计算舒适度惩罚（逐元素 αcomfort）
        alpha_comfort = self.sampled_params[..., self._param_name_to_idx['comfort_alpha']]
        comfort_penalty = -alpha_comfort * total_violations
        
        return comfort_penalty

    def calculate_lane_alignment_reward(self,
                                       theta_f: torch.Tensor,
                                       speeds: torch.Tensor,
                                       dt: float = 0.3) -> torch.Tensor:
        """
        计算车道对齐奖励 R_{l-align}。
        R_{l-align} = α_{l-align} * Δt * (min(cos(θ_f), 0) + α_{vel-align} * min(cos(θ_f) * v, 0) + 0.0025 * (1 - |θ_f|/(π/2)))
        Args:
            theta_f (torch.Tensor): 车道角度误差 (B, M)，弧度
            speeds (torch.Tensor): 智能体速度 (B, M)
            dt (float): 时间步长，默认0.3
            
        Returns:
            torch.Tensor: 车道对齐奖励 (B, M)
        """
        # 计算 cos(θ_f)
        cos_theta_f = torch.cos(theta_f)
        
        # 第一项: min(cos(θ_f), 0)
        term1 = torch.min(cos_theta_f, torch.zeros_like(cos_theta_f))
        
        # 第二项: α_{vel-align} * min(cos(θ_f) * v, 0)
        cos_theta_v = cos_theta_f * speeds
        alpha_vel_align = self.sampled_params[..., self._param_name_to_idx['vel_align_alpha']]
        term2 = alpha_vel_align * torch.min(cos_theta_v, torch.zeros_like(cos_theta_v))
        
        # 第三项: 0.0025 * (1 - |θ_f|/(π/2))
        theta_ratio = torch.abs(theta_f) / (torch.pi / 2)
        term3 = 0.0025 * (1.0 - theta_ratio)
        
        # 计算总奖励（逐元素 α_{l-align}）
        alpha_l_align = self.sampled_params[..., self._param_name_to_idx['l_align_alpha']]
        lane_alignment_reward = alpha_l_align * dt * (term1 + term2 + term3)
        
        return lane_alignment_reward

    def calculate_lane_center_reward(self,
                                    theta_f: torch.Tensor,
                                    d: torch.Tensor,
                                    dt: float = 0.3) -> torch.Tensor:
        """
        计算车道中心对齐奖励 R_{l-center}。
        R_{l-center} = -α_{l-center} * Δt * (1_{cos(θ_f) > 0.5} * |d - α_{center-bias}| - 0.05/exp(|d - α_{center-bias}| - 0.5))
        Args:
            theta_f (torch.Tensor): 车道角度误差 (B, M)
            d (torch.Tensor): Frenet坐标系中的横向距离 (B, M)
            dt (float): 时间步长，默认0.3
        Returns:
            torch.Tensor: 车道中心对齐奖励 (B, M)
        """

        # 计算 cos(θ_f)
        cos_theta_f = torch.cos(theta_f)
        # 计算 |d - α_{center-bias}|
        alpha_center_bias = self.sampled_params[..., self._param_name_to_idx['center_bias_alpha']]
        d_center_diff = torch.abs(d - alpha_center_bias)
        # 第一项: 1_{cos(θ_f) > 0.5} * |d - α_{center-bias}|
        cos_condition = (cos_theta_f > 0.5).float()
        term1 = cos_condition * d_center_diff
        # 第二项: 0.05/exp(|d - α_{center-bias}| - 0.5)
        exp_term = torch.exp(d_center_diff - 0.5)
        term2 = 0.05 / exp_term
        
        # 计算总奖励 (注意是负号，因为这是惩罚)（逐元素 α_{l-center}）
        alpha_l_center = self.sampled_params[..., self._param_name_to_idx['l_center_alpha']]
        lane_center_reward = -alpha_l_center * dt * (term1 - term2)
        
        return lane_center_reward

    def calculate_velocity_reward(self,
                                 theta_f: torch.Tensor,
                                 speeds: torch.Tensor,
                                 dt: float = 0.3) -> torch.Tensor:
        """
        计算速度奖励 Rvelocity。
        
        Rvelocity = αvelocity * Δt * max(cos(θ_f), 0.0) * 1_{|v| > 2.5}
        
        Args:
            theta_f (torch.Tensor): 车道角度误差 (B, M)
            speeds (torch.Tensor): 智能体速度 (B, M)
            dt (float): 时间步长，默认0.3
            
        Returns:
            torch.Tensor: 速度奖励 (B, M)
        """
        # 计算 cos(θ_f)
        cos_theta_f = torch.cos(theta_f)
        
        # 计算 max(cos(θ_f), 0.0)
        max_cos_theta = torch.max(cos_theta_f, torch.zeros_like(cos_theta_f))
        
        # 计算速度条件: 1_{|v| > 2.5}
        speed_condition = (torch.abs(speeds) > 2.5).float()
        
        # 计算速度奖励（逐元素 alpha_velocity，与 C_reward conditioning 一致）
        alpha_velocity = self.sampled_params[..., self._param_name_to_idx['velocity_alpha']]
        velocity_reward = alpha_velocity * dt * max_cos_theta * speed_condition
        
        return velocity_reward

    def calculate_reverse_penalty(self,
                                 speeds: torch.Tensor,
                                 dt: float = 0.3) -> torch.Tensor:
        """
        计算倒车惩罚 Rreverse。
        
        Rreverse = -αreverse * Δt * 1_{v < 0}
        
        Args:
            speeds (torch.Tensor): 智能体速度 (B, M)
            dt (float): 时间步长，默认0.3
            
        Returns:
            torch.Tensor: 倒车惩罚 (B, M)
        """
        # 计算倒车条件: 1_{v < 0}
        reverse_condition = (speeds < 0.0).float()
        # 计算倒车惩罚（逐元素 αreverse）
        alpha_reverse = self.sampled_params[..., self._param_name_to_idx['reverse_alpha']]
        reverse_penalty = -alpha_reverse * dt * reverse_condition
        
        return reverse_penalty

    def calculate_stop_line_penalty(self,
                                  stop_line_violation: torch.Tensor) -> torch.Tensor:
        """
        计算停止线违规惩罚 Rstop-line。
        
        Rstop-line = -αstop-line * 1stop-line-violation
        
        Args:
            stop_line_violation (torch.Tensor): 停止线违规掩码 (B, M) - 布尔值
            
        Returns:
            torch.Tensor: 停止线违规惩罚 (B, M)
        """
        # 如果stop_line_violation为None，返回标量零
        if stop_line_violation is None:
            return torch.tensor(0.0, device=self.device)
        
        # 计算停止线违规惩罚
        alpha_stop = self.sampled_params[..., self._param_name_to_idx['stop_line_alpha']]
        stop_line_penalty = -alpha_stop * stop_line_violation.float()
        
        return stop_line_penalty

    def calculate_timestep_penalty(self,
                                  speeds: torch.Tensor,
                                  along: torch.Tensor,
                                  alat: torch.Tensor,
                                  dt: float = 0.3) -> torch.Tensor:
        """
        计算时间步惩罚 Rtimestep。
        
        Rtimestep = -(αtimestep * Δt) * 1_{|v| > 0 ∨ |a| > 0}
        
        Args:
            speeds (torch.Tensor): 智能体速度 (B, M)
            along (torch.Tensor): 纵向加速度 (B, M)
            alat (torch.Tensor): 横向加速度 (B, M)
            dt (float): 时间步长，默认0.3
            
        Returns:
            torch.Tensor: 时间步惩罚 (B, M)
        """
        # 计算速度条件: |v| > 0
        speed_condition = (torch.abs(speeds) > 0.0).float()
        
        # 计算加速度条件: |a| > 0 (使用纵向和横向加速度的合成)
        acceleration_magnitude = torch.sqrt(along**2 + alat**2)
        acceleration_condition = (acceleration_magnitude > 0.0).float()
        
        # 综合条件: |v| > 0 ∨ |a| > 0
        timestep_condition = torch.max(speed_condition, acceleration_condition)
        
        # 计算时间步惩罚（逐元素 alpha_timestep，与 C_reward conditioning 一致）
        alpha_timestep = self.sampled_params[..., self._param_name_to_idx['timestep_alpha']]
        timestep_penalty = -(alpha_timestep * dt) * timestep_condition
        
        return timestep_penalty

    def calculate(self, 
                  agents_state: torch.Tensor,
                  all_collisions: torch.Tensor, 
                  offroad_mask: torch.Tensor,
                  dt: float = 0.3,
                  goal_positions: torch.Tensor = None,
                  waypoint_reached: torch.Tensor = None,
                  stop_line_violation: torch.Tensor = None) -> torch.Tensor:
        """
        为所有智能体计算总奖励。
        Args:
            agents_state (torch.Tensor): 智能体状态张量，形状为 (B, M, N)。
                B: 批次大小 (batch size)
                M: 智能体数量
                N: 状态维度
                论文中的设置是[38400,150,10]
                其中 N = 10，包含以下字段:
                - [0]: x - 全局 x 坐标
                - [1]: y - 全局 y 坐标  
                - [2]: heading - 朝向角度 (弧度)
                - [3]: speed - 速度大小
                - [4]: along - 纵向加速度
                - [5]: alat - 横向加速度
                - [6]: along_jerk - 纵向加加速度
                - [7]: alat_jerk - 横向加加速度
                - [8]: theta_f - Frenet 角度误差
                - [9]: d - Frenet 横向距离
            all_collisions (torch.Tensor): 碰撞掩码 (B, M)，布尔值张量
            offroad_mask (torch.Tensor): 离路掩码 (B, M)，布尔值张量
            goal_positions (torch.Tensor): 目标位置 (B, M, 2)
            waypoint_reached (torch.Tensor): 路点到达掩码 (B, M)，布尔值张量
            stop_line_violation (torch.Tensor): 停止线违规掩码 (B, M)，布尔值张量
            
        Returns:s
            torch.Tensor: 每个智能体的总奖励 (B, M)
        """

        # 提取信息
        agent_positions = agents_state[..., :2]  # 提取位置信息 (x, y)
        speeds = agents_state[..., 3]  # 速度
        along = agents_state[..., 4]  # 纵向加速度
        alat = agents_state[..., 5]   # 横向加速度
        along_jerk = agents_state[..., 6]  # 纵向加加速度
        alat_jerk = agents_state[..., 7]   # 横向加加速度
        theta_f = agents_state[..., 8]  # 车道角度误差
        d = agents_state[..., 9]  # Frenet坐标系中的横向距离

        B, M, _ = agents_state.shape
        # 归零奖励
        reward = torch.zeros((B, M), device=self.device)

        # 计算目标奖励
        goal_reached = torch.zeros((B, M), dtype=torch.bool, device=self.device)
        goal_rewards = self.calculate_goal_reward(agent_positions, goal_positions, speeds, waypoint_reached)
        reward += goal_rewards
        # 判断是否到达目标：如果获得了目标奖励，说明到达了目标
        goal_reached = (goal_rewards > 0)


        # 计算碰撞惩罚
        collision_penalty = self.calculate_collision_penalty(agents_state, all_collisions)
        reward += collision_penalty

        # 计算离路惩罚
        offroad_penalty = self.calculate_offroad_penalty(offroad_mask)
        reward += offroad_penalty
        
        # 计算舒适度惩罚 (如果提供了加速度和加加速度信息)
        comfort_penalty = self.calculate_comfort_penalty(along, alat, along_jerk, alat_jerk)
        reward += comfort_penalty
        
        # 计算车道对齐奖励 (如果提供了车道角度信息)
        lane_alignment_reward = self.calculate_lane_alignment_reward(theta_f, speeds, dt)
        reward += lane_alignment_reward
        
        # 计算车道中心对齐奖励 (如果提供了横向位置信息)
        lane_center_reward = self.calculate_lane_center_reward(theta_f, d, dt)
        reward += lane_center_reward
        
        # 计算新的速度奖励 (如果提供了车道角度信息)
        new_velocity_reward = self.calculate_velocity_reward(theta_f, speeds, dt)
        reward += new_velocity_reward
        
        # 计算倒车惩罚
        reverse_penalty = self.calculate_reverse_penalty(speeds, dt)
        reward += reverse_penalty
        
        # 计算停止线违规惩罚 (如果提供了停止线违规信息)
        stop_line_penalty = self.calculate_stop_line_penalty(stop_line_violation)
        reward += stop_line_penalty
        
        # 计算时间步惩罚 (如果提供了加速度信息)
        timestep_penalty = self.calculate_timestep_penalty(speeds, along, alat, dt)
        reward += timestep_penalty

        self.last_reward_components = {
            'goal': goal_rewards,
            'collision': collision_penalty,
            'offroad': offroad_penalty,
            'comfort': comfort_penalty,
            'lane_align': lane_alignment_reward,
            'lane_center': lane_center_reward,
            'velocity': new_velocity_reward,
            'reverse': reverse_penalty,
            'stop_line': stop_line_penalty,
            'timestep': timestep_penalty,
        }
        
        return reward, goal_reached 


    # reward模块已经全部通过测试，上述代码保留。
