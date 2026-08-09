from copy import deepcopy
import numpy as np
from gym import spaces
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
from offpolicy.envs.env_wrappers import DummyVecEnv


class MobileProductionEnv:
    """
    小米手机生产线多智能体环境
    三个agent顺序处理任务：SMT -> Assembly -> Test
    - Task按泊松分布到达SMT环节（同质化任务）
    - 每个环节有独立buffer容量
    - Agent可选择不同处理速率（对应不同能耗），每个agent的速率配置不同
    - 处理速率 = 每分钟能完成的整数任务数
    - 完全合作：共享奖励函数
    - 全局观测：所有agent的buffer容量 + 使用情况
    """
    def __init__(self, args=None, task_arrival_rate=3, episode_length=250):
        self.num_agents = 3
        self.agent_names = ["SMT", "Assembly", "Test"]

        self.episode_length = args.episode_length if args else episode_length
    
        # 任务参数
        self.task_arrival_rate = args.task_arrival_rate if args else task_arrival_rate  # 泊松到达率λ

        # 处理速率与能耗配置（每个agent独立）
        self.buffer_max_sizes = np.array([10, 8, 6], dtype=np.int32)  # SMT、组装、测试
        self.processing_rates = np.array([
            [1, 3, 5],    # SMT: 低速/中速/高速
            [1, 2, 4],    # Assembly: 相对慢一些
            [2, 3, 5]     # Test: 相对快一些
        ], dtype=np.float32)

        self.energy_costs = np.array([
            [1.0, 4.0, 9.0],   # SMT能耗
            [0.8, 2.5, 5.5],   # Assembly能耗
            [1.5, 4.0, 9.0]    # Test能耗
        ], dtype=np.float32)

        # 奖励参数
        self.task_completion_factor = 5.0  # 每完成一个任务的奖励
        self.buffer_overflow_factor = 10.0  # Buffer溢出惩罚（每次）
        self.energy_penalty_factor = 0.2  # 能耗惩罚系数

        # 动作空间：离散选择处理速率 {0:低速, 1:中速, 2:高速}
        self.action_space = [spaces.Discrete(3) for _ in range(self.num_agents)]

        # 观测内容：[所有agent的buffer容量, 所有agent的buffer当前使用量]
        obs_dim = self.num_agents * 2
        self.observation_space = [spaces.Box(
            low=0.0, high=10.0, shape=(obs_dim,), dtype=np.float32)] * self.num_agents

        # 共享观测空间（用于集中式critic）
        self.share_observation_space = [spaces.Box(
            low=0.0, high=10.0, shape=(obs_dim * self.num_agents,), dtype=np.float32)] * self.num_agents

        # 内部状态
        self.current_step = 0
        self.buffers = np.zeros(self.num_agents, dtype=np.int32)

        # 统计信息
        self.total_tasks_completed = 0
        self.total_buffer_overflows = 0
        self.total_tasks_arrived = 0  # 记录总到达任务数


    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)


    def reset(self):
        """重置环境"""
        self.current_step = 0
        
        # 重置内部状态
        self.buffers = np.zeros(self.num_agents, dtype=np.int32)
        
        # 统计信息
        self.total_tasks_completed = 0
        self.total_buffer_overflows = 0
        self.total_tasks_arrived = 0

        return self._get_observations()


    def _get_observations(self):
        """生成所有agent的观测（全局相同）"""
        # 组合成全局观测
        global_obs = np.concatenate([
            self.buffer_max_sizes,
            self.buffers,
        ]).astype(np.float32)

        # 所有agent获得相同的观测
        obs_n = [global_obs.copy() for _ in range(self.num_agents)]

        return obs_n


    def _generate_new_task_arrivals(self):
        """按泊松分布生成新任务到达数量"""
        # 泊松分布：返回整数到达数量
        arrivals = np.random.poisson(self.task_arrival_rate)
        if arrivals > 0:
            self.total_tasks_arrived += arrivals
        return arrivals


    def _format_actions(self, action_n):
        action_n = np.array(action_n, dtype=np.int32)
        # 检查 shape
        if np.prod(action_n.shape) == self.num_agents * 3:  # one-hot 编码
            action_n = action_n.reshape(self.num_agents, 3)
            action_array = np.argmax(action_n, axis=-1).flatten()
        elif np.prod(action_n.shape) == self.num_agents:  # 直接索引
            action_array = action_n.flatten()
        return action_array


    def step(self, action_n):
        """执行一步仿真"""
        self.current_step += 1

        # 解析动作
        action_array = self._format_actions(action_n)

        # 1. 任务到达（SMT环节）
        new_arrivals = self._generate_new_task_arrivals()
        self.buffers[0] += new_arrivals  # SMT环节的buffer增加新到达任务
        
        # 2. 各agent处理任务
        stage_completion = np.zeros(self.num_agents)
        energy_costs = self.energy_costs[:, action_array]

        for i in range(self.num_agents):
            # 获取当前动作对应的处理速率（每分钟完成的任务数）
            action = action_array[i]
            processing_rate = self.processing_rates[i, action]

            # 处理任务（不能超过buffer中的任务数）
            tasks_to_process = int(min(self.buffers[i], processing_rate))

            if tasks_to_process > 0:
                stage_completion[i] = tasks_to_process
                # 从当前buffer移除任务
                self.buffers[i] -= tasks_to_process

                # 如果是最后一个环节，任务完成
                if i == self.num_agents - 1:
                    self.total_tasks_completed += tasks_to_process
                else:
                    # 转移到下一环节
                    next_i = i + 1
                    self.buffers[next_i] += tasks_to_process

        # 3. 处理buffer溢出惩罚
        buffer_overflows = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            if self.buffers[i] > self.buffer_max_sizes[i]:
                buffer_overflows[i] = self.buffers[i] - self.buffer_max_sizes[i]
                self.buffers[i] = self.buffer_max_sizes[i]
        self.total_buffer_overflows += np.sum(buffer_overflows)

        # 4. 计算总奖励（完全合作）
        # MARK: gt reward
        task_completion_reward = self.task_completion_factor * stage_completion[-1]
        overflow_penalty = self.buffer_overflow_factor * np.sum(buffer_overflows)
        energy_penalty = self.energy_penalty_factor * np.sum(energy_costs)
        total_reward = (task_completion_reward - overflow_penalty - energy_penalty) * 0.1
        # 
        # 计算个性化奖励（每个 agent 独立）
        # MARK: personalized reward
        personalized_rewards = []
        for i in range(self.num_agents):
            if i == self.num_agents - 1:
                # 最后一个环节, 只考虑自己的完成情况和溢出
                task_contrib = self.task_completion_factor * stage_completion[i]
                overflow_contrib = 0.5 * self.buffer_overflow_factor * buffer_overflows[i]
            else:
                # 非最后环节, 考虑自己和下游环节的完成情况和溢出
                task_contrib = 0.5 * self.task_completion_factor * (stage_completion[i] + stage_completion[-1])
                overflow_contrib = 0.5 * self.buffer_overflow_factor * (buffer_overflows[i] + buffer_overflows[i + 1])
                if i == 0:
                    overflow_contrib += 0.5 * self.buffer_overflow_factor * buffer_overflows[i]

            energy_contrib = self.energy_penalty_factor * energy_costs[i][action_array[i]]
            personalized_reward = (task_contrib - energy_contrib - overflow_contrib) * 0.1
            personalized_rewards.append(personalized_reward)

        # 5. 生成返回数据
        reward_n = [[total_reward] for _ in range(self.num_agents)]
        obs_n = self._get_observations()
        done_n = [[self.current_step >= self.episode_length] for _ in range(self.num_agents)]

        # 6. 生成info
        info_n = []
        for i in range(self.num_agents):
            info_n.append({
                'individual_reward': total_reward,
                'personalized_reward': personalized_rewards[i],
                # 把能想到的都塞进 info 了
                'stage_name': self.agent_names[i],
                'action_taken': action_array[i],
                'buffer_size': int(self.buffer_max_sizes[i]),
                'cur_buffer_size': int(self.buffers[i]),
                'energy_cost': energy_costs[i],
                'tasks_processed': int(stage_completion[i]),
                'buffer_overflow': int(buffer_overflows[i]),
                'tasks_completed': int(stage_completion[-1]),
                'total_tasks_arrived': self.total_tasks_arrived,
                'total_tasks_completed': self.total_tasks_completed,
                'total_buffer_overflows': self.total_buffer_overflows,
            })

        return obs_n, reward_n, done_n, info_n


    def close(self):
        """关闭环境"""
        pass



if __name__ == "__main__":
    # 测试环境
    env = MobileProductionEnv()
    env.seed(42)
    
    # 创建向量化环境
    vecenv = DummyVecEnv([lambda: env])
    
    # 重置环境
    obs_n = vecenv.reset()
    print("Initial Observations (Agent 0):", obs_n[0, 0])
    
    # 执行测试步骤
    for step in range(20):
        # 随机选择动作（0:低速, 1:中速, 2:高速）
        actions = [[np.random.randint(3) for _ in range(env.num_agents)]]
        
        obs_n, reward_n, done_n, info_n = vecenv.step(actions)
        print(f"\nStep {step + 1}:")
        print("Actions:", actions)
        print("Rewards:", reward_n)
        print("Buffers:", env.buffers)
        print("Info 1:", {k: v for k, v in info_n[0][0].items() if k in ['stage_name', 'cur_buffer_size', 'tasks_processed', 'buffer_overflow']})
        print("Info 2:", {k: v for k, v in info_n[0][1].items() if k in ['stage_name', 'cur_buffer_size', 'tasks_processed', 'buffer_overflow']})
        print("Info 3:", {k: v for k, v in info_n[0][2].items() if k in ['stage_name', 'cur_buffer_size', 'tasks_processed', 'buffer_overflow']})
        
        # 检查是否结束
        if all(done_n[0]):
            break
    
    vecenv.close()
    print(f"\nTotal tasks completed: {env.total_tasks_completed}")
    print(f"Total buffer overflows: {env.total_buffer_overflows}")
    print(f"Total tasks arrived: {env.total_tasks_arrived}")

