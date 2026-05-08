'''
两台老虎机，左右每台有1个拉杆，奖励满足正态分布，分别是N(500,50)和N(550,100)
玩家每拉一次拉杆，老虎机会根据奖励分布生成一个奖励值，玩家需要选择拉哪一台老虎机的拉杆，使得奖励值最大化。
从初始状态Q(left)=Q(right)=998开始，每次拉杆后，玩家可以获得奖励值，并根据奖励值更新老虎机的奖励分布。
'''
import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    def __init__(self):
        self.arms = {
            0:{'mean': 500, 'std': 50},
            1:{'mean': 550, 'std': 100}
        }

    def step(self, action):
        return np.random.normal(self.arms[action]['mean'], self.arms[action]['std'])

class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.1, initial_q=998, k_arms=2):
        self.epsilon = epsilon
        self.k = k_arms
        # 创建长度为k的数组，所有元素初始化为initial_q（998），用于保存每个动作的期望价值 
        self.q_values = np.full(self.k, initial_q)
        # 创建长度为k的数组，所有元素初始化为0，用于记录每个动作被选了多少次
        self.action_counts = np.zeros(self.k)

    def choose_action(self):
        '''
        以小于0.1的概率随机选择，否则选q更大的
        '''
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        else:
            return np.argmax(self.q_values)

    def update_q_value(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (1.0/n) * (reward - self.q_values[action])
        
if __name__ == "__main__":
    # 先定种子数
    np.random.seed(42)
    env = BanditEnvironment()
    agent = EpsilonGreedyAgent(epsilon=0.1, initial_q=998, k_arms=2)
    # 用于记录数据 方便作图
    steps = 1000
    rewards_history = []

    for i in range(steps):
        # 先观察状态，作出动作
        action = agent.choose_action()
        # 再根据动作，给出奖励
        reward = env.step(action)
        # 根据奖励，更新q预期
        agent.update_q_value(action, reward)
        rewards_history.append(reward)
    
    # 可视化
    print(f"1000次试验后：")
    print(f"左侧机器(Left) 被拉动次数: {agent.action_counts[0]}")
    print(f"右侧机器(Right) 被拉动次数: {agent.action_counts[1]}")
    print(f"最终 Q 值估算: Left: {agent.q_values[0]:.2f}, Right: {agent.q_values[1]:.2f}")

    # 简单画一下平均奖励的收敛曲线
    cumulative_average = np.cumsum(rewards_history) / (np.arange(steps) + 1)
    plt.plot(cumulative_average)
    plt.axhline(y=550, color='r', linestyle='--', label='Optimal Expected Reward (550)')
    plt.title('Epsilon-Greedy (Init Q=998, e=0.1)')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()