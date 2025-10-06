import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Hiperparâmetros para o Frozen Lake 8x8 ---
HYPERPARAMETERS = {
    "total_episodes": 100_000,
    "max_steps": 250,
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon": 1.0,
    "min_epsilon": 0.01,
    "epsilon_decay_rate": 0.00002
}

def plot_training_progress(history, agent_name, scenario_name):
    """Gera e salva o gráfico de progresso do treinamento."""
    plt.figure(figsize=(12, 6))
    # Eixo X representando o final de cada bloco de 1000 episódios
    x_axis = np.arange(1, len(history) + 1) * 1000
    plt.plot(x_axis, history)
    plt.title(f'Progresso do Treinamento - {agent_name} ({scenario_name})')
    plt.xlabel('Episódios')
    plt.ylabel('Taxa de Sucesso Média (%) nos Últimos 1000 Episódios')
    plt.grid(True)
    plt.savefig(f'frozen_lake_{scenario_name}_{agent_name}_progresso.png')
    plt.close()

def plot_policy_heatmap(q_table, agent_name, scenario_name):
    """Gera e salva um heatmap da política e dos valores de estado."""
    map_size = 8
    value_function = np.max(q_table, axis=1).reshape(map_size, map_size)
    policy = np.argmax(q_table, axis=1).reshape(map_size, map_size)
    
    actions_map = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    policy_symbols = np.vectorize(actions_map.get)(policy)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(value_function, annot=policy_symbols, fmt='', cmap='viridis', cbar=True, ax=ax,
                linewidths=.5, linecolor='black')
    ax.set_title(f'Política Óptima e Valor de Estado - {agent_name} ({scenario_name})')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.savefig(f'frozen_lake_{scenario_name}_{agent_name}_politica.png')
    plt.close()

class Agent:
    """Classe base para os agentes de Aprendizado por Reforço."""
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.epsilon = params["epsilon"]

    def choose_action(self, state, greedy=False):
        """Escolhe uma ação. Se greedy=True, sempre escolhe a melhor ação."""
        if greedy:
            return np.argmax(self.q_table[state, :])
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state, :])

    def update_epsilon(self):
        """Aplica o decaimento constante ao epsilon."""
        self.epsilon = max(self.params["min_epsilon"], self.epsilon - self.params["epsilon_decay_rate"])

    def get_custom_reward(self, terminated, original_reward):
        """Aplica a função de reforço customizada."""
        if not terminated: return -0.01
        elif terminated and original_reward == 1.0: return 1.0
        else: return -0.5

    def evaluate_policy(self, num_episodes=10000):
        """Avalia o desempenho da política final sem exploração."""
        success_count = 0
        for _ in tqdm(range(num_episodes), desc=f"Avaliando política (greedy)"):
            state, _ = self.env.reset()
            for _ in range(self.params["max_steps"]):
                action = self.choose_action(state, greedy=True)
                next_state, reward, terminated, _, _ = self.env.step(action)
                if terminated:
                    if reward == 1.0:
                        success_count += 1
                    break
                state = next_state
        return (success_count / num_episodes) * 100

    def train(self):
        raise NotImplementedError

class MonteCarloAgent(Agent):
    """Agente Monte Carlo."""
    def __init__(self, env, params):
        super().__init__(env, params)
        # A linha abaixo foi adicionada para corrigir o erro
        self.action_space_size = env.action_space.n
        self.returns_sum = defaultdict(lambda: np.zeros(self.action_space_size))
        self.returns_count = defaultdict(lambda: np.zeros(self.action_space_size))

    def train(self):
        history = []
        success_chunk = 0
        log_interval = 1000

        for i in tqdm(range(self.params["total_episodes"]), desc="Treinando com Monte Carlo"):
            episode_trajectory = []
            state, _ = self.env.reset()
            
            for _ in range(self.params["max_steps"]):
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)
                custom_reward = self.get_custom_reward(terminated, reward)
                episode_trajectory.append((state, action, custom_reward))
                
                if terminated:
                    if reward == 1.0: success_chunk += 1
                    break
                state = next_state
            
            visited_state_actions = set()
            G = 0
            for state, action, reward_g in reversed(episode_trajectory):
                G = self.params["discount_factor"] * G + reward_g
                if (state, action) not in visited_state_actions:
                    self.returns_sum[state][action] += G
                    self.returns_count[state][action] += 1
                    self.q_table[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
                    visited_state_actions.add((state, action))

            if (i + 1) % log_interval == 0:
                history.append((success_chunk / log_interval) * 100)
                success_chunk = 0
            
            self.update_epsilon()

        final_performance = self.evaluate_policy()
        return final_performance, history

class QLearningAgent(Agent):
    """Agente que utiliza o algoritmo Q-Learning."""
    def train(self):
        history = []
        success_chunk = 0
        log_interval = 1000
        alpha = self.params["learning_rate"]
        gamma = self.params["discount_factor"]

        for i in tqdm(range(self.params["total_episodes"]), desc="Treinando com Q-Learning"):
            state, _ = self.env.reset()
            for _ in range(self.params["max_steps"]):
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)
                custom_reward = self.get_custom_reward(terminated, reward)

                best_next_action_value = np.max(self.q_table[next_state, :])
                td_target = custom_reward + gamma * best_next_action_value
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += alpha * td_error

                if terminated:
                    if reward == 1.0: success_chunk += 1
                    break
                state = next_state
            
            if (i + 1) % log_interval == 0:
                history.append((success_chunk / log_interval) * 100)
                success_chunk = 0
                
            self.update_epsilon()
            
        final_performance = self.evaluate_policy()
        return final_performance, history

class SARSAAgent(Agent):
    """Agente que utiliza o algoritmo SARSA."""
    def train(self):
        history = []
        success_chunk = 0
        log_interval = 1000
        alpha = self.params["learning_rate"]
        gamma = self.params["discount_factor"]

        for i in tqdm(range(self.params["total_episodes"]), desc="Treinando com SARSA"):
            state, _ = self.env.reset()
            action = self.choose_action(state)

            for _ in range(self.params["max_steps"]):
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_action = self.choose_action(next_state)
                custom_reward = self.get_custom_reward(terminated, reward)
                
                td_target = custom_reward + gamma * self.q_table[next_state, next_action]
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += alpha * td_error

                if terminated:
                    if reward == 1.0: success_chunk += 1
                    break
                state = next_state
                action = next_action

            if (i + 1) % log_interval == 0:
                history.append((success_chunk / log_interval) * 100)
                success_chunk = 0

            self.update_epsilon()
            
        final_performance = self.evaluate_policy()
        return final_performance, history

if __name__ == "__main__":
    scenarios = [
        ("Estocastico", True),
        ("Deterministico", False),
    ]

    for name, is_slippery in scenarios:
        print(f"\n--- INICIANDO TESTES PARA O CENÁRIO: {name} ---\n")
        
        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=is_slippery)

        agents = {
            "Q-Learning": QLearningAgent(env, HYPERPARAMETERS),
            "SARSA": SARSAAgent(env, HYPERPARAMETERS),
            "Monte-Carlo": MonteCarloAgent(env, HYPERPARAMETERS),
        }
        
        print(f"Parâmetros utilizados:")
        for key, value in HYPERPARAMETERS.items(): print(f"  - {key}: {value}")
        print("-" * 50)

        for agent_name, agent in agents.items():
            final_perf, history = agent.train()
            print(f"Resultado final (greedy) para {agent_name}: Taxa de Sucesso = {final_perf:.2f}%")
            
            # Gerar e salvar gráficos
            plot_training_progress(history, agent_name, name)
            plot_policy_heatmap(agent.q_table, agent_name, name)
        
        print("-" * 50)
        env.close()