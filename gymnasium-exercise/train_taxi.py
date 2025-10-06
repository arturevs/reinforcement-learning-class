import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Hiperparâmetros ---
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
    x_axis = np.arange(1, len(history) + 1) * 1000
    plt.plot(x_axis, history)
    plt.title(f'Progresso do Treinamento - {agent_name} ({scenario_name})')
    plt.xlabel('Episódios')
    plt.ylabel('Taxa de Sucesso Média (%) nos Últimos 1000 Episódios')
    plt.grid(True)
    plt.savefig(f'taxi_{scenario_name.replace(" ", "_")}_{agent_name.replace(" ", "-")}_progresso.png')
    plt.close()

def plot_q_table_heatmap(q_table, agent_name, scenario_name):
    """Gera um heatmap simplificado da Q-Table."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(q_table, cmap='viridis', cbar_kws={'label': 'Q-Value'})
    plt.title(f'Heatmap da Q-Table - {agent_name} ({scenario_name})')
    plt.xlabel('Ações')
    plt.ylabel('Estados')
    plt.savefig(f'taxi_{scenario_name.replace(" ", "_")}_{agent_name.replace(" ", "-")}_politica.png')
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

    def evaluate_policy(self, num_episodes=10000):
        """Avalia o desempenho da política final sem exploração."""
        success_count = 0
        for _ in tqdm(range(num_episodes), desc="Avaliando política (greedy)"):
            state, _ = self.env.reset()
            for _ in range(self.params["max_steps"]):
                action = self.choose_action(state, greedy=True)
                next_state, reward, terminated, _, _ = self.env.step(action)
                if terminated:
                    if reward == 20: # Recompensa de sucesso no Taxi é +20
                        success_count += 1
                    break
                state = next_state
        return (success_count / num_episodes) * 100

    def train(self):
        raise NotImplementedError

class MonteCarloAgent(Agent):
    """Agente que utiliza o método de Controle Monte Carlo."""
    def __init__(self, env, params):
        super().__init__(env, params)
        # Linha corrigida para evitar o AttributeError
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
                episode_trajectory.append((state, action, reward))
                if terminated:
                    if reward == 20: success_chunk += 1
                    break
                state = next_state
            
            visited_state_actions = set()
            G = 0
            for state_g, action_g, reward_g in reversed(episode_trajectory):
                G = self.params["discount_factor"] * G + reward_g
                if (state_g, action_g) not in visited_state_actions:
                    self.returns_sum[state_g][action_g] += G
                    self.returns_count[state_g][action_g] += 1
                    self.q_table[state_g][action_g] = self.returns_sum[state_g][action_g] / self.returns_count[state_g][action_g]
                    visited_state_actions.add((state_g, action_g))

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
                
                best_next_action_value = np.max(self.q_table[next_state, :])
                td_target = reward + gamma * best_next_action_value
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += alpha * td_error

                if terminated:
                    if reward == 20: success_chunk += 1
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
                
                td_target = reward + gamma * self.q_table[next_state, next_action]
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += alpha * td_error

                if terminated:
                    if reward == 20: success_chunk += 1
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
        {"name": "Padrão", "params": {"fickle_passenger": False}},
        {"name": "Passageiro Indeciso", "params": {"fickle_passenger": True}},
    ]

    print("--- INICIANDO TESTES PARA O AMBIENTE TAXI-V3 ---\n")
    print(f"Parâmetros de Treinamento Utilizados:")
    for key, value in HYPERPARAMETERS.items(): print(f"  - {key}: {value}")

    for scenario in scenarios:
        scenario_name = scenario["name"]
        scenario_params = scenario["params"]
        
        print(f"\n{'='*60}")
        print(f"CENÁRIO: {scenario_name}")
        print(f"{'='*60}\n")
        
        env = gym.make('Taxi-v3', **scenario_params)

        agents = {
            "Q-Learning": QLearningAgent(env, HYPERPARAMETERS),
            "SARSA": SARSAAgent(env, HYPERPARAMETERS),
            "Monte-Carlo": MonteCarloAgent(env, HYPERPARAMETERS),
        }

        for agent_name, agent in agents.items():
            final_perf, history = agent.train()
            print(f"Resultado final (greedy) para {agent_name}: Taxa de Sucesso = {final_perf:.2f}%")
            
            plot_training_progress(history, agent_name, scenario_name)
            plot_q_table_heatmap(agent.q_table, agent_name, scenario_name)
            print(f"Gráficos para '{agent_name}' em '{scenario_name}' salvos com sucesso.")
        
        env.close()