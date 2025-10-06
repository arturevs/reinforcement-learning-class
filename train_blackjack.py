import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Hiperparâmetros ---
HYPERPARAMETERS = {
    "total_episodes": 1_000_000,
    "learning_rate": 0.1,      # Alpha
    "discount_factor": 0.95,   # Gamma
    "epsilon": 1.0,
    "min_epsilon": 0.1,
    "epsilon_decay_rate": (1.0 - 0.1) / 1_000_000 # Decaimento linear
}

def plot_training_progress(history, agent_name):
    """Gera e salva o gráfico de progresso do treinamento para o Blackjack."""
    history = np.array(history)
    wins = history[:, 0]
    losses = history[:, 1]
    draws = history[:, 2]
    
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(1, len(wins) + 1) * 10000 # Intervalo de log
    plt.plot(x_axis, wins, label='Vitórias %', color='g')
    plt.plot(x_axis, losses, label='Derrotas %', color='r')
    plt.plot(x_axis, draws, label='Empates %', color='b')
    plt.title(f'Progresso do Treinamento - {agent_name}')
    plt.xlabel('Episódios')
    plt.ylabel('Resultado Médio (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'blackjack_{agent_name.replace(" ", "-")}_progresso.png')
    plt.close()

def plot_blackjack_policy(q_table, agent_name):
    """Gera heatmaps para a política do Blackjack."""
    def get_policy_grid(usable_ace):
        # Soma do jogador (12-21) vs. Carta do Dealer (Ás=1, 2-10)
        policy_grid = np.zeros((10, 10)) 
        for player_sum in range(12, 22):
            for dealer_card in range(1, 11):
                state = (player_sum, dealer_card, usable_ace)
                # Se o estado não foi visitado, defaultdict retorna 0s, argmax será 0 (Ficar)
                action = np.argmax(q_table[state])
                policy_grid[player_sum - 12, dealer_card - 1] = action
        return policy_grid

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Estratégia Final - {agent_name} (0=Ficar, 1=Pedir)', fontsize=16)

    # Plot para "Sem Ás Usável"
    sns.heatmap(get_policy_grid(False), cmap='coolwarm', ax=ax[0],
                yticklabels=range(12, 22), xticklabels=range(1, 11), annot=True, cbar=False)
    ax[0].set_title('Política - Sem Ás Usável')
    ax[0].set_xlabel('Carta Visível do Dealer')
    ax[0].set_ylabel('Soma do Jogador')
    
    # Plot para "Com Ás Usável"
    sns.heatmap(get_policy_grid(True), cmap='coolwarm', ax=ax[1],
                yticklabels=range(12, 22), xticklabels=range(1, 11), annot=True, cbar=False)
    ax[1].set_title('Política - Com Ás Usável')
    ax[1].set_xlabel('Carta Visível do Dealer')
    ax[1].set_ylabel('') # Remove o label y para limpar a imagem
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'blackjack_{agent_name.replace(" ", "-")}_politica.png')
    plt.close()

class Agent:
    """Classe base para os agentes de Aprendizado por Reforço."""
    def __init__(self, env, params):
        self.env = env
        self.params = params
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.epsilon = params["epsilon"]

    def choose_action(self, state, greedy=False):
        """Escolhe uma ação. Se greedy=True, sempre escolhe a melhor ação."""
        if greedy:
            return np.argmax(self.q_table[state])
        
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_epsilon(self):
        """Aplica o decaimento linear ao epsilon."""
        self.epsilon = max(self.params["min_epsilon"], self.epsilon - self.params["epsilon_decay_rate"])

    def evaluate_policy(self, num_episodes=10000):
        """Avalia o desempenho da política final sem exploração."""
        stats = {'wins': 0, 'losses': 0, 'draws': 0}
        for _ in tqdm(range(num_episodes), desc=f"Avaliando política (greedy)"):
            state, _ = self.env.reset()
            terminated = False
            while not terminated:
                action = self.choose_action(state, greedy=True)
                next_state, reward, terminated, _, _ = self.env.step(action)
                state = next_state
            if reward > 0: stats['wins'] += 1
            elif reward < 0: stats['losses'] += 1
            else: stats['draws'] += 1
        
        total = sum(stats.values())
        return (stats['wins']/total*100, stats['losses']/total*100, stats['draws']/total*100)

    def train(self):
        raise NotImplementedError

class MonteCarloAgent(Agent):
    """Agente que utiliza o método de Controle Monte Carlo."""
    def __init__(self, env, params):
        super().__init__(env, params)
        # A linha abaixo foi adicionada para corrigir o erro
        self.action_space_size = env.action_space.n 
        self.returns_sum = defaultdict(lambda: np.zeros(self.action_space_size))
        self.returns_count = defaultdict(lambda: np.zeros(self.action_space_size))
        
    def train(self):
        history = []
        stats_chunk = {'wins': 0, 'losses': 0, 'draws': 0}
        log_interval = 10000

        for i in tqdm(range(self.params["total_episodes"]), desc="Treinando com Monte Carlo"):
            episode_trajectory = []
            state, _ = self.env.reset()
            terminated = False

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)
                episode_trajectory.append((state, action, reward))
                state = next_state
            
            if reward > 0: stats_chunk['wins'] += 1
            elif reward < 0: stats_chunk['losses'] += 1
            else: stats_chunk['draws'] += 1

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
                total = sum(stats_chunk.values())
                history.append((
                    stats_chunk['wins'] / total * 100,
                    stats_chunk['losses'] / total * 100,
                    stats_chunk['draws'] / total * 100
                ))
                stats_chunk = {'wins': 0, 'losses': 0, 'draws': 0}

            self.update_epsilon()
        
        final_performance = self.evaluate_policy()
        return final_performance, history

class QLearningAgent(Agent):
    """Agente que utiliza o algoritmo Q-Learning."""
    def train(self):
        history = []
        stats_chunk = {'wins': 0, 'losses': 0, 'draws': 0}
        log_interval = 10000
        alpha = self.params["learning_rate"]
        gamma = self.params["discount_factor"]

        for i in tqdm(range(self.params["total_episodes"]), desc="Treinando com Q-Learning"):
            state, _ = self.env.reset()
            terminated = False

            while not terminated:
                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)

                best_next_action_value = np.max(self.q_table[next_state])
                td_target = reward + gamma * best_next_action_value
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += alpha * td_error
                state = next_state
            
            if reward > 0: stats_chunk['wins'] += 1
            elif reward < 0: stats_chunk['losses'] += 1
            else: stats_chunk['draws'] += 1
                
            if (i + 1) % log_interval == 0:
                total = sum(stats_chunk.values())
                history.append((
                    stats_chunk['wins'] / total * 100,
                    stats_chunk['losses'] / total * 100,
                    stats_chunk['draws'] / total * 100
                ))
                stats_chunk = {'wins': 0, 'losses': 0, 'draws': 0}

            self.update_epsilon()
            
        final_performance = self.evaluate_policy()
        return final_performance, history

class SARSAAgent(Agent):
    """Agente que utiliza o algoritmo SARSA."""
    def train(self):
        history = []
        stats_chunk = {'wins': 0, 'losses': 0, 'draws': 0}
        log_interval = 10000
        alpha = self.params["learning_rate"]
        gamma = self.params["discount_factor"]

        for i in tqdm(range(self.params["total_episodes"]), desc="Treinando com SARSA"):
            state, _ = self.env.reset()
            action = self.choose_action(state)

            while True:
                next_state, reward, terminated, _, _ = self.env.step(action)
                next_action = self.choose_action(next_state)

                td_target = reward + gamma * self.q_table[next_state][next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += alpha * td_error

                state = next_state
                action = next_action
                
                if terminated:
                    break

            if reward > 0: stats_chunk['wins'] += 1
            elif reward < 0: stats_chunk['losses'] += 1
            else: stats_chunk['draws'] += 1

            if (i + 1) % log_interval == 0:
                total = sum(stats_chunk.values())
                history.append((
                    stats_chunk['wins'] / total * 100,
                    stats_chunk['losses'] / total * 100,
                    stats_chunk['draws'] / total * 100
                ))
                stats_chunk = {'wins': 0, 'losses': 0, 'draws': 0}

            self.update_epsilon()
            
        final_performance = self.evaluate_policy()
        return final_performance, history

if __name__ == "__main__":
    env = gym.make("Blackjack-v1")
    
    agents = {
        "Monte Carlo": MonteCarloAgent(env, HYPERPARAMETERS),
        "Q-Learning": QLearningAgent(env, HYPERPARAMETERS),
        "SARSA": SARSAAgent(env, HYPERPARAMETERS)
    }

    print("Iniciando treinamento consecutivo dos agentes no ambiente Blackjack...\n")
    print(f"Parâmetros utilizados:")
    for key, value in HYPERPARAMETERS.items(): print(f"  - {key}: {value}")
    print("-" * 50)
    
    for name, agent in agents.items():
        (wins, losses, draws), history = agent.train()
        print(f"\nResultados Finais (Greedy) para o agente {name}:")
        print(f"  - Vitórias: {wins:.2f}%")
        print(f"  - Derrotas: {losses:.2f}%")
        print(f"  - Empates:  {draws:.2f}%")
        
        plot_training_progress(history, name)
        plot_blackjack_policy(agent.q_table, name)
        print(f"Gráficos para '{name}' salvos com sucesso.")
        print("-" * 50)

    env.close()