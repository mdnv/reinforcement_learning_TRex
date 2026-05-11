import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

st.set_page_config(page_title="Chrome Dino RL", page_icon="🦕", layout="wide")

st.markdown("""
<style>
    .stApp { background: #0f1117; color: #f7f7f7; }
    .block-container { padding-top: 2rem; max-width: 1450px; }
    div[data-testid="stMetric"], .stButton > button {
        background: #171a23; border: 1px solid #2b3140; border-radius: 8px; color: #f7f7f7;
    }
</style>
""", unsafe_allow_html=True)

st.title("🦕 Chrome Dino RL")

class DinoGame:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dino_y = 0
        self.dino_vy = 0
        self.is_jumping = False
        self.obstacles = []
        self.score = 0
        self.game_over = False
        self.gravity = 0.8
        self.jump_force = -12
        return self.get_state()
        
    def get_state(self):
        next_obstacle_dist = 600
        next_obstacle_height = 0
        next_obstacle_width = 0
        
        for obs in self.obstacles:
            if obs['x'] > 95:
                next_obstacle_dist = obs['x']
                next_obstacle_height = obs['height']
                next_obstacle_width = obs['width']
                break
                
        return [self.dino_y, self.dino_vy / 15, next_obstacle_dist / 600, 
                next_obstacle_height / 60, next_obstacle_width / 50]
    
    def step(self, action):
        if self.game_over:
            return self.get_state(), 0, True
        reward = 0.1
        old_state = self.get_state()
            
        if action == 1 and not self.is_jumping:
            self.dino_vy = self.jump_force
            self.is_jumping = True
            if old_state[2] > 0.34:
                reward -= 1
            
        self.dino_vy += self.gravity
        self.dino_y += self.dino_vy
        
        if self.dino_y >= 0:
            self.dino_y = 0
            self.dino_vy = 0
            self.is_jumping = False
            
        for obs in self.obstacles:
            obs['x'] -= 7
            
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -100]
        
        if len(self.obstacles) == 0 or self.obstacles[-1]['x'] < 300:
            if np.random.random() < 0.3:
                self.obstacles.append({
                    'x': 600,
                    'width': np.random.choice([20, 25, 30]),
                    'height': np.random.choice([30, 40, 50]),
                    'passed': False
                })
                
        dino_rect = {'x': 60, 'y': self.dino_y, 'w': 30, 'h': 40}
        for obs in self.obstacles:
            obs_rect = {'x': obs['x'], 'y': 0, 'w': obs['width'], 'h': obs['height']}
            if self.check_collision(dino_rect, obs_rect):
                self.game_over = True
                return self.get_state(), -5, True
            if not obs.get('passed') and obs['x'] + obs['width'] < 60:
                obs['passed'] = True
                reward += 10
                
        self.score += 1
        return self.get_state(), reward, False
        
    def check_collision(self, rect1, rect2):
        return (rect1['x'] < rect2['x'] + rect2['w'] and
                rect1['x'] + rect1['w'] > rect2['x'] and
                rect1['y'] < rect2['y'] + rect2['h'] and
                rect1['y'] + rect1['h'] > rect2['y'])

class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.2
        self.gamma = 0.9
        self.epsilon = 0.12
        
    def discretize(self, state):
        y, vy, dist, height, width = state
        return (
            0 if y == 0 else 1,
            np.digitize(vy, [-0.4, 0, 0.4]),
            np.digitize(dist, [0.18, 0.22, 0.26, 0.30, 0.36, 0.50]),
            np.digitize(height, [0.55, 0.75]),
            np.digitize(width, [0.45, 0.55])
        )
    
    def get_action(self, state):
        dino_y, _, obstacle_dist, obstacle_height, _ = state
        if obstacle_height == 0 or obstacle_dist > 0.36 or dino_y < 0:
            return 0

        state_key = self.discretize(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(2)
            if 0.20 < obstacle_dist < 0.32:
                self.q_table[state_key][1] = 1
            
        if np.random.random() < self.epsilon:
            return np.random.randint(2)
        return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state, done): # Функция обучения агента
        state_key = self.discretize(state)
        next_key = self.discretize(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(2)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(2)
            
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])
            
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action]) # Обновление Q-таблицы (сравнение)

def draw_game_frame(game):
    """Рисует HTML кадр игры"""
    canvas_width = 600
    canvas_height = 200
    sky_color = "#fbfcff"
    line_color = "#6b7280"
    dino_color = "#323744"
    cactus_color = "#23834a"
    score = f"{game.score:.1f}"
    
    html = f'''
    <div class="game-shell">
    <svg viewBox="0 0 {canvas_width} {canvas_height}" width="100%" height="100%" role="img" aria-label="Chrome Dino training game">
        <rect width="{canvas_width}" height="{canvas_height}" rx="14" fill="{sky_color}"/>
        <circle cx="500" cy="42" r="18" fill="#f2c14e" opacity=".55"/>
        <path d="M0 154 C70 150 118 158 190 153 S310 148 410 153 535 159 600 151 L600 200 L0 200 Z" fill="#eef2f7"/>
        <!-- Земля -->
        <line x1="0" y1="150" x2="{canvas_width}" y2="150" stroke="{line_color}" stroke-width="2"/>
        <g fill="{line_color}" opacity=".35">
            <rect x="25" y="166" width="34" height="3" rx="1.5"/>
            <rect x="176" y="176" width="48" height="3" rx="1.5"/>
            <rect x="372" y="166" width="42" height="3" rx="1.5"/>
            <rect x="508" y="181" width="58" height="3" rx="1.5"/>
        </g>
        
        <!-- Динозавр -->
        <g transform="translate(60, {150 - 40 + game.dino_y})">
            <!-- Тело -->
            <rect x="0" y="0" width="30" height="40" fill="{dino_color}" rx="4"/>
            <!-- Глаз -->
            <circle cx="22" cy="8" r="3" fill="white"/>
            <circle cx="23" cy="8" r="1.5" fill="{dino_color}"/>
            <!-- Ноги -->
            <rect x="5" y="35" width="6" height="10" fill="{dino_color}" rx="2"/>
            <rect x="19" y="35" width="6" height="10" fill="{dino_color}" rx="2"/>
            <!-- Хвост -->
            <polygon points="0,10 -12,5 -10,27" fill="{dino_color}"/>
        </g>
    '''
    
    # Препятствия
    for obs in game.obstacles:
        html += f'''
        <g transform="translate({obs['x']}, 0)">
            <rect x="0" y="{150 - obs['height']}" 
                width="{obs['width']}" height="{obs['height']}" 
                fill="{cactus_color}" rx="3"/>
            <!-- Шипы на кактусе -->
            <line x1="{obs['width']/2}" y1="{150 - obs['height']}" 
                x2="{obs['width']/2}" y2="{150 - obs['height'] - 10}" 
                stroke="{cactus_color}" stroke-width="2" stroke-linecap="round"/>
        </g>
        '''
    
    html += f'''
        <text x="560" y="26" text-anchor="end" fill="{dino_color}" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700">
            {score}
        </text>
    </svg>
    </div>
    '''
    return html

def render_game(game, placeholder):
    frame = f"""
    <style>
        html, body {{
            margin: 0;
            background: transparent;
        }}
        .game-shell {{
            width: 100%;
            height: 260px;
            border-radius: 8px;
            border: 1px solid #2a2f3d;
            background: #fbfcff;
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .55);
            overflow: hidden;
        }}
        svg {{
            display: block;
            image-rendering: pixelated;
        }}
    </style>
    {draw_game_frame(game)}
    """
    placeholder.empty()
    with placeholder:
        components.html(frame, height=268)

def plot_scores(scores, title="Прогресс обучения", save=False):
    fig, ax = plt.subplots(figsize=(8, 4.3), facecolor="white")
    if scores:
        window = min(10, len(scores))
        visible_scores = np.clip(scores, 0, np.percentile(scores, 90))
        smooth = np.convolve(visible_scores, np.ones(window) / window, mode="valid")
        y_max = max(30, np.max(smooth) * 1.25)
        ax.plot(visible_scores, color="#9bbcff", alpha=0.35, linewidth=1, label="Эпизод")
        ax.plot(range(window - 1, len(scores)), smooth, color="#ff4d4d", linewidth=2.5, label="Среднее")
    else:
        y_max = 30
    ax.set(title=title, xlabel="Эпизод", ylabel="Награда", xlim=(0, max(10, len(scores))), ylim=(0, y_max))
    ax.grid(True, alpha=0.25)
    if scores:
        ax.legend()
    if save:
        fig.savefig("training_graph.png", dpi=160, bbox_inches="tight")
    return fig

# Инициализация session state
if 'game' not in st.session_state:
    st.session_state.game = DinoGame()
    st.session_state.agent = QLearningAgent()
    st.session_state.scores_history = []
    st.session_state.current_episode = 0
    st.session_state.state = st.session_state.game.reset()
    st.session_state.running = False
    st.session_state.total_episodes = 0

# Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🎮 Игра")
    game_display = st.empty()
    
    # Показываем начальный кадр
    render_game(st.session_state.game, game_display)
    
    col1_1, col1_2, col1_3 = st.columns(3)
    with col1_1:
        if st.button("▶️ Запустить обучение"):
            st.session_state.running = True
            st.session_state.game = DinoGame()
            st.session_state.agent = QLearningAgent()
            st.session_state.scores_history = []
            st.session_state.current_episode = 0
            st.session_state.state = st.session_state.game.reset()
            st.session_state.total_episodes = 0
            
    with col1_2:
        if st.button("⏸️ Пауза"):
            st.session_state.running = False
            
    with col1_3:
        episodes_slider = st.slider("Эпизодов:", 10, 500, 50)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Эпизод", st.session_state.total_episodes)
    metric_col2.metric("Счет", f"{st.session_state.game.score:.1f}")
    metric_col3.metric("Состояний Q", len(st.session_state.agent.q_table))

with col2:
    st.subheader("📊 Обучение")
    chart_placeholder = st.empty()
    fig = plot_scores(st.session_state.scores_history)
    chart_placeholder.pyplot(fig)
    plt.close()

# Основной цикл обучения
if st.session_state.running:
    total_episodes = episodes_slider
    progress_bar = st.progress(0)
    
    for episode in range(total_episodes):
        if not st.session_state.running:
            break
            
        state = st.session_state.game.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 1000 and st.session_state.running:
            action = st.session_state.agent.get_action(state)
            next_state, reward, done = st.session_state.game.step(action)
            st.session_state.agent.learn(state, action, reward, next_state, done) # обучения агента
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            render_game(st.session_state.game, game_display)
            time.sleep(0.01)
            
            if done:
                break
        
        st.session_state.total_episodes += 1
        st.session_state.scores_history.append(max(0, episode_reward))
        st.session_state.agent.epsilon = max(0.02, st.session_state.agent.epsilon * 0.98)
        
        fig = plot_scores(
            st.session_state.scores_history,
            f"Прогресс обучения (эпизод {st.session_state.total_episodes})",
            save=True,
        )
        chart_placeholder.pyplot(fig)
        plt.close()
        
        progress_bar.progress((episode + 1) / total_episodes)
    
    st.session_state.running = False
    progress_bar.empty()
    st.success(f"✅ Обучение завершено! Всего эпизодов: {st.session_state.total_episodes}")

# Демонстрация после обучения
if not st.session_state.running and st.session_state.total_episodes > 0:
    if st.button("🎯 Показать лучшую игру"):
        game = DinoGame()
        state = game.reset()
        
        for _ in range(500):
            action = st.session_state.agent.get_action(state)
            next_state, reward, done = game.step(action)
            state = next_state
            
            render_game(game, game_display)
            time.sleep(0.3)
            
            if done:
                break
        
        st.write(f"Финальный счет: {game.score:.1f}")

    if st.button("💾 Сохранить график"):
        fig = plot_scores(st.session_state.scores_history, save=True)
        plt.close(fig)
        st.success("График сохранён: training_graph.png")
