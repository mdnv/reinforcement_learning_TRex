import numpy as np

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
        self.ground_y = 0
        self.gravity = 0.8
        self.jump_force = -12
        return self.get_state()
        
    def get_state(self):
        # Находим ближайшее препятствие
        next_obstacle_dist = 400
        next_obstacle_height = 0
        next_obstacle_width = 0
        
        for obs in self.obstacles:
            if obs['x'] > -50:
                next_obstacle_dist = obs['x']
                next_obstacle_height = obs['height']
                next_obstacle_width = obs['width']
                break
                
        return [self.dino_y, self.dino_vy, next_obstacle_dist, 
                next_obstacle_height, next_obstacle_width]
    
    def step(self, action):
        # action: 0 - ничего, 1 - прыжок
        if self.game_over:
            return self.get_state(), 0, True
            
        # Прыжок
        if action == 1 and not self.is_jumping:
            self.dino_vy = self.jump_force
            self.is_jumping = True
            
        # Физика
        self.dino_vy += self.gravity
        self.dino_y += self.dino_vy
        
        if self.dino_y >= self.ground_y:
            self.dino_y = self.ground_y
            self.dino_vy = 0
            self.is_jumping = False
            
        # Двигаем препятствия
        for obs in self.obstacles:
            obs['x'] -= 10
            
        # Удаляем старые препятствия
        self.obstacles = [obs for obs in self.obstacles if obs['x'] > -100]
        
        # Создаем новые препятствия
        if len(self.obstacles) == 0 or self.obstacles[-1]['x'] < 300:
            if np.random.random() < 0.3:
                self.obstacles.append({
                    'x': 600,
                    'width': np.random.choice([20, 30, 40]),
                    'height': np.random.choice([30, 40, 50])
                })
                
        # Проверка столкновений
        dino_rect = {'x': 60, 'y': self.dino_y, 'w': 30, 'h': 40}
        for obs in self.obstacles:
            obs_rect = {'x': obs['x'], 'y': 0, 'w': obs['width'], 'h': obs['height']}
            if self.check_collision(dino_rect, obs_rect):
                self.game_over = True
                return self.get_state(), -10, True
                
        self.score += 0.1
        return self.get_state(), 0.1, False
        
    def check_collision(self, rect1, rect2):
        return (rect1['x'] < rect2['x'] + rect2['w'] and
                rect1['x'] + rect1['w'] > rect2['x'] and
                rect1['y'] < rect2['y'] + rect2['h'] and
                rect1['y'] + rect1['h'] > rect2['y'])