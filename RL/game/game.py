import pygame
import random
import numpy as np

# Game settings
WIDTH, HEIGHT = 800, 600
FPS = 60
AGENT_RADIUS = 15
TARGET_RADIUS = 20
AGENT_COLOR = (0, 0, 255)    # Blue
TARGET_COLOR = (255, 0, 0)   # Red

ACCEL = 0.5                 # Acceleration per key press
MAX_SPEED = 4.0             # Max speed magnitude
FRICTION = 0.95             # Friction factor
TRAIL_LENGTH = 50           # Number of positions to keep in trail

class Circle:
    def __init__(self, x, y, radius, color):
        self.pos = np.array([x, y], dtype=float)
        self.radius = radius
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.pos.astype(int), self.radius)

class Agent(Circle):
    def __init__(self, x, y):
        super().__init__(x, y, AGENT_RADIUS, AGENT_COLOR)
        self.vel = np.zeros(2, dtype=float)
        self.trail = []

    def handle_input(self, keys):
        if keys[pygame.K_w]: self.vel[1] -= ACCEL
        if keys[pygame.K_s]: self.vel[1] += ACCEL
        if keys[pygame.K_a]: self.vel[0] -= ACCEL
        if keys[pygame.K_d]: self.vel[0] += ACCEL

        # Clamp to max speed
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = (self.vel / speed) * MAX_SPEED

    def move(self):
        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

        self.pos += self.vel
        self.vel *= FRICTION  # Apply friction

        # Keep agent within screen bounds
        self.pos[0] = np.clip(self.pos[0], self.radius, WIDTH - self.radius)
        self.pos[1] = np.clip(self.pos[1], self.radius, HEIGHT - self.radius)

    def draw_trail(self, surface):
        for i, point in enumerate(self.trail):
            alpha = int(255 * (i + 1) / TRAIL_LENGTH)
            trail_color = (0, 0, 255, alpha)
            s = pygame.Surface((4, 4), pygame.SRCALPHA)
            pygame.draw.circle(s, trail_color, (2, 2), 2)
            surface.blit(s, point - 2)

    def draw_velocity_vector(self, surface):
        if np.linalg.norm(self.vel) > 0.5:
            start = self.pos.astype(int)
            end = (self.pos + self.vel * 10).astype(int)
            pygame.draw.line(surface, (0, 100, 0), start, end, 3)
            # pygame.draw.circle(surface, (0, 100, 0), end, 5)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Robot with Trail & Velocity Vector")
        self.clock = pygame.time.Clock()
        self.agent = Agent(WIDTH // 2, HEIGHT // 2)
        self.target_pos = [random.randint(TARGET_RADIUS, WIDTH - TARGET_RADIUS),
                           random.randint(TARGET_RADIUS, HEIGHT - TARGET_RADIUS)]
        self.target = Circle(
            self.target_pos[0],
            self.target_pos[1],
            TARGET_RADIUS,
            TARGET_COLOR
        )
        self.running = True

    def compute_reward(self):
        dist = np.linalg.norm(np.array(self.target_pos) - self.agent.pos)
        return -dist  # Negative distance as reward

    def check_success(self):
        return np.linalg.norm(np.array(self.target_pos) - self.agent.pos) < 6.0

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

            reward = self.compute_reward()
            print(f"Reward: {reward:.2f}")

        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        keys = pygame.key.get_pressed()
        self.agent.handle_input(keys)
        self.agent.move()

    def draw(self):
        self.screen.fill((255, 255, 255))  # White background
        self.target.draw(self.screen)
        self.agent.draw_trail(self.screen)
        self.agent.draw(self.screen)
        self.agent.draw_velocity_vector(self.screen)
        pygame.display.flip()

if __name__ == "__main__":
    Game().run()
