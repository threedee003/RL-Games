import pygame
import random
import numpy as np

# Game settings
WIDTH, HEIGHT = 800, 600
# WIDTH, HEIGHT = 400, 300
FPS = 60
AGENT_RADIUS = 15
TARGET_RADIUS = 20
OBSTACLE_RADIUS = 25
AGENT_COLOR = (0, 0, 255)
TARGET_COLOR = (255, 0, 0)
OBSTACLE_COLOR = (0, 0, 0)

ACCEL_SCALE = 0.5
MAX_SPEED = 4.0
FRICTION = 0.95
TRAIL_LENGTH = 50

NUM_OBSTACLES = 15
OBSTACLE_MARGIN = 100

class Circle:
    def __init__(self, x, y, radius, color):
        self.pos = np.array([x, y], dtype=float)
        self.radius = radius
        self.color = color

    def draw(self, surface):
        pygame.draw.circle(surface, self.color, self.pos.astype(int), self.radius)

    def is_colliding(self, other_pos, other_radius):
        return np.linalg.norm(self.pos - other_pos) < (self.radius + other_radius)


class Agent(Circle):
    def __init__(self, x, y):
        super().__init__(x, y, AGENT_RADIUS, AGENT_COLOR)
        self.vel = np.zeros(2, dtype=float)
        self.trail = []

    def apply_action(self, accel):
        self.vel += accel
        speed = np.linalg.norm(self.vel)
        if speed > MAX_SPEED:
            self.vel = (self.vel / speed) * MAX_SPEED

    def move(self, obstacles):
        next_pos = self.pos + self.vel

        for obs in obstacles:
            if obs.is_colliding(next_pos, self.radius):
                self.vel *= -0.3  # Bounce back with damping
                return  # Skip movement this frame

        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)

        self.pos = next_pos
        self.vel *= FRICTION

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
            pygame.draw.circle(surface, (0, 100, 0), end, 5)


class GameEnv:
    def __init__(self, render_mode=True):
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Robot Reaching Environment with Obstacles")
            self.clock = pygame.time.Clock()
        self.running = True
        self.reset()


    def generate_obstacles(self):
        obstacles = []

        # Place 5 obstacles roughly between agent and target
        for i in range(5):
            t = (i + 1) / 6  # t in (0,1) to interpolate
            mid_pos = (1 - t) * self.agent.pos + t * self.target.pos
            offset = np.random.uniform(-60, 60, size=2)  # small random offset
            pos = mid_pos + offset
            pos = np.clip(pos, OBSTACLE_RADIUS, [WIDTH - OBSTACLE_RADIUS, HEIGHT - OBSTACLE_RADIUS])
            obstacles.append(Circle(pos[0], pos[1], OBSTACLE_RADIUS, OBSTACLE_COLOR))

        # Place remaining obstacles randomly
        while len(obstacles) < NUM_OBSTACLES:
            x = random.randint(OBSTACLE_RADIUS + OBSTACLE_MARGIN, WIDTH - OBSTACLE_RADIUS - OBSTACLE_MARGIN)
            y = random.randint(OBSTACLE_RADIUS + OBSTACLE_MARGIN, HEIGHT - OBSTACLE_RADIUS - OBSTACLE_MARGIN)
            obs = Circle(x, y, OBSTACLE_RADIUS, OBSTACLE_COLOR)

            # Avoid overlapping existing obstacles, agent, or target
            if any(obs.is_colliding(o.pos, o.radius + 10) for o in obstacles):
                continue
            if obs.is_colliding(self.agent.pos, self.agent.radius + OBSTACLE_MARGIN):
                continue
            if obs.is_colliding(self.target.pos, self.target.radius + OBSTACLE_MARGIN):
                continue

            obstacles.append(obs)

        return obstacles

    def reset(self, randomize_target=True):
        while True:
            self.agent = Agent(
                random.randint(AGENT_RADIUS, WIDTH - AGENT_RADIUS),
                random.randint(AGENT_RADIUS, HEIGHT - AGENT_RADIUS)
            )

            self.target_pos = np.array([
                random.randint(TARGET_RADIUS, WIDTH - TARGET_RADIUS),
                random.randint(TARGET_RADIUS, HEIGHT - TARGET_RADIUS)
            ]) if randomize_target else np.array([WIDTH // 3, HEIGHT // 3])

            self.target = Circle(self.target_pos[0], self.target_pos[1], TARGET_RADIUS, TARGET_COLOR)

            # Temporarily empty obstacle list to validate spawn points
            self.obstacles = []

            # Check agent and target are not overlapping
            if np.linalg.norm(self.agent.pos - self.target.pos) < (AGENT_RADIUS + TARGET_RADIUS + OBSTACLE_MARGIN):
                continue  # too close

            self.obstacles = self.generate_obstacles()

            # Ensure agent and target are not overlapping any obstacle
            if any(obs.is_colliding(self.agent.pos, AGENT_RADIUS + 5) for obs in self.obstacles):
                continue
            if any(obs.is_colliding(self.target.pos, TARGET_RADIUS + 5) for obs in self.obstacles):
                continue

            break

        return self.get_state()

    def get_state(self):
        return np.concatenate([self.agent.pos - self.target.pos, self.agent.vel]).astype(np.float32)

    def compute_reward(self):
        dist = np.linalg.norm(self.target.pos - self.agent.pos)
        return -dist

    def is_done(self):
        return np.linalg.norm(self.target.pos - self.agent.pos) < 10.0

    def step(self, action):
        action = np.array(action).astype(np.float32)
        assert action.shape == (4,), "Action must be a 4D vector: [up, down, left, right]"

        accel_y = action[1] - action[0]  # down - up
        accel_x = action[3] - action[2]  # right - left
        accel = np.array([accel_x, accel_y]) * ACCEL_SCALE

        self.agent.apply_action(accel)
        self.agent.move(self.obstacles)

        reward = self.compute_reward()
        done = self.is_done()
        info = {}
        return self.get_state(), reward, done, info

    def render(self):
        if not self.render_mode:
            return
        self.screen.fill((255, 255, 255))
        self.target.draw(self.screen)
        for obs in self.obstacles:
            obs.draw(self.screen)
        self.agent.draw_trail(self.screen)
        self.agent.draw(self.screen)
        self.agent.draw_velocity_vector(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        if self.render_mode:
            pygame.quit()

    def manual_control(self):
        keys = pygame.key.get_pressed()
        action = np.zeros(4)  # [up, down, left, right]
        if keys[pygame.K_w]: action[0] = 1
        if keys[pygame.K_s]: action[1] = 1
        if keys[pygame.K_a]: action[2] = 1
        if keys[pygame.K_d]: action[3] = 1
        return action


# Example manual run
if __name__ == "__main__":
    env = GameEnv(render_mode=True)
    state = env.reset()

    while env.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False

        action = env.manual_control()
        state, reward, done, _ = env.step(action)
        env.render()

        if done:
            print("Target Reached. Resetting...")
            env.reset()

    env.close()
