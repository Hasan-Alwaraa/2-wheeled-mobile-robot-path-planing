import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
import random
import math

# Grid and obstacles
GRID_SIZE = 1
width, height = 20, 20
cols, rows = int(width / GRID_SIZE), int(height / GRID_SIZE)
grid = np.zeros((rows, cols))

def generate_random_obstacles(num=7, max_w=6, max_h=6):
    obstacles = []
    for _ in range(num):
        w = random.randint(2, max_w)
        h = random.randint(2, max_h)
        ox = random.randint(0, cols - w - 1)
        oy = random.randint(0, rows - h - 1)
        obstacles.append((ox, oy, w, h))
    return obstacles

obstacles = generate_random_obstacles()

grid.fill(0)
for ox, oy, w, h in obstacles:
    grid[int(oy):int(oy + h), int(ox):int(ox + w)] = 1

# Inflate obstacles by one cell to account for robot radius
inflated = grid.copy()
for i in range(rows):
    for j in range(cols):
        if grid[i, j] == 1:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        inflated[ni, nj] = 1

# RRT* Parameters
start = np.array([1.0, 1.0])
goal = np.array([18.0, 18.0])
max_iters = 5000
step_size = 0.5
goal_radius = 1.0
neighbor_radius = 2.0

class Node:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.parent = None
        self.cost = 0.0

# Utility functions
def obstacle_free(p1, p2):
    steps = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 10)
    if steps == 0:
        return True
    for i in range(steps + 1):
        t = i / steps
        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])
        ix, iy = int(y), int(x)
        if ix < 0 or ix >= rows or iy < 0 or iy >= cols or inflated[ix][iy] == 1:
            return False
    return True

def sample():
    return np.array([random.uniform(0, cols), random.uniform(0, rows)])

def nearest(nodes, pt):
    return min(nodes, key=lambda n: np.linalg.norm(n.pos - pt))

def steer(from_node, to_pt, max_dist=step_size):
    dir_vec = to_pt - from_node.pos
    dist = np.linalg.norm(dir_vec)
    new_pos = to_pt if dist <= max_dist else from_node.pos + dir_vec / dist * max_dist
    new_node = Node(new_pos)
    new_node.parent = from_node
    new_node.cost = from_node.cost + np.linalg.norm(new_node.pos - from_node.pos)
    return new_node

# RRT* Planning
nodes = [Node(start)]
tree_edges = []
goal_node = None

for i in range(max_iters):
    rnd = sample()
    nearest_node = nearest(nodes, rnd)
    new_node = steer(nearest_node, rnd)
    if not obstacle_free(nearest_node.pos, new_node.pos):
        continue

    # rewire
    neighbor_idxs = [j for j, n in enumerate(nodes)
                     if np.linalg.norm(n.pos - new_node.pos) < neighbor_radius
                     and obstacle_free(n.pos, new_node.pos)]
    min_cost, best_parent = nearest_node.cost + np.linalg.norm(nearest_node.pos - new_node.pos), nearest_node
    for j in neighbor_idxs:
        potential = nodes[j]
        cost = potential.cost + np.linalg.norm(potential.pos - new_node.pos)
        if cost < min_cost:
            min_cost, best_parent = cost, potential

    new_node.parent, new_node.cost = best_parent, min_cost
    nodes.append(new_node)
    tree_edges.append((best_parent.pos.copy(), new_node.pos.copy()))

    for j in neighbor_idxs:
        potential = nodes[j]
        new_cost = new_node.cost + np.linalg.norm(new_node.pos - potential.pos)
        if new_cost < potential.cost and obstacle_free(new_node.pos, potential.pos):
            potential.parent, potential.cost = new_node, new_cost

    if np.linalg.norm(new_node.pos - goal) < goal_radius:
        goal_node = steer(new_node, goal)
        if obstacle_free(new_node.pos, goal_node.pos):
            goal_node.parent = new_node
            goal_node.cost = new_node.cost + np.linalg.norm(new_node.pos - goal)
            nodes.append(goal_node)
            tree_edges.append((new_node.pos.copy(), goal_node.pos.copy()))
            print(f"Goal reached at iteration {i}")
            break

if goal_node is None:
    raise Exception("No path found to goal")

# Extract final path
def extract_path(node):
    path = []
    while node:
        path.append(node.pos.copy())
        node = node.parent
    return path[::-1]

path = extract_path(goal_node)

# Visualization setup
fig, ax = plt.subplots()
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')
plt.grid(True)
plt.title("Robot Moving with RRT*")

for ox, oy, w, h in obstacles:
    ax.add_patch(Rectangle((ox, oy), w, h, color='black'))

ax.plot(start[0], start[1], 'go', label="Start")
ax.plot(goal[0], goal[1], 'ro', label="Goal")
ax.legend()

# Robot parameters
robot_radius = 0.4
wheel_radius = 0.1
wheel_offset = 0.45  # lateral distance from center

# Create robot body and wheels
robot_body = Circle((start[0], start[1]), robot_radius, color='orange', zorder=5)
left_wheel  = Circle((0, 0), wheel_radius, color='gray', zorder=6)
right_wheel = Circle((0, 0), wheel_radius, color='gray', zorder=6)
ax.add_patch(robot_body)
ax.add_patch(left_wheel)
ax.add_patch(right_wheel)

# Animation parameters
build_frames   = len(tree_edges)
travel_steps   = 10
total_frames   = build_frames + (len(path) - 1) * travel_steps

def update(frame):
    if frame < build_frames:
        # drawing the RRT* tree
        p, c = tree_edges[frame]
        ax.plot([p[0], c[0]], [p[1], c[1]], 'b-', linewidth=0.5)
        ax.plot(c[0], c[1], 'ko', markersize=2)
    else:
        # draw final path once
        if frame == build_frames:
            px = [p[0] for p in path]
            py = [p[1] for p in path]
            ax.plot(px, py, 'r-', linewidth=2, label="Final Path")

        t = frame - build_frames
        seg = t // travel_steps
        t_norm = (t % travel_steps) / travel_steps
        if seg >= len(path) - 1:
            seg, t_norm = len(path) - 2, 1.0

        p1, p2 = path[seg], path[seg + 1]
        # compute current position
        cx = (1 - t_norm) * p1[0] + t_norm * p2[0]
        cy = (1 - t_norm) * p1[1] + t_norm * p2[1]
        robot_body.center = (cx, cy)

        # compute heading
        theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        # wheel offsets rotated by heading ±90°
        dx = wheel_offset * math.cos(theta + math.pi/2)
        dy = wheel_offset * math.sin(theta + math.pi/2)
        left_wheel.center  = (cx + dx, cy + dy)
        right_wheel.center = (cx - dx, cy - dy)

    return robot_body, left_wheel, right_wheel

ani = animation.FuncAnimation(fig, update,
                              frames=total_frames,
                              interval=20,
                              blit=False,
                              repeat=False)

plt.show()

