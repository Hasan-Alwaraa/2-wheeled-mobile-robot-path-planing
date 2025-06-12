import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
import heapq
import random
import math

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

# mark raw occupancy
grid.fill(0)
for ox, oy, w, h in obstacles:
    grid[int(oy):int(oy + h), int(ox):int(ox + w)] = 1

# Inflate obstacles by one cell to account for robot radius (~0.4 < 1 cell)
inflated = grid.copy()
for i in range(rows):
    for j in range(cols):
        if grid[i, j] == 1:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        inflated[ni, nj] = 1

# ========== A* ==========
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def neighbors(node):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),   # cardinal directions
            (-1, -1), (-1, 1), (1, -1), (1, 1)]  # diagonal directions
    result = []
    for d in dirs:
        ni, nj = node[0] + d[0], node[1] + d[1]
        if 0 <= ni < rows and 0 <= nj < cols and inflated[ni, nj] == 0:
            result.append((ni, nj))
    return result


def astar(start, goal):
    start = (int(start[1]), int(start[0]))
    goal = (int(goal[1]), int(goal[0]))
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        for nb in neighbors(current):
            new_cost = cost + 1
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                priority = new_cost + heuristic(nb, goal)
                heapq.heappush(open_set, (priority, new_cost, nb))
                came_from[nb] = current
    return None

start = np.array([1.0, 1.0])
goal  = np.array([18.0, 18.0])

path = astar(start, goal)
if not path:
    raise Exception("No path found!")

fig, ax = plt.subplots()
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_aspect('equal')
plt.grid(True)
plt.title("Robot Moving with A* Path (with Clearance)")

# draw only obstacles in black
for ox, oy, w, h in obstacles:
    ax.add_patch(Rectangle((ox, oy), w, h, color='black'))

# plot start/goal and path
px = [p[1] + 0.5 for p in path]
py = [p[0] + 0.5 for p in path]
ax.plot(px, py, 'b--', linewidth=1)
ax.plot(start[0] + 0.5, start[1] + 0.5, 'go')
ax.plot(goal[0] + 0.5, goal[1] + 0.5, 'ro')

# Robot parameters
robot_radius = 0.4
wheel_radius = 0.1
wheel_offset = 0.45

# create robot patches
robot_body = Circle((px[0], py[0]), robot_radius, color='orange', zorder=5)
left_wheel  = Circle((0, 0), wheel_radius, color='gray', zorder=6)
right_wheel = Circle((0, 0), wheel_radius, color='gray', zorder=6)
ax.add_patch(robot_body)
ax.add_patch(left_wheel)
ax.add_patch(right_wheel)

# Animation
steps_between = 10
total_frames = (len(path) - 1) * steps_between

def update(frame):
    seg = frame // steps_between
    t   = (frame % steps_between) / steps_between
    if seg >= len(path) - 1:
        seg, t = len(path) - 2, 1.0

    y1, x1 = path[seg]
    y2, x2 = path[seg + 1]
    cx = (1 - t)*(x1 + 0.5) + t*(x2 + 0.5)
    cy = (1 - t)*(y1 + 0.5) + t*(y2 + 0.5)

    # update robot body
    robot_body.center = (cx, cy)

    # compute heading and wheel positions
    theta = math.atan2(y2 - y1, x2 - x1)
    dx = wheel_offset * math.cos(theta + math.pi/2)
    dy = wheel_offset * math.sin(theta + math.pi/2)
    left_wheel.center  = (cx + dx, cy + dy)
    right_wheel.center = (cx - dx, cy - dy)

    return robot_body, left_wheel, right_wheel

ani = animation.FuncAnimation(fig, update,
                              frames=total_frames,
                              interval=30,
                              blit=True,
                              repeat=False)

plt.show()

