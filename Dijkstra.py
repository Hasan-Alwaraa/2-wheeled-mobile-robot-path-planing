import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation
import heapq, random, math

# Grid and obstacles
grid_size = 1
width, height = 20, 20
cols, rows = int(width/grid_size), int(height/grid_size)
grid = np.zeros((rows, cols))

def generate_random_obstacles(num=7, max_w=6, max_h=6):
    obs = []
    for _ in range(num):
        w = random.randint(2, max_w); h = random.randint(2, max_h)
        ox = random.randint(0, cols-w-1); oy = random.randint(0, rows-h-1)
        obs.append((ox, oy, w, h))
    return obs

obstacles = generate_random_obstacles()
for ox, oy, w, h in obstacles:
    grid[oy:oy+h, ox:ox+w] = 1

# Inflate obstacles by one cell to account for robot radius+wheels clearance (~0.6 < 1 cell)
inflated = grid.copy()
for i in range(rows):
    for j in range(cols):
        if grid[i, j] == 1:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        inflated[ni, nj] = 1

# Dijkstra planning
start = (1,1); goal = (18,18)

def in_bounds(x,y):
    return 0<=x<cols and 0<=y<rows

def neighbors(x,y):
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    out = []
    for dx, dy in dirs:
        nx,ny = x+dx, y+dy
        # use inflated grid for safety
        if in_bounds(nx,ny) and inflated[ny][nx]==0:
            out.append((nx,ny))
    return out

# Dijkstra algorithm
def dijkstra(start, goal):
    pq=[(0,start)]; costs={start:0}; prev={}
    while pq:
        cost,u = heapq.heappop(pq)
        if u==goal: break
        for v in neighbors(*u):
            newc = cost + math.hypot(v[0]-u[0], v[1]-u[1])
            if v not in costs or newc < costs[v]:
                costs[v]=newc; prev[v]=u; heapq.heappush(pq,(newc,v))
    if goal not in prev: raise Exception("No path found")
    path=[]; cur=goal
    while cur!=start:
        path.append(cur); cur=prev[cur]
    path.append(start); path.reverse()
    return path

path = dijkstra(start, goal)

# Visualization setup
fig, ax = plt.subplots()
ax.set_xlim(0,cols); ax.set_ylim(0,rows)
ax.set_aspect('equal'); plt.grid(True)
plt.title("Realistic Robot Simulation with Dijkstra (with Clearance)")
for ox,oy,w,h in obstacles:
    ax.add_patch(Rectangle((ox,oy),w,h,color='black'))
ax.plot(start[0],start[1],'go'); ax.plot(goal[0],goal[1],'ro')

# Robot parameters
robot_radius = 0.5
wheel_r = 0.1
d = robot_radius  # distance from center to wheel

# Create patches: circular body and two wheels
robot_body = Circle((start[0], start[1]), robot_radius, color='orange', zorder=5)
left_wheel = Circle((start[0] + d*math.cos(0) - d*math.sin(0),
                     start[1] + d*math.sin(0) + d*math.cos(0)),
                    wheel_r, color='gray', zorder=6)
right_wheel = Circle((start[0] + d*math.cos(0) + d*math.sin(0),
                      start[1] + d*math.sin(0) - d*math.cos(0)),
                     wheel_r, color='gray', zorder=6)
ax.add_patch(robot_body); ax.add_patch(left_wheel); ax.add_patch(right_wheel)

# Draw path
px = [p[0] for p in path]; py = [p[1] for p in path]
ax.plot(px, py, 'r-', linewidth=2)

# Animation
steps_per_seg = 10
total_frames = (len(path)-1)*steps_per_seg

def update(f):
    seg = f // steps_per_seg
    t = (f % steps_per_seg)/steps_per_seg
    if seg >= len(path)-1:
        seg, t = len(path)-2, 1.0
    p1 = np.array(path[seg]); p2 = np.array(path[seg+1])
    # compute current position
    pos = p1 + (p2-p1)*t
    theta = math.atan2(p2[1]-p1[1], p2[0]-p1[0])
    # update body center
    robot_body.center = (pos[0], pos[1])
    # compute rotated wheel offsets
    offs = [(math.cos(theta+math.pi/2)*d, math.sin(theta+math.pi/2)*d),
            (math.cos(theta-math.pi/2)*d, math.sin(theta-math.pi/2)*d)]
    left_wheel.center = (pos[0] + offs[0][0], pos[1] + offs[0][1])
    right_wheel.center = (pos[0] + offs[1][0], pos[1] + offs[1][1])
    return robot_body, left_wheel, right_wheel

ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=30, blit=False, repeat=False)
plt.show()

