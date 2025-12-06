#!/usr/bin/env python3
"""
ant_navigation_simulation.py
Simple 2D ant-inspired navigation simulation:
- Outbound stochastic search (records step vectors)
- Path integration (sum of step vectors)
- Homing via fusion of PI estimate and noisy sun-compass

Produces:
 - ant_navigation_simulation.gif
 - ant_bio_inspired_report.pdf
 - README_ant_simulation.txt

Dependencies:
 - numpy, matplotlib, pillow
Install: pip install numpy matplotlib pillow
Run: python ant_navigation_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import atan2
from matplotlib.backends.backend_pdf import PdfPages

# ----------------------------
# Parameters (tweak as needed)
# ----------------------------
np.random.seed(42)
step_length = 0.5
max_steps_outbound = 400
search_turn_prob = 0.15
food_radius = 0.8
nest = np.array([0.0, 0.0])
sun_azimuth = np.deg2rad(40)
compass_noise_std = np.deg2rad(6)
theta0 = np.deg2rad(0.0)
r_food = 12.0
theta_food = np.deg2rad(65)
food = np.array([r_food * np.cos(theta_food), r_food * np.sin(theta_food)])
homing_k = 0.9  # PI trust (0..1), sun trust = 1-homing_k
max_homing_steps = 500

# ----------------------------
# Outbound search
# ----------------------------
pos = nest.copy()
theta = theta0
positions_out = [pos.copy()]
path_vectors = []
found = False

for step in range(max_steps_outbound):
    if np.random.rand() < search_turn_prob:
        theta += np.random.normal(scale=np.deg2rad(30))
    step_vec = np.array([step_length * np.cos(theta), step_length * np.sin(theta)])
    pos = pos + step_vec
    positions_out.append(pos.copy())
    path_vectors.append(step_vec.copy())
    if np.linalg.norm(pos - food) <= food_radius:
        found = True
        break

if not found:
    # fallback
    food = positions_out[-1] + np.array([2.0, 0.0])

positions_out = np.array(positions_out)
pi_vector = np.sum(path_vectors, axis=0)
pi_distance = np.linalg.norm(pi_vector)
pi_angle = atan2(pi_vector[1], pi_vector[0])

# ----------------------------
# Homing (fusion of PI and sun)
# ----------------------------
pos_home = positions_out[-1].copy()
positions_home = [pos_home.copy()]
sun_k = 1 - homing_k

for step in range(max_homing_steps):
    to_nest_vec = nest - pos_home
    true_to_nest_angle = atan2(to_nest_vec[1], to_nest_vec[0])
    noisy_sun = sun_azimuth + np.random.normal(scale=compass_noise_std)
    sun_relative_to_nest = true_to_nest_angle - sun_azimuth
    compass_est_angle = noisy_sun + sun_relative_to_nest
    drift = np.random.normal(scale=np.deg2rad(1.2))  # small PI drift
    pi_est_angle = atan2(-pi_vector[1], -pi_vector[0]) + drift  # negative PI points to nest
    fused_angle = homing_k * pi_est_angle + sun_k * compass_est_angle
    step_vec = np.array([step_length * np.cos(fused_angle), step_length * np.sin(fused_angle)])
    pos_home = pos_home + step_vec
    positions_home.append(pos_home.copy())
    if np.linalg.norm(pos_home - nest) <= 0.8:
        homing_success = True
        break
else:
    homing_success = False

positions_home = np.array(positions_home)

# ----------------------------
# Animation
# ----------------------------
all_positions = np.vstack([positions_out, positions_home, nest[None, :], food[None, :]])
min_x, max_x = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
min_y, max_y = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
span = max(max_x - min_x, max_y - min_y)
cx, cy = (max_x + min_x) / 2, (max_y + min_y) / 2
margin = 2.0

fig, ax = plt.subplots(figsize=(6,6))
fig.subplots_adjust(right=0.8)  # leave space for legend outside plot box
ax.set_aspect('equal', 'box')
ax.set_xlim(cx - span/2 - margin, cx + span/2 + margin)
ax.set_ylim(cy - span/2 - margin, cy + span/2 + margin)
ax.grid(True, linewidth=0.3)
ax.set_title('Ant-inspired navigation: outbound (blue) and homing (red)')
nest_dot, = ax.plot([], [], 'ks', markersize=6, label='Nest')
food_dot, = ax.plot([], [], 'go', markersize=8, label='Food')
out_line, = ax.plot([], [], '-', lw=1, label='Outbound')
home_line, = ax.plot([], [], '-', lw=1, color='red', label='Homing')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0, framealpha=0.8)

max_frames = len(positions_out) + len(positions_home)
def init():
    nest_dot.set_data([nest[0]], [nest[1]])
    food_dot.set_data([food[0]], [food[1]])
    out_line.set_data([], [])
    home_line.set_data([], [])
    return nest_dot, food_dot, out_line, home_line, time_text

def animate(i):
    if i < len(positions_out):
        out_line.set_data(positions_out[:i+1,0], positions_out[:i+1,1])
        home_line.set_data([], [])
    else:
        out_line.set_data(positions_out[:,0], positions_out[:,1])
        j = i - len(positions_out)
        home_line.set_data(positions_home[:j+1,0], positions_home[:j+1,1])
    time_text.set_text(f'Frame {i}')
    return out_line, home_line, time_text

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=max_frames, interval=60, blit=False)

gif_path = 'ant_navigation_simulation.gif'
anim.save(gif_path, writer='pillow', fps=10)
plt.close(fig)

# ----------------------------
# PDF report (2 pages)
# ----------------------------
report_pdf = 'ant_bio_inspired_report.pdf'
with PdfPages(report_pdf) as pdf:
    # Page 1 (title, summary, algorithm)
    fig1 = plt.figure(figsize=(8.27, 11.69))
    fig1.text(0.05, 0.95, 'Bio-inspired navigation: Desert-ant inspired homing', fontsize=14, weight='bold')
    summary = (
        "Summary (~300 words):\n\n"
        "Desert ants (Cataglyphis spp.) navigate in featureless environments by combining a path-integration "
        "strategy with a robust sun-compass. While foraging, ants record their outbound route as a sequence of "
        "step vectors; summing these yields a home-vector pointing back to the nest. The sun provides an absolute "
        "directional reference which ants use to reduce cumulative errors in their internal estimate. In the video, "
        "ants demonstrate stereotyped straight-line homebound paths after outward meandering searches, indicating "
        "integration of directional cues and dead-reckoning. To model this behavior, I implemented a minimal 2D "
        "simulation where an agent (ant) performs an outbound stochastic search until it encounters food. The agent "
        "stores its step vectors to compute a path-integration (PI) vector. For homing, the agent fuses the PI-based "
        "estimate of nest direction with a noisy sun-compass reading; a tunable trust parameter controls the relative "
        "weight between PI and compass. This fusion captures the ant's ability to correct accumulated PI error using an "
        "external celestial cue. Results show outbound meandering and a generally successful homing trajectory; homing "
        "accuracy depends on compass noise and the trust weight. The simulation illustrates how simple vector-summing, "
        "combined with a global reference, can produce robust navigation in sparse environments."
    )
    fig1.text(0.05, 0.62, summary, fontsize=10)
    algo_title = "Algorithm (codified):"
    algo_text = (
        "1. Outbound search:\n"
        "   - Start at nest (0,0). Initialize heading.\n"
        "   - Perform stochastic steps: each step of fixed length with occasional random turns.\n"
        "   - Record each step vector in a list (for path integration).\n"
        "   - Stop when the agent is within a small radius of the food.\n\n"
        "2. Path Integration (PI):\n"
        "   - Sum all outbound step vectors: PI_vector = sum of steps.\n"
        "   - PI gives an estimate of the vector from nest to current position; negating it points towards nest.\n\n"
        "3. Homing (fused cue):\n"
        "   - Obtain a noisy sun-compass reading (absolute azimuth + Gaussian noise).\n"
        "   - Form two direction estimates: PI-based angle and compass-based angle (to-nest estimate with noise).\n"
        "   - Fuse angles by weighted average: fused = w_PI * angle_PI + w_sun * angle_compass.\n"
        "   - Step towards fused direction until within nest radius or max steps.\n"
    )
    fig1.text(0.05, 0.35, algo_title, fontsize=12, weight='bold')
    fig1.text(0.05, 0.10, algo_text, fontsize=10)
    pdf.savefig(fig1)
    plt.close(fig1)

    # Page 2 (trajectories)
    fig2 = plt.figure(figsize=(8.27, 11.69))
    fig2.subplots_adjust(right=0.78)  # leave margin for legend/metrics
    ax2 = fig2.add_axes([0.05, 0.05, 0.72, 0.9])
    ax2.set_title('Screenshot: trajectories and PI arrow')
    ax2.set_xlim(cx - span/2 - margin, cx + span/2 + margin)
    ax2.set_ylim(cy - span/2 - margin, cy + span/2 + margin)
    ax2.grid(True, linewidth=0.3)
    ax2.plot(positions_out[:,0], positions_out[:,1], '-', lw=1, label='Outbound')
    ax2.plot(positions_home[:,0], positions_home[:,1], '-', lw=1, color='red', label='Homing')
    ax2.plot(nest[0], nest[1], 'ks', label='Nest')
    ax2.plot(food[0], food[1], 'go', label='Food')
    ax2.arrow(nest[0], nest[1], pi_vector[0], pi_vector[1], color='orange', width=0.12, label='PI vector')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.0, framealpha=0.8)
    metrics = (
        f"Homing success: {homing_success}\n"
        f"Outbound steps: {len(positions_out)-1}\n"
        f"Homing steps: {len(positions_home)-1}\n"
        f"PI distance: {pi_distance:.2f}\n"
        f"Final homing error (distance to nest): {np.linalg.norm(positions_home[-1]-nest):.2f}\n"
        f"Sun azimuth (deg): {np.rad2deg(sun_azimuth):.1f}\n"
    )
    fig2.text(0.80, 0.5, metrics, fontsize=10, ha='left', va='center',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
              transform=fig2.transFigure)
    pdf.savefig(fig2)
    plt.close(fig2)

# README
with open('README_ant_simulation.txt', 'w') as f:
    f.write("Files created by ant_navigation_simulation.py\n")
    f.write("- ant_navigation_simulation.gif\n")
    f.write("- ant_bio_inspired_report.pdf\n")
    f.write("\nRun the script with: python ant_navigation_simulation.py\n")
    f.write("Requires: numpy, matplotlib, pillow\n")

print("Done. Generated:")
print(" - ant_navigation_simulation.gif")
print(" - ant_bio_inspired_report.pdf")
print(" - README_ant_simulation.txt")
