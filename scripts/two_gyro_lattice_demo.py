# scripts/two_gyro_lattice_demo.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
from tqdm import tqdm

# ==================== QUATERNION HELPERS ====================
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def q_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_normalize(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q

def small_rotor(theta, axis=np.array([0., 0., 1.])):
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half = theta / 2
    return np.array([np.cos(half), *(np.sin(half) * axis)])

# ==================== LATTICE DEMO ====================
class TwoGyroLattice:
    def __init__(self, mode="stable", n_sites=96, gauge_strength=0.85):
        self.mode = mode
        self.n = n_sites
        self.gauge_strength = gauge_strength

        self.q = np.array([q_normalize(np.random.randn(4)) for _ in range(n_sites)])
        self.identity = np.array([q_normalize(np.random.randn(4)) for _ in range(n_sites)])
        self.initial_identity = self.identity.copy()

        self.twist = np.zeros(n_sites)
        self.burst_events = []
        self.pointer_history = []
        self.mean_twist_history = []
        self.identity_preservation = []
        self.omega_L = 0.025
        self.omega_R = 0.023 if mode == "stable" else 0.018

    def run(self, frames=1200):
        for frame in tqdm(range(frames), desc=f"{self.mode.capitalize()} 2-Gyro Run"):
            delta_L = small_rotor(self.omega_L)
            delta_R = small_rotor(self.omega_R)

            # Two-gyro update
            for i in range(self.n):
                q_temp = q_mult(delta_L, self.q[i])
                self.q[i] = q_mult(q_temp, q_conj(delta_R))
                self.q[i] = q_normalize(self.q[i])
                self.twist[i] = 2 * np.arccos(np.clip(self.q[i][0], -1.0, 1.0))

            # Gauge connection (analytical scale pointer)
            avg_imbalance = np.mean(self.twist) % (2 * np.pi)
            gauge_alpha = -self.gauge_strength * avg_imbalance
            gauge_rot = np.array([np.cos(gauge_alpha), 0., 0., np.sin(gauge_alpha)])

            for i in range(self.n):
                self.q[i] = q_mult(self.q[i], gauge_rot)
                self.q[i] = q_normalize(self.q[i])
                self.identity[i] = q_mult(self.identity[i], gauge_rot)
                self.identity[i] = q_normalize(self.identity[i])

            # Burst / reconnection
            bursts_this_step = 0
            for i in range(self.n):
                if self.twist[i] > 5.8:
                    self.q[i] = q_normalize(0.3 * np.array([1., 0., 0., 0.]) + 0.7 * self.q[i])
                    self.twist[i] *= 0.15
                    bursts_this_step += 1

            if bursts_this_step > 0:
                self.burst_events.append((frame, bursts_this_step))

            pointer = np.tanh(gauge_alpha * 6)
            self.pointer_history.append(pointer)
            self.mean_twist_history.append(np.mean(self.twist))
            cosines = np.sum(self.identity * self.initial_identity, axis=1)
            self.identity_preservation.append(np.mean(cosines))

        return self

# ==================== VISUALIZATION ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stable", "chaotic"], default="stable")
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--gauge", type=float, default=0.85)
    args = parser.parse_args()

    print(f"Running {args.mode} two-gyro gauged lattice demo...")
    demo = TwoGyroLattice(mode=args.mode, gauge_strength=args.gauge)
    demo.run(frames=args.frames)


    class LatticeDemo:
        def __init__(self, mode="stable", n_sites=96):
            self.mode = mode
            self.n = n_sites
            self.q = np.random.randn(n_sites, 4)
            self.q = np.array([q_normalize(row) for row in self.q])
            self.twist = np.zeros(n_sites)
            self.identity = np.random.randn(n_sites, 4)
            self.identity = np.array([q_normalize(row) for row in self.identity])
            self.initial_identity = self.identity.copy()
            self.burst_events = []
            self.burst_sizes = []
            self.total_bursts = 0
            self.pointer_history = []
            self.mean_twist_history = []
            self.identity_preservation = []
            self.omega_L = 0.025
            self.omega_R = 0.023
            self.gauge_strength = 0.85 if mode == "stable" else 0.08

        def run(self, frames=1200):
            for frame in tqdm(range(frames), desc=f"{self.mode.capitalize()} simulation"):
                axis = np.array([0., 0., 1.])
                delta_L = small_rotor(self.omega_L, axis)
                delta_R = small_rotor(self.omega_R, axis)

                for i in range(self.n):
                    q_temp = q_mult(delta_L, self.q[i])
                    self.q[i] = q_mult(q_temp, q_conj(delta_R))
                    self.q[i] = q_normalize(self.q[i])
                    self.twist[i] = 2 * np.arccos(np.clip(self.q[i][0], -1.0, 1.0))

                avg_imbalance = np.mean(self.twist) % (2 * np.pi)
                self.gauge_alpha = -self.gauge_strength * avg_imbalance
                gauge_rot = np.array([np.cos(self.gauge_alpha), 0., 0., np.sin(self.gauge_alpha)])

                for i in range(self.n):
                    self.q[i] = q_mult(self.q[i], gauge_rot)
                    self.q[i] = q_normalize(self.q[i])
                    self.identity[i] = q_mult(self.identity[i], gauge_rot)
                    self.identity[i] = q_normalize(self.identity[i])

                bursts_this_step = 0
                for i in range(self.n):
                    if self.twist[i] > 5.8:
                        self.q[i] = q_normalize(0.3 * np.array([1., 0., 0., 0.]) + 0.7 * self.q[i])
                        self.twist[i] *= 0.15
                        bursts_this_step += 1

                self.burst_sizes.append(bursts_this_step)
                if bursts_this_step > 0:
                    self.burst_events.append((frame, bursts_this_step))
                    self.total_bursts += bursts_this_step

                pointer = np.tanh(self.gauge_alpha * 6)
                self.pointer_history.append(pointer)
                self.mean_twist_history.append(np.mean(self.twist))
                cosines = np.sum(self.identity * self.initial_identity, axis=1)
                self.identity_preservation.append(np.mean(cosines))


    # Run both
    print("Running chaotic + stable simulations...")
    chaotic = LatticeDemo(mode="chaotic")
    stable = LatticeDemo(mode="stable")
    chaotic.run(frames=1200)
    stable.run(frames=1200)

    # === FINAL OPTIMIZED LANDSCAPE LAYOUT ===
    fig, axs = plt.subplots(2, 2, figsize=(20, 11),
                            gridspec_kw={'height_ratios': [1, 3.2], 'width_ratios': [1, 1]})
    fig.suptitle(
        '2-Gyro Gauged Quaternion Lattice + QVPIC\nChaotic (left) vs Stable Balanced (right) — Gauge Fiber in Action',
        fontsize=16, fontweight='bold')

    ax_chaotic_ptr = axs[0, 0]
    ax_stable_ptr = axs[0, 1]
    ax_chaotic_hist = axs[1, 0]
    ax_stable_hist = axs[1, 1]

    # Chaotic pointer (bright red)
    ax_chaotic_ptr.set_title('Chaotic / Unbalanced — Pointer Swings Visibly', fontsize=13)
    ax_chaotic_ptr.set_xlim(-1.2, 1.2)
    ax_chaotic_ptr.set_ylim(-0.4, 0.4)
    ax_chaotic_ptr.axhline(0, color='black', lw=4)
    ax_chaotic_ptr.axvline(0, color='gold', lw=6, alpha=0.7)
    needle_chaotic, = ax_chaotic_ptr.plot([0, 0], [0, 0.28], color='#FF3333', lw=8)  # vivid red

    # Stable pointer (teal/cyan)
    ax_stable_ptr.set_title('Stable / Gauged + QVPIC — Centered with Subtle Gauge Jitter', fontsize=13)
    ax_stable_ptr.set_xlim(-1.2, 1.2)
    ax_stable_ptr.set_ylim(-0.4, 0.4)
    ax_stable_ptr.axhline(0, color='black', lw=4)
    ax_stable_ptr.axvline(0, color='gold', lw=6, alpha=0.7)
    needle_stable, = ax_stable_ptr.plot([0, 0], [0, 0.28], color='#00E5CC', lw=8)  # teal

    # Chaotic history
    line_twist_c, = ax_chaotic_hist.plot([], [], color='#1E90FF', lw=2, label='Mean Twist (rad)')
    line_pointer_c, = ax_chaotic_hist.plot([], [], color='#FF8C00', lw=2, label='Pointer Position')
    line_id_c, = ax_chaotic_hist.plot([], [], color='#00FF9F', lw=2, label='Identity Preservation (cosine)')
    ax_chaotic_hist.set_ylim(0, 7)
    ax_chaotic_hist.legend(loc='upper right', fontsize=10)
    ax_chaotic_hist.grid(True, alpha=0.2)
    ax_chaotic_hist.set_title('Chaotic History', fontsize=13)

    # Stable history
    line_twist_s, = ax_stable_hist.plot([], [], color='#1E90FF', lw=2, label='Mean Twist (rad)')
    line_pointer_s, = ax_stable_hist.plot([], [], color='#FF8C00', lw=2, label='Pointer Position')
    line_id_s, = ax_stable_hist.plot([], [], color='#00FF9F', lw=2, label='Identity Preservation (cosine)')
    ax_stable_hist.set_ylim(0, 7)
    ax_stable_hist.legend(loc='upper right', fontsize=10)
    ax_stable_hist.grid(True, alpha=0.2)
    ax_stable_hist.set_title('Stable History', fontsize=13)

    # Avalanche markers (bright magenta)
    for ax, events in [(ax_chaotic_hist, chaotic.burst_events), (ax_stable_hist, stable.burst_events)]:
        for step, size in events:
            ax.axvline(x=step, ymin=0, ymax=0.25, color='#FF00AA', alpha=0.85, lw=1.8, linestyle='--')


    def update(frame):
        x = np.arange(frame + 1)

        p_c = chaotic.pointer_history[frame]
        needle_chaotic.set_data([0, np.sin(p_c * np.pi / 2.3) * 0.95], [0, np.cos(p_c * np.pi / 2.3) * 0.28])

        line_pointer_c.set_data(x, chaotic.pointer_history[:frame + 1])
        line_twist_c.set_data(x, chaotic.mean_twist_history[:frame + 1])
        line_id_c.set_data(x, chaotic.identity_preservation[:frame + 1])

        p_s = stable.pointer_history[frame]
        needle_stable.set_data([0, np.sin(p_s * np.pi / 2.3) * 0.95], [0, np.cos(p_s * np.pi / 2.3) * 0.28])

        line_pointer_s.set_data(x, stable.pointer_history[:frame + 1])
        line_twist_s.set_data(x, stable.mean_twist_history[:frame + 1])
        line_id_s.set_data(x, stable.identity_preservation[:frame + 1])

        ax_chaotic_hist.set_xlim(0, frame)
        ax_stable_hist.set_xlim(0, frame)

        return needle_chaotic, needle_stable, line_pointer_c, line_twist_c, line_id_c, line_pointer_s, line_twist_s, line_id_s


    print("🎥 Rendering final tweaked color scheme version...")
    ani = FuncAnimation(fig, update, frames=1200, interval=25, blit=False)

    ani.save('two_gyro_full_split_demo_FINAL.mp4', writer='ffmpeg', fps=30, dpi=160,
             extra_args=['-pix_fmt', 'yuv420p', '-crf', '17'])

    print("\n✅ Saved as 'two_gyro_full_split_demo_FINAL.mp4'")
    print("   → New professional color scheme applied")
    print("   → High-contrast, demonstration-ready")
    print("   → Grids added for readability")

    plt.close(fig)
    print("✅ Simulation complete — ready for integration into conduit.")
