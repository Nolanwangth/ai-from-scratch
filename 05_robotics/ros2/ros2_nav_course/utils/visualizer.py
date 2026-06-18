"""
Navigation Visualizer — matplotlib real-time animation
======================================================
3 panels: Map (SLAM + path + robot) | Status | D435 depth view

Renders at ~8Hz. Costmap overlay at ~1Hz (it's heavy).
Control loop calls set_state() every frame, draw() only fires at render rate.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from typing import List, Tuple, Optional


class NavigationVisualizer:

    def __init__(self, world):
        self.world = world
        self.fig = plt.figure(figsize=(16, 7))
        self.fig.canvas.manager.set_window_title('ROS2 Navigation - Live')

        gs = self.fig.add_gridspec(1, 3, width_ratios=[3.5, 1.2, 2.5])
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_info = self.fig.add_subplot(gs[1])
        self.ax_depth = self.fig.add_subplot(gs[2])

        self.trajectory_x = []
        self.trajectory_y = []
        self.goal = (8.0, 8.0)
        self.costmap_img = None
        self._obs_patches = []
        self._obs_texts = []
        self.robot_arrow = None
        self.info_keys = {}
        self._depth_cache = None
        self._depth_frame = 0
        self._pending_costmap = None
        self._frame_count = 0

        self._build_map_view()
        self._build_info_panel()
        self._build_depth_view()

        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

    # ─── Map View ────────────────────────────────────────────

    def _build_map_view(self):
        ax = self.ax_map
        w, h = self.world.width, self.world.height
        ax.set_xlim(-0.5, w + 0.5)
        ax.set_ylim(-0.5, h + 0.5)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('SLAM Map + Navigation', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.15)

        self._draw_obstacles()

        for name, (lx, ly) in self.world.landmarks.items():
            is_goal = (name == 'workstation')
            ax.plot(lx, ly, marker='D' if is_goal else 's',
                    color='#D32F2F' if is_goal else '#757575',
                    markersize=12 if is_goal else 7, zorder=5)
            ax.annotate(name, (lx, ly), textcoords="offset points",
                        xytext=(5, 11), ha='center', fontsize=8, color='#222')

        self.robot_marker, = ax.plot([], [], 'o', color='#1976D2',
                                      markersize=13, zorder=10,
                                      markeredgewidth=2.5,
                                      markeredgecolor='#0D47A1', label='Robot')
        self.traj_line, = ax.plot([], [], '-', color='#42A5F5', linewidth=2,
                                   alpha=0.7, label='Trajectory')
        self.path_line, = ax.plot([], [], '--', color='#43A047', linewidth=2.5,
                                   alpha=0.85, label='A* Plan')
        self.kf_scatter = ax.scatter([], [], c='#EF6C00', s=28, marker='s',
                                      alpha=0.55, zorder=3,
                                      edgecolors='#BF360C', linewidth=0.5,
                                      label='SLAM KFs')
        ax.legend(loc='lower left', fontsize=8, ncol=2, framealpha=0.8)

    def _draw_obstacles(self):
        ax = self.ax_map
        for p in self._obs_patches:
            try: p.remove()
            except Exception: pass
        for t in self._obs_texts:
            try: t.remove()
            except Exception: pass
        self._obs_patches.clear()
        self._obs_texts.clear()

        for wall in self.world.walls:
            r = patches.Rectangle(
                (wall.left, wall.bottom), wall.w, wall.h,
                linewidth=1, edgecolor='#666', facecolor='#888', alpha=0.2)
            ax.add_patch(r)
            self._obs_patches.append(r)

        for obs in self.world.obstacles:
            is_dyn = (obs.name == 'BOX!')
            r = patches.Rectangle(
                (obs.left, obs.bottom), obs.w, obs.h,
                linewidth=2.5 if is_dyn else 1.5,
                edgecolor='#B71C1C' if is_dyn else '#5D4037',
                facecolor='#EF9A9A' if is_dyn else '#D7CCC8',
                alpha=0.85 if is_dyn else 0.7)
            ax.add_patch(r)
            self._obs_patches.append(r)
            t = ax.text(obs.x, obs.y, obs.name, ha='center', va='center',
                        fontsize=8 if is_dyn else 7,
                        color='#B71C1C' if is_dyn else '#3E2723',
                        fontweight='bold')
            self._obs_texts.append(t)

    def refresh_obstacles(self):
        self._draw_obstacles()

    # ─── Info Panel ──────────────────────────────────────────

    def _build_info_panel(self):
        ax = self.ax_info
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('Status', fontsize=12, fontweight='bold')

        y, dy = 9.4, 0.65
        for label, keys in [
            ('-- Robot --', ['pose', 'vel', 'dist']),
            ('-- SLAM --',  ['kfs', 'loops', 'wm']),
            ('-- Control --', ['track', 'steps']),
        ]:
            ax.text(0.3, y, label, fontsize=8, color='#999', fontweight='bold')
            y -= dy * 0.8
            for k in keys:
                t = ax.text(0.5, y, f'{k}: --', fontsize=8, color='#333',
                           fontfamily='monospace')
                self.info_keys[k] = t
                y -= dy
            y -= dy * 0.3

    # ─── Depth View ──────────────────────────────────────────

    def _build_depth_view(self):
        ax = self.ax_depth
        ax.set_title('D435 Depth (robot POV)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Distance (m)')
        ax.set_xlim(-90, 90)
        ax.set_ylim(0, 6)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=5.0, color='r', linestyle=':', alpha=0.3, label='max')
        self.depth_line, = ax.plot([], [], '-', color='#FF5722', linewidth=2.5,
                                    label='scan')
        self.depth_true_line, = ax.plot([], [], '--', color='#4CAF50',
                                         linewidth=1.5, alpha=0.5, label='true')
        self.depth_hits = ax.scatter([], [], c='red', s=20, marker='x', zorder=5)
        ax.legend(loc='upper right', fontsize=7)

    # ─── Public API ──────────────────────────────────────────

    def update_costmap(self, costmap_data):
        self._pending_costmap = costmap_data

    def draw(self, robot_x, robot_y, robot_theta,
             robot_v, robot_omega, path=None, kfs=None,
             n_loops=0, wm_size=0, goal=None, step=0):
        """Render one frame (call at ~8Hz, not 50Hz!)"""
        self._frame_count += 1

        # --- Map ---
        self.trajectory_x.append(robot_x)
        self.trajectory_y.append(robot_y)
        self.robot_marker.set_data([robot_x], [robot_y])

        if self.robot_arrow is not None:
            try: self.robot_arrow.remove()
            except Exception: pass
        al = 0.5
        self.robot_arrow = FancyArrowPatch(
            (robot_x, robot_y),
            (robot_x + al * math.cos(robot_theta),
             robot_y + al * math.sin(robot_theta)),
            arrowstyle='->', mutation_scale=18,
            color='#0D47A1', linewidth=2.5, zorder=11)
        self.ax_map.add_patch(self.robot_arrow)
        self.traj_line.set_data(self.trajectory_x, self.trajectory_y)
        if path:
            self.path_line.set_data([p[0] for p in path], [p[1] for p in path])
        else:
            self.path_line.set_data([], [])
        if kfs:
            self.kf_scatter.set_offsets(
                np.column_stack([[k[0] for k in kfs], [k[1] for k in kfs]]))
        elif kfs is not None:
            self.kf_scatter.set_offsets(np.empty((0, 2)))

        # Costmap (heavy — only at ~1Hz)
        if self._pending_costmap is not None and self._frame_count % 6 == 0:
            self._draw_costmap(self._pending_costmap)
            self._pending_costmap = None

        # --- Status ---
        dist = math.sqrt((robot_x-goal[0])**2 + (robot_y-goal[1])**2) if goal else 0
        panel = {
            'pose': f"{robot_x:.2f}, {robot_y:.2f}, {math.degrees(robot_theta):.0f}deg",
            'vel': f"v={robot_v:.2f} w={robot_omega:.2f}",
            'dist': f"{dist:.2f}m",
            'kfs': f"{len(kfs) if kfs else 0}",
            'loops': f"{n_loops}",
            'wm': f"{wm_size}",
            'track': f"{dist:.3f}m",
            'steps': f"{step}",
        }
        for k, t in self.info_keys.items():
            if k in panel:
                t.set_text(f"{k}: {panel[k]}")

        # --- Depth (at ~3Hz) ---
        if self._frame_count % 3 == 0:
            self._draw_depth(robot_x, robot_y, robot_theta)

        # --- Flush ---
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass  # window closed

    def _draw_costmap(self, costmap_data):
        if costmap_data is None:
            return
        if self.costmap_img is not None:
            try: self.costmap_img.remove()
            except Exception: pass
        data = np.array(costmap_data.data).reshape(
            costmap_data.height, costmap_data.width).astype(np.float32)
        H, W = costmap_data.height, costmap_data.width
        rgba = np.zeros((H, W, 4), dtype=np.float32)
        occ = data >= 200
        inf = (data >= 50) & (data < 200)
        rgba[inf, 0] = 1.0; rgba[inf, 1] = 0.6
        rgba[inf, 2] = 0.0; rgba[inf, 3] = 0.25
        rgba[occ, 0] = 0.7; rgba[occ, 1] = 0.1
        rgba[occ, 2] = 0.1; rgba[occ, 3] = 0.45
        extent = [costmap_data.origin_x,
                  costmap_data.origin_x + W * costmap_data.resolution,
                  costmap_data.origin_y,
                  costmap_data.origin_y + H * costmap_data.resolution]
        self.costmap_img = self.ax_map.imshow(
            rgba, extent=extent, origin='lower',
            interpolation='nearest', zorder=1, aspect='auto')

    def _draw_depth(self, rx, ry, rtheta):
        self._depth_frame += 1
        if self._depth_frame % 5 != 0 and self._depth_cache is not None:
            return

        angles = np.linspace(-85, 85, 45)
        depths, depths_true, hits = [], [], []
        for a in np.radians(angles):
            wa = rtheta + a
            md = 5.5
            for s in np.arange(0.15, 5.5, 0.15):
                if self.world.is_collision(rx + s*math.cos(wa),
                                           ry + s*math.sin(wa), 0.05):
                    md = s; break
            depths.append(md + np.random.normal(0, max(0.01, md*0.01)))
            depths_true.append(md)

        self.depth_line.set_data(angles, depths)
        self.depth_true_line.set_data(angles, depths_true)

    def finalize(self, success):
        status = 'ARRIVED' if success else 'FAILED'
        self.ax_map.set_title(f'Navigation — [{status}]', fontsize=14,
                              fontweight='bold', color='green' if success else 'red')
        self.fig.canvas.draw_idle()
        plt.pause(3)
