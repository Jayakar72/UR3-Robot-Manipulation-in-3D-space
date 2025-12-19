# This code contains code blocks for fully functional UR3 robot of (6 DOF)
# 1. calculating and devising the Robot taskspace and workspace
# 2. Buidling the robot - DH parameters, Robot Forward Kinematics, End effector analysis, Trajectory following
# 3. Visualization include The robots Home pose, Endeffectors reach of workspace
# 4. 3D animations include robot following desired path( Hexagon, star etc.) and plot of acc and vel of joints
# 5. Interactive mode using matploltlib
# Need only two libraries for this - numpy=="1.24.4" and matplotlib=="3.7.5" | works on later versions too!!

# SImulation of UR3 robot
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
# Using FuncAnimation for simulating the robot in 3d space and Interaction with the robot
from matplotlib.animation import FuncAnimation 
from matplotlib.widgets import Button


AX_LIM = 1.0
FLOOR_Z = -0.10 # to set the floor plane for the robot (which is Z plane)

# I have taken these UR3 DH parameters from the  - https://www.universal-robots.com/ & https://ieeexplore.ieee.org/document/10335347
a = np.array([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0])
d = np.array([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819])
alpha = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])


# Robots configuration for every joint
qmin = np.deg2rad([-180, -180, -180, -180, -180, -180]) 
qmax = np.deg2rad([+180, +180, +180, +180, +180, +180])

# This creats a DH Transform function 
def T_dh(a_i, alpha_i, d_i, theta_i):
    ca, sa = np.cos(alpha_i), np.sin(alpha_i)
    ct, st = np.cos(theta_i), np.sin(theta_i)
    return np.array([
        [ct, -st*ca,  st*sa, a_i*ct],
        [st,  ct*ca, -ct*sa, a_i*st],   # makes a 4 x 4 homogeneous transform
        [0.,     sa,     ca,    d_i],
        [0.,    0.,    0.,     1.0]
    ])

# Only for the end effector - from section 2.2 Forward Kinematics Formulation
def fk(q): 
    T = np.eye(4) 

    for joint in range(6):
        T = T @ T_dh(a[joint], alpha[joint], d[joint], q[joint])  
    return T

# for all frames - We need this for Jacobian (Bascially returns a list - T_(0,0, T_(0,1),....,T_(0,6))
def fk_all(q): 
    T = np.eye(4)
    Ts = [T.copy()]  
    for joint in range(6):
        T = T @ T_dh(a[joint], alpha[joint], d[joint], q[joint])
        Ts.append(T.copy())
    return Ts


#Geometric Jacobian - This computes the Jacobian for all the 6R arm.
def jacobian(q): 
    Range = 6 
    T = np.eye(4)

    jo_list = [T[:3, 3].copy()]  # Joint origin  
    ja_list = [T[:3, 2].copy()]  # Joint axis

    for joint in range(Range):
        T = T @ T_dh(a[joint], alpha[joint], d[joint], q[joint])
        jo_list.append(T[:3, 3].copy())
        ja_list.append(T[:3, 2].copy())

    jo_end = jo_list[-1]

    Jac = np.zeros((6, 6))
    for joint in range(Range):
        ja = ja_list[joint]     
        jo = jo_list[joint]    
        Jac[:3, joint] = np.cross(ja, (jo_end - jo)) 
        Jac[3:, joint] = ja                         

    return Jac


# helps us find the Orientation errors | Basically used to covert the rotation error into a 3D vector for Newtok IK
def so3_log(R): 
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)

    if np.isclose(theta, 0.0):
        return np.zeros(3)

    W = (R - R.T) / (2.0*np.sin(theta))
    return theta * np.array([W[2, 1], W[0, 2], W[1, 0]])


# Newton / Damped least squres inverse kinematics | 
# It takes the desired pose and the current pose and computes the errors then solves for joint update using damped least square
# shown in section 3.2 Newton-Based Inverse Kinematics
def ik_newton(T_goal, q0, 
              iters=80,
              tol_p=1e-4,
              tol_o=1e-3,
              ori_weight=0.3,
              step=0.7):
    q = q0.copy()

    for _ in range(iters):
        T = fk(q)
        p = T[:3, 3]
        R = T[:3, :3]

        p_des = T_goal[:3, 3]
        R_des = T_goal[:3, :3]

        ep = (p_des - p)
        eo = so3_log(R_des @ R.T)

        if np.linalg.norm(ep) < tol_p and np.linalg.norm(eo) < tol_o:
            return q, True

        e = np.hstack([ep, ori_weight * eo])

        J = jacobian(q)
        lam = 1e-4
        dq = J.T @ np.linalg.solve(J @ J.T + lam*np.eye(6), e)

        q = q + step * dq
        q = np.clip(q, qmin, qmax)

    # if it doesn't converge, still return last q
    return q, False

# The paths/Trajectories are drawn/defined in the taskspace 
def make_infinity_xy(c, N=800, r=0.08):
    cx, cy, cz = c
    t = np.linspace(0, 2*np.pi, N)
    x = cx + r*np.sin(t)
    y = cy + r*np.sin(t)*np.cos(t)
    z = np.full_like(x, cz)
    return np.c_[x, y, z]


def make_hex_xy(c, per_edge=60, r=0.14):
    cx, cy, cz = c
    verts = []
    for k in range(6):
        ang = 2*np.pi*k/6
        verts.append(np.array([cx + r*np.cos(ang), cy + r*np.sin(ang), cz]))
    verts.append(verts[0].copy())

    pts = []
    for i in range(6):
        p0, p1 = verts[i], verts[i+1]
        for s in np.linspace(0, 1, per_edge, endpoint=False):
            pts.append((1-s)*p0 + s*p1)
    pts.append(verts[0])
    return np.array(pts)


def make_star_xy(c, per_edge=25, r_out=0.18, r_in=0.09):
    cx, cy, cz = c
    verts = []
    for k in range(10):
        ang = k*np.pi/5
        r = r_out if (k % 2 == 0) else r_in
        verts.append(np.array([cx + r*np.cos(ang), cy + r*np.sin(ang), cz]))
    verts.append(verts[0].copy())

    pts = []
    for i in range(len(verts)-1):
        p0, p1 = verts[i], verts[i+1]
        for s in np.linspace(0, 1, per_edge, endpoint=False):
            pts.append((1-s)*p0 + s*p1)
    pts.append(verts[0])
    return np.array(pts)


def make_ellipse_3d(c, N=800, a_=0.12, b_=0.06, z_amp=0.06):
    cx, cy, cz = c
    t = np.linspace(0, 2*np.pi, N)
    x = cx + a_*np.cos(t)
    y = cy + b_*np.sin(t)
    z = cz + z_amp*np.sin(t)
    return np.c_[x, y, z]


def solve_path(points, R_fixed, q_start):
    qs = []
    q = q_start.copy()
    bad = 0

    for p in points:
        T_goal = np.eye(4)
        T_goal[:3, :3] = R_fixed
        T_goal[:3, 3] = p

        q, ok = ik_newton(T_goal, q)
        if not ok:
            bad += 1
        qs.append(q.copy())

    if bad > 0:
        print(f"IK didn't fully converge at {bad}/{len(points)} points (still animating).")

    return np.array(qs)


# 3D Geometry of the Robots End effector 
# Drawing the RGB frame the SE(3) pose - Visualized in Fig 2. 
def setup_ax(ax, lim=AX_LIM):
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(FLOOR_Z, lim)
    ax.view_init(elev=30, azim=-45)


def draw_frame(ax, T, L=0.05):
    o = T[:3, 3]
    X = o + L*T[:3, 0]
    Y = o + L*T[:3, 1]
    Z = o + L*T[:3, 2]
    ax.plot([o[0], X[0]], [o[1], X[1]], [o[2], X[2]], 'r-')
    ax.plot([o[0], Y[0]], [o[1], Y[1]], [o[2], Y[2]], 'g-')
    ax.plot([o[0], Z[0]], [o[1], Z[1]], [o[2], Z[2]], 'b-')


def draw_frame_labeled(ax, T, L=0.06):
    o = T[:3, 3]
    axes = ['x', 'y', 'z']
    cols = ['r', 'g', 'b']
    for i in range(3):
        end = o + L*T[:3, i]
        ax.plot([o[0], end[0]], [o[1], end[1]], [o[2], end[2]], cols[i] + '-')
        ax.text(end[0] + 0.01, end[1] + 0.01, end[2] + 0.01, axes[i], color=cols[i], fontsize=8)


# 3D Geometry of the Robot| Making the cylinder as links and Spheres as joints and a base cylinder
def draw_sphere(ax, c, r=0.02, color='r', n=16):
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    U, V = np.meshgrid(u, v)
    x = c[0] + r*np.cos(U)*np.sin(V)
    y = c[1] + r*np.sin(U)*np.sin(V)
    z = c[2] + r*np.cos(V)
    ax.plot_surface(x, y, z, color=color, linewidth=0, shade=True)


def R_from_z(u):
    u = u / np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(u, z):
        return np.eye(3)
    if np.allclose(u, -z):
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=float)

    axis = np.cross(z, u)
    s = np.linalg.norm(axis)
    axis = axis / s
    c = float(np.dot(z, u))

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)

    # Rodrigues 
    return c*np.eye(3) + (1-c)*np.outer(axis, axis) + s*K


def draw_cylinder(ax, p0, p1, r=0.02, color='b', n_th=16, n_z=8):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    v = p1 - p0
    L = np.linalg.norm(v)
    if L < 1e-9:
        return

    u = v / L
    R = R_from_z(u)

    th = np.linspace(0, 2*np.pi, n_th)
    zz = np.linspace(0, L, n_z)
    TH, ZZ = np.meshgrid(th, zz)

    x0 = r*np.cos(TH)
    y0 = r*np.sin(TH)
    z0 = ZZ

    pts = np.vstack([x0.ravel(), y0.ravel(), z0.ravel()])
    pts = R @ pts

    X = pts[0].reshape(ZZ.shape) + p0[0]
    Y = pts[1].reshape(ZZ.shape) + p0[1]
    Z = pts[2].reshape(ZZ.shape) + p0[2]

    ax.plot_surface(X, Y, Z, color=color, linewidth=0, shade=True)


def draw_base(ax, radius=0.08, height=0.05, color='0.5'):
    zb = FLOOR_Z
    zt = FLOOR_Z + height

    # side
    draw_cylinder(ax, [0, 0, zb], [0, 0, zt], r=radius, color=color)

    # disks
    th = np.linspace(0, 2*np.pi, 40)
    rr = np.linspace(0, radius, 10)
    RR, TH = np.meshgrid(rr, th)
    X = RR*np.cos(TH)
    Y = RR*np.sin(TH)
    ax.plot_surface(X, Y, np.full_like(X, zb), color=color, linewidth=0, shade=True)
    ax.plot_surface(X, Y, np.full_like(X, zt), color=color, linewidth=0, shade=True)


# Visualizing the workspace - Section 4.2 Visualization and workspace Analysis
def show_workspace(samples=2000): # took randomly generated 2000 joint configurations
    qs = np.random.uniform(qmin, qmax, size=(samples, 6))
    pts = np.array([fk(q)[:3, 3] for q in qs])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    setup_ax(ax)
    draw_base(ax)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=4, c='b', alpha=0.4)

    # shows an example robot pose
    q0 = np.deg2rad([0, -90, 90, -90, -90, 0])
    Ts = fk_all(q0)
    joints = np.array([T[:3, 3] for T in Ts])

    link_r = 0.02
    joint_r = 0.025
    ee_r = 0.015

    for i in range(len(joints)-1):
        draw_cylinder(ax, joints[i], joints[i+1], r=link_r, color='b')
        draw_sphere(ax, joints[i], r=joint_r, color='r')

    draw_sphere(ax, joints[-1], r=ee_r, color='y')

    for T in Ts[:-1]:
        draw_frame(ax, T, L=0.04)
    draw_frame_labeled(ax, Ts[-1], L=0.06)

    ax.set_title("UR3 EE workspace (random sampling)")
    plt.tight_layout()
    plt.show()


# We also visualize the Joint's Speed and Accleration plots|
# We compute the velocity and Acceleration using finite difference
def joint_kinematics(qtraj, dt):
    qtraj = np.asarray(qtraj)
    N = qtraj.shape[0]
    t = np.arange(N) * dt

    dq = np.zeros_like(qtraj)
    ddq = np.zeros_like(qtraj)

    dq[1:-1] = (qtraj[2:] - qtraj[:-2]) / (2*dt)
    dq[0] = (qtraj[1] - qtraj[0]) / dt
    dq[-1] = (qtraj[-1] - qtraj[-2]) / dt

    ddq[1:-1] = (dq[2:] - dq[:-2]) / (2*dt)
    ddq[0] = (dq[1] - dq[0]) / dt
    ddq[-1] = (dq[-1] - dq[-2]) / dt

    return t, dq, ddq

# this plots the graphs for all the joint(6)
def make_joint_plot(qtraj, dt, title=""):
    t, dq, ddq = joint_kinematics(qtraj, dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    labels = [f"Link_{i+1}" for i in range(qtraj.shape[1])]

    for j in range(qtraj.shape[1]):
        ax1.plot(t, dq[:, j], label=labels[j])
    ax1.set_title("speed")
    ax1.set_xlabel("time [s]")
    ax1.set_ylabel("rad/s")
    ax1.grid(True)
    ax1.legend()

    for j in range(qtraj.shape[1]):
        ax2.plot(t, ddq[:, j], label=labels[j])
    ax2.set_title("acceleration")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("rad/s^2")
    ax2.grid(True)
    ax2.legend()

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    return fig

#This function helps us creat the 3d plot and helps us animate the robot configuration frame by frame
def animate(qtraj, path_xyz, title="trajectory", dt=0.03, interval_ms=30, show_joint_plots=True):
    path_xyz = np.asarray(path_xyz)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    setup_ax(ax)
    draw_base(ax)
    fig.suptitle(title)

    
    if show_joint_plots:
        _ = make_joint_plot(qtraj, dt, title=title)

    link_r = 0.02
    joint_r = 0.025
    ee_r = 0.015

    def update(k):
        
        elev, azim = ax.elev, ax.azim
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()

        ax.cla()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        draw_base(ax)
        ax.set_title(title)

        
        ax.plot(path_xyz[:, 0], path_xyz[:, 1], path_xyz[:, 2], linestyle=":", color="0.6", linewidth=1)
        ax.plot(path_xyz[:k+1, 0], path_xyz[:k+1, 1], path_xyz[:k+1, 2], "-", color="b", linewidth=2)

        Ts = fk_all(qtraj[k])
        joints = np.array([T[:3, 3] for T in Ts])

        for i in range(len(joints)-1):
            draw_cylinder(ax, joints[i], joints[i+1], r=link_r, color='b')
            draw_sphere(ax, joints[i], r=joint_r, color='r')

        draw_sphere(ax, joints[-1], r=ee_r, color='y')

        for T in Ts[:-1]:
            draw_frame(ax, T, L=0.04)
        draw_frame_labeled(ax, Ts[-1], L=0.06)

        return []

    _ani = FuncAnimation(fig, update, frames=len(qtraj), interval=interval_ms, blit=False)
    plt.show()


# This is a very interesting functions - Used in interactive plot
# creates the rotation steps and the buttons and it runs IK after each click and redraws the robot
def rot_about(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s,  c]])
    if axis == 'y':
        return np.array([[ c, 0, s],
                         [ 0, 1, 0],
                         [-s, 0, c]])
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def teleop(q_init=None, step_p=0.01, step_deg=5):
    if q_init is None:
        q_init = np.deg2rad([0, -90, 90, -90, -90, 0])

    q = np.array(q_init, dtype=float)
    T_goal = fk(q)
    step_ang = np.deg2rad(step_deg)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.05, right=0.65, bottom=0.05, top=0.95)

    link_r = 0.02
    joint_r = 0.025
    ee_r = 0.015

    def draw():
        ax.cla()
        setup_ax(ax)
        draw_base(ax)

        Ts = fk_all(q)
        joints = np.array([T[:3, 3] for T in Ts])

        for i in range(len(joints)-1):
            draw_cylinder(ax, joints[i], joints[i+1], r=link_r, color='b')
            draw_sphere(ax, joints[i], r=joint_r, color='r')
        draw_sphere(ax, joints[-1], r=ee_r, color='y')

        for T in Ts[:-1]:
            draw_frame(ax, T, L=0.04)
        draw_frame_labeled(ax, Ts[-1], L=0.06)

        p = T_goal[:3, 3]
        ax.set_title(f"teleop\nEE: [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}]")

    def apply(dp=None, raxis=None, sign=1.0):
        nonlocal q, T_goal
        if dp is not None:
            T_goal = T_goal.copy()
            T_goal[:3, 3] += dp
        if raxis is not None:
            Rstep = rot_about(raxis, sign*step_ang)
            T_goal = T_goal.copy()
            T_goal[:3, :3] = Rstep @ T_goal[:3, :3]

        q_new, ok = ik_newton(T_goal, q)
        if not ok:
            print("teleop IK warning: not fully converged")
        q = q_new
        draw()
        fig.canvas.draw_idle()

   
    def x_plus(_):  apply(dp=np.array([ step_p, 0, 0]))
    def x_minus(_): apply(dp=np.array([-step_p, 0, 0]))
    def y_plus(_):  apply(dp=np.array([0,  step_p, 0]))
    def y_minus(_): apply(dp=np.array([0, -step_p, 0]))
    def z_plus(_):  apply(dp=np.array([0, 0,  step_p]))
    def z_minus(_): apply(dp=np.array([0, 0, -step_p]))

    def roll_plus(_):   apply(raxis='x', sign=+1)
    def roll_minus(_):  apply(raxis='x', sign=-1)
    def pitch_plus(_):  apply(raxis='y', sign=+1)
    def pitch_minus(_): apply(raxis='y', sign=-1)
    def yaw_plus(_):    apply(raxis='z', sign=+1)
    def yaw_minus(_):   apply(raxis='z', sign=-1)

    def add_btn(x, y, label, fn):
        axb = plt.axes([x, y, 0.1, 0.05])
        b = Button(axb, label)
        b.on_clicked(fn)
        return b

    
    btns = []
    btns += [add_btn(0.70, 0.75, "X+", x_plus),
             add_btn(0.82, 0.75, "X-", x_minus)]
    btns += [add_btn(0.70, 0.67, "Y+", y_plus),
             add_btn(0.82, 0.67, "Y-", y_minus)]
    btns += [add_btn(0.70, 0.59, "Z+", z_plus),
             add_btn(0.82, 0.59, "Z-", z_minus)]

    btns += [add_btn(0.70, 0.47, "Roll+", roll_plus),
             add_btn(0.82, 0.47, "Roll-", roll_minus)]
    btns += [add_btn(0.70, 0.39, "Pitch+", pitch_plus),
             add_btn(0.82, 0.39, "Pitch-", pitch_minus)]
    btns += [add_btn(0.70, 0.31, "Yaw+", yaw_plus),
             add_btn(0.82, 0.31, "Yaw-", yaw_minus)]

    fig._btn_keepalive = btns

    draw()
    plt.show()

# this is the first plot where we just see the robot only 
def show_robot_only(q=None, title="UR3 robot (home pose)"):
    if q is None:
        q = np.deg2rad([0, -90, 90, -90, -90, 0])

    Ts = fk_all(q)
    joints = np.array([T[:3, 3] for T in Ts])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    setup_ax(ax)
    draw_base(ax)

    link_r = 0.02
    joint_r = 0.025
    ee_r = 0.015

    # links + joints
    for i in range(len(joints) - 1):
        draw_cylinder(ax, joints[i], joints[i + 1], r=link_r, color='b')
        draw_sphere(ax, joints[i], r=joint_r, color='r')

    # EE Marking
    draw_sphere(ax, joints[-1], r=ee_r, color='y')

    
    for T in Ts[:-1]:
        draw_frame(ax, T, L=0.04)
    draw_frame_labeled(ax, Ts[-1], L=0.06)

    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# you can experiment with the interactive plot, if it goes over constrained we see the error - 
# IK didn't fully converge at {a specifi point} points (still animating). You can still continue you wont break anything ;)
def main():
    q_home = np.deg2rad([0, -90, 0, -90, -90, 0]) # This pose can be changed when you want to experiment with the robot position
    show_robot_only(q_home)
    show_workspace(samples=2000) #These are some example samples that i have taken for the EE workspace
    
    q_home = np.deg2rad([0, -90, 90, -90, -90, 0])
    T_home = fk(q_home)
    center = T_home[:3, 3]
    R_fixed = T_home[:3, :3]

    print("\nPick one: (Press Q to close window) or (close the window after done to next part)")
    print("  1 - infinity loop Path(XY)")
    print("  2 - hexagon Path(XY)")
    print("  3 - star Path(XY)")
    print("  4 - 3D ellipse Path")
    print("  5 - running all Trajectories at once")
    print("  6 - Interactive mode for controling the robot")
    sel = input("choice: ").strip()

    dt = 0.06 # this parameter is used for time frame you can increase of decrease it to make robot fast or slow

    if sel in ("1", "5"):
        pts = make_infinity_xy(center, N=800, r=0.08)
        qtraj = solve_path(pts, R_fixed, q_home)
        animate(qtraj, pts, title="UR3 - infinity", dt=dt, interval_ms=120, show_joint_plots=True)

    if sel in ("2", "5"):
        pts = make_hex_xy(center, per_edge=60, r=0.14)
        qtraj = solve_path(pts, R_fixed, q_home)
        animate(qtraj, pts, title="UR3 - hexagon", dt=dt, interval_ms=120, show_joint_plots=True)

    if sel in ("3", "5"):
        pts = make_star_xy(center, per_edge=25, r_out=0.18, r_in=0.09)
        qtraj = solve_path(pts, R_fixed, q_home)
        animate(qtraj, pts, title="UR3 - star", dt=dt, interval_ms=120, show_joint_plots=True)

    if sel in ("4", "5"):
        pts = make_ellipse_3d(center, N=800, a_=0.12, b_=0.06, z_amp=0.06)
        qtraj = solve_path(pts, R_fixed, q_home)
        animate(qtraj, pts, title="UR3 - 3D ellipse", dt=dt, interval_ms=120, show_joint_plots=True)

    if sel == "6":
        teleop(q_init=q_home)


if __name__ == "__main__":
    main()
