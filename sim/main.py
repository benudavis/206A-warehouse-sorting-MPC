import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter


if __name__ == "__main__":
    # load mujoco model files
    scene = mujoco.MjSpec.from_file('sim/models/scene.xml')
    arm_spec = mujoco.MjSpec.from_file('sim/models/universal_robots_ur5e/ur5e.xml')
    hand_spec = mujoco.MjSpec.from_file('sim/models/robotiq_2f85/2f85.xml')
    
    # attach hand
    attachment_site = arm_spec.site('attachment_site')
    attachment_site.attach_body(hand_spec.worldbody, "hand_", "")

    # merge arm and hand into scene
    robot_site = scene.site('robot_site')
    robot_site.attach_body(arm_spec.worldbody, "arm_", "")
    
    # add cube object
    obj_spec = mujoco.MjSpec.from_file('sim/models/cube.xml')
    obj_frame = scene.worldbody.add_frame(pos=[0, -0.6, 0.5])
    obj_body = obj_frame.attach_body(obj_spec.worldbody, "obj_", "")
    obj_body.add_freejoint(name='obj_freejoint')

    model = scene.compile()
    model.opt.timestep = 0.0001 # physics timestep
    ctrl_rate = 0.01 # control and visualization timestep
    rate = RateLimiter(frequency=1/ctrl_rate, warn=False) # loop rate limiter, enforces real-time sim (otherwise sim runs as fast as possible)
    data = mujoco.MjData(model)

    # reverse shoulder joint in keyframe
    model.key_qpos[0][model.jnt('arm_shoulder_pan_joint').qposadr] += np.pi
    model.key_ctrl[0][model.jnt('arm_shoulder_pan_joint').dofadr] += np.pi
    mujoco.mj_resetDataKeyframe(model, data, 0)


    # example offline trajectory of joints
    n_seconds = 5
    n_steps = int(n_seconds / ctrl_rate)
    traj_qpos = np.zeros((n_steps, 6)) # 6DOF arm
    traj_qpos[0, :] = model.key_qpos[0][:6].copy()
    for i in range(1, n_steps):
        if i < n_steps // 4:
            traj_qpos[i, :2] = traj_qpos[i-1, :2] + np.array([0.001, 0.001])
            traj_qpos[i, 2:] = traj_qpos[i-1, 2:]
        elif i < n_steps // 2:
            traj_qpos[i, :4] = traj_qpos[i-1, :4] + np.array([0.001, -0.001, 0.001, -0.001])
            traj_qpos[i, 4:] = traj_qpos[i-1, 4:]
        else:
            traj_qpos[i, :6] = traj_qpos[i-1, :6] + np.array([-0.001, 0.001, -0.001, 0.001, -0.001, 0.001])

    counter = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():  ### this loop should run at ctrl/visualization rate ###

            # get current step in trajectory
            if counter == n_steps:
                # 6th index is for gripper, ranges from 0 to 255
                data.ctrl[6] = 255
                counter = 0
            elif counter == n_steps // 2:
                # open gripper
                data.ctrl[6] = 0
            
            # set joint commands (first 6 elements of ctrl) to desired joint positions (from traj)
            data.ctrl[:6] = traj_qpos[counter, :]
            counter += 1

            mujoco.mj_step(model, data, nstep=int(ctrl_rate / model.opt.timestep)) # step physics until next ctrl step
            viewer.sync()

            rate.sleep()