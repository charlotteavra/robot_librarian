from bullet_scene_interface import BulletSceneInterface
from bodies.panda import Panda
from bookshelf import Bookshelf
from astar import AStar
import numpy as np
import time
import pybullet as p


def run_trials(robot_, bookshelf, num_trials=50):
    """
    Runs trials with randomly configured boxes
    """
    # Robot joint angles to return to
    robot_ja_init = robot_.get_joint_angles(robot_.controllable_joints)

    box0_configs = []
    box1_configs = []
    planning_times = []
    success = []
    for _ in range(num_trials):
        np.random.seed(int(time.time()))
        bookshelf.create_boxes()
        robot_.env.step_simulation(steps=50)

        # Save box configurations
        box0_configs.append(p.getBasePositionAndOrientation(bookshelf.boxes[0].body))
        box1_configs.append(p.getBasePositionAndOrientation(bookshelf.boxes[1].body))

        # Create planner and search push poses
        time_init = time.time()
        map, normal = bookshelf._discretize_surface(env)
        planner = AStar(robot, map, normal, bookshelf)
        goal = planner.plan(start_idx=(0, 0))
        planning_times.append(time.time() - time_init)

        if goal is not None:
            best_push_ee_pose = (goal[0], goal[1])
            joint_angles = goal[2]
            success.append(1)
            bookshelf.push_box(robot, best_push_ee_pose, normal, joint_angles)

        # delete boxes and return robot to original position
        robot_.env.clear_visual_item(bookshelf.boxes)
        robot_.control(robot_ja_init, set_instantly=True)
        robot_.set_gripper_position(
            robot_.gripper_home_angles_, set_instantly=True
        )  # close gripper

    success_rate = (len(success) / num_trials) * 100

    # np.save(file="box0_configs.npy", arr=np.asarray(box0_configs))
    # np.save(file="box1_configs.npy", arr=np.asarray(box1_configs))
    np.save(file="planning_times.npy", arr=np.asarray(planning_times))
    np.save(file="success_rate.npy", arr=np.asarray(success_rate))


if __name__ == "__main__":
    # Setup environment
    env = BulletSceneInterface(render=True, shadows=True)
    env.info["table_id"] = [1]
    env.info["robot_id"] = -1
    env.reset()
    env.visualize_coordinate_frame()

    # Create Panda manipulator
    robot = Panda(env, position=[0.0, 0, 0.76])

    # Create Bookshelf
    bookshelf = Bookshelf(env, robot)

    # Run trials
    run_trials(robot, bookshelf, num_trials=50)

    print("============ COMPLETE ============")
    while True:
        env.step_simulation(steps=100, realtime=True)
