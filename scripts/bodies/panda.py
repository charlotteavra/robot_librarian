import os
import pybullet as p
import pybullet_data
import numpy as np
from .robot import Robot


class Panda(Robot):
    def __init__(self, env,
                 position=(0, 0, 0),
                 orientation=(0, 0, 0, 1),
                 controllable_joints=None,
                 fixed_base=True):
        controllable_joints = [0, 1, 2, 3, 4, 5, 6] if controllable_joints is None else controllable_joints
        end_effector = 11  # Used to get the pose of the end effector
        gripper_joints = [9, 10]  # Gripper actuated joints

        # body = env.sim.loadURDF(os.path.join(env.directory, 'panda', 'panda.urdf'), useFixedBase=fixed_base,
        #                         basePosition=position, baseOrientation=orientation)
        body = env.sim.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
                                useFixedBase=fixed_base,
                                basePosition=position,
                                baseOrientation=orientation)
        super().__init__(body, env, controllable_joints, end_effector, gripper_joints)

        self.groups_ = {"arm": controllable_joints,
                        "gripper": gripper_joints}
        self.arm_home_angles_ = [0, -np.pi / 6, 0, -3 * np.pi / 4, 0, 3 * np.pi / 5, np.pi / 4]
        # [-1.39, -1.83, -1.42, -2.27, -1.68, 1.29, -1.50]
        self.gripper_home_angles_ = [0.0] * 2
        self.reset_joints()
        # Close gripper
        self.set_gripper_position(self.gripper_home_angles_, set_instantly=True)

    def reset_joints(self):
        # self.set_joint_angles([-1.39, -1.83, -1.42, -2.27, -1.68, 1.29, -1.50, 0, 0, 0, 0, 0])
        self.set_joint_angles([0, -np.pi / 6, 0, -3 * np.pi / 4, 0, 3 * np.pi / 5, np.pi / 4, 0.4, 0.4])

    def reset_group_joints(self, group_name):
        if group_name in self.groups_:
            if group_name == "arm":
                self.set_joint_angles(self.arm_home_angles_, joints=self.groups_[group_name])
            elif group_name == "gripper":
                self.set_joint_angles(self.gripper_home_angles_, joints=self.groups_[group_name])
        else:
            print("Group name not found")
