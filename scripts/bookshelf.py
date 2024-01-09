import objects_handler as oh
import numpy as np
from utils import *
from bullet_scene_interface import Camera
import pybullet as p
import open3d as o3d
import time
import copy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


class Bookshelf:
    def __init__(self, env, robot):
        self.env = env
        self.robot = robot

        self.table_obj = oh.URDF(
            self.env,
            os.path.join(assets_directory, "table", "table.urdf"),
            position=[0, 0, 0],
            orientation=[0, 0, 0, 1],
        )
        self.bookshelf_pos = [0.637, 0.51, 1.06]
        self.bookshelf_orn = p.getQuaternionFromEuler([-np.pi / 2, 0, np.pi])
        self.bookshelf_obj = oh.URDF(
            self.env,
            os.path.join(assets_directory, "bookshelf", "bookshelf.urdf"),
            static=True,
            position=self.bookshelf_pos,
            orientation=self.bookshelf_orn,
        )
        self.obstacles = [self.table_obj, self.bookshelf_obj]
        self.box_size = np.array([0.051, 0.185, 0.118])
        self.box_initial_pos = (
            (0.3, 0.32, 0.9),
            (0.3 - (self.box_size[0] * 2), 0.32, 0.9),
            (0.4, -0.25, 1.0),
        )
        self.box_grasp_pose = [0.5, -0.22, 0.85]
        self.box_grasp_orientation = (
            0.7071067811865476,
            0.7071067811865476,
            4.329780281177467e-17,
            4.329780281177467e-17,
        )
        self.box_positions = ((0.28, 0.11, 0.99), (0.18, 0.11, 0.99))

    def create_boxes(self):
        """Create boxes to grasp and place on bookshelf"""
        boxes = []
        box_initial_pos = self.box_initial_pos

        roll = (
            np.random.uniform(-np.pi / 4, -np.pi / 6)
            if np.random.choice([True, False])
            else np.random.uniform(np.pi / 6, np.pi / 4)
        )  # random angle between (-pi/3, -pi/6) or (pi/6, pi/3)
        # yaw = np.random.uniform(-np.pi / 6, np.pi / 6)

        box_initial_orn = (
            get_quaternion([np.pi / 2, 0, 0], self.env),
            get_quaternion([np.pi / 2, roll, 0], self.env),
            get_quaternion([0, 0, 0], self.env),
        )

        for i in range(3):
            boxes.append(
                oh.Shape(
                    self.env,
                    oh.Box(half_extents=self.box_size / 2),
                    static=False,
                    mass=0.5,
                    position=box_initial_pos[i],
                    orientation=box_initial_orn[i],
                    rgba=[np.random.uniform(0, 1) for _ in range(3)] + [1],
                )
            )
        boxes[-1].set_whole_body_frictions(
            lateral_friction=1.0, spinning_friction=1.0, rolling_friction=0
        )
        self.obstacles += boxes
        self.boxes = boxes

    def move_to_start_pose(self, robot_):
        joint_target = robot_.arm_home_angles_
        self.moveto(robot_, joint_angles=joint_target)
        self.robot.set_gripper_position([1] * 2, set_instantly=True)  # open gripper

    def _robot_in_collision(self, q, robot_, obstacles):
        """
        Checks for collision

        Args:
            q (NDArray): Robot joint angles
            robot_ (Robot): Robot body object

        Returns:
            bool: False if not in collision, True otherwise
        """

        prev_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joints)
        robot_.control(q, set_instantly=True)
        for obstacle in obstacles:
            if len(robot_.get_closest_points(obstacle, distance=0)[-1]) != 0:
                robot_.control(prev_joint_angles, set_instantly=True)
                return True

        robot_.control(prev_joint_angles, set_instantly=True)
        return False

    def _execute_push(self, robot_, ee_pose, normal, realTime: bool = True):
        """
        Goes to ee pose and moves along y in the direction opposite that
        of the normal

        Args:
            robot_ (Robot): Robot body object
            ee_pose (tuple): Position, orientation
            normal (NP.Array): Array of size (3,) with x,y,z normal at position
            realTime (bool):
        """
        translation = 0.1
        time_init = int(time.time())
        while int(time.time()) - time_init < 1:
            cur_ee_pose = (
                ee_pose[0] + np.array([-translation * np.sign(normal[0]), 0, 0.01]),
                ee_pose[1],
            )
            self.moveto(robot_, ee_pose=cur_ee_pose, forces=40)
            robot_.env.step_simulation(steps=100, realtime=realTime)
        self.moveto(robot_, ee_pose)
        robot_.env.step_simulation(steps=200, realtime=realTime)

    def push_box(self, robot_, push_ee_pose, normal, joint_angles):
        """
        Args:
            robot_ (Robot): Robot body object
            best_push_ee_pose (tuple[pose, orientation]): Tuple with ee_pose and ee_orientation
            joint_angles (): TODO
        """
        # MOVE to front of bookshelf
        robot_.set_gripper_position(robot_.gripper_home_angles_)  # close gripper
        robot_.env.step_simulation(steps=100, realtime=True)
        self.moveto(robot_, ee_pose=(push_ee_pose[0] - [0, 0.1, 0], push_ee_pose[1]))
        robot_.env.step_simulation(steps=200, realtime=False)

        # MOVE to best push pose and push
        robot_.control(joint_angles)
        # self.moveto(robot_, ee_pose=push_ee_pose)
        robot_.env.step_simulation(steps=100, realtime=False)
        self._execute_push(robot_, push_ee_pose, normal, realTime=False)

        # MOVE out of bookshelf and return to home
        curr_pos, curr_ori = robot_.get_link_pos_orient(robot_.end_effector)
        self.moveto(robot_, ee_pose=(curr_pos - [0, 0.1, 0], curr_ori))
        self.move_to_start_pose(robot_)

    def run_trials(self, robot_):
        """
        Runs 50 trials with randomly configured boxes
        """
        robot_ja_init = robot_.get_joint_angles(robot_.controllable_joints)

        box_configs = 50
        box0_configs = []
        box1_configs = []
        planning_times = []
        success_rates = []
        for _ in range(box_configs):
            np.random.seed(int(time.time()))
            self.create_boxes()
            robot_.env.step_simulation(steps=50)

            box0_configs.append(p.getBasePositionAndOrientation(self.boxes[0].body))
            box1_configs.append(p.getBasePositionAndOrientation(self.boxes[1].body))

            time_init = time.time()
            best_push_ee_pose, normal, scores = self._sample_push_poses(robot_)
            planning_times.append(time.time() - time_init)

            success = [value for value in scores if value > 0.95]
            success_rate = (len(success) / len(scores)) * 100
            success_rates.append(success_rate)

            # delete boxes and return robot to original position
            robot_.env.clear_visual_item(self.boxes)
            robot_.control(robot_ja_init, set_instantly=True)
            robot_.set_gripper_position(
                robot_.gripper_home_angles_, set_instantly=True
            )  # close gripper
            # robot_.env.step_simulation(steps=100, realtime=True)

        # np.save(file="box0_configs.npy", arr=np.asarray(box0_configs))
        # np.save(file="box1_configs.npy", arr=np.asarray(box1_configs))
        np.save(file="planning_times.npy", arr=np.asarray(planning_times))
        np.save(file="success_rates.npy", arr=np.asarray(success_rates))

    def _sample_push_poses(self, robot_, num_samples=1000):
        """
        Uninformed sampling approach: sample end-effector poses to push box and compute
        push score for each

        Args:
            robot_ (Robot): Robot body object
            num_samples (int): number of grasp samples

        Returns:
            tuple: ee pose (position, quaternion) of push with highest score

        """
        pcd, normals = self._get_pc_box(robot_.env, idx=1)
        points = np.asarray(pcd.points)

        scores = []
        ee_poses_sampled = []
        normal_vecs = []
        for _ in range(num_samples):
            sample_idx = np.random.uniform(0, points.shape[0])
            sample_point = points[int(sample_idx), :]

            # get normal at sample point and normalize
            normal = normals[int(sample_idx), :]
            normal /= np.linalg.norm(normal)
            normal_vecs.append(normal)

            # offset point from surface to avoid gripper collision
            offset_point = sample_point + 0.03 * normal

            # get push score for offset point
            ee_pose = (offset_point, [-np.pi / 2, 0, 0])
            ee_poses_sampled.append(ee_pose)
            scores.append(self._evaluate_push(robot_, ee_pose, normal))

        best_push_idx = np.argmax(scores)
        best_push_score = scores[best_push_idx]
        best_push_ee_pose = ee_poses_sampled[best_push_idx]
        print("Best push score: ", best_push_score)
        print("Best push ee pose: ", best_push_ee_pose)

        self.move_to_start_pose(robot_)

        return best_push_ee_pose, normal_vecs[best_push_idx], scores

    def _evaluate_push(self, robot_, ee_pose, normal):
        """
        Executes push and determines quality based on difference between
        box orientation and a vertical orientation

        Args:
        """
        # get current joint angles to reset to
        robot_ja_init = robot_.get_joint_angles(robot_.controllable_joints)

        # get joint angles for ee pose and check for IK solution of None or in collision
        robot_joint_angles_ = robot_.ik(
            robot_.end_effector,
            target_pos=ee_pose[0],
            target_orient=ee_pose[1],
            use_current_joint_angles=True,
        )
        if normal[0] < np.pi / 4:  # facing front
            # print("X Front Face")
            return 0
        if robot_joint_angles_ is None or self._robot_in_collision(
            robot_joint_angles_, robot_, self.obstacles
        ):
            # print("X Collision")
            return 0
        print("Valid Point")

        # save current poses for boxes to return to after testing
        box0_pose_init = p.getBasePositionAndOrientation(self.boxes[0].body)
        box1_pose_init = p.getBasePositionAndOrientation(self.boxes[1].body)

        robot_.control(robot_joint_angles_, set_instantly=True)
        self._execute_push(robot_, ee_pose, normal, realTime=False)

        # get final pose of box post-push
        box1_pose_fin = p.getBasePositionAndOrientation(self.boxes[1].body)

        # define score [0,1] as dot product between quaternions
        ideal_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        # score = np.dot(box1_pose_fin[1], ideal_quat)
        score = 1 - (
            (np.arccos((2 * np.dot(ideal_quat, box1_pose_fin[1]) ** 2) - 1)) / np.pi
        )
        print(score)

        # return boxes and robot to original position
        p.resetBasePositionAndOrientation(
            self.boxes[0].body,
            box0_pose_init[0],
            box0_pose_init[1],
        )
        p.resetBasePositionAndOrientation(
            self.boxes[1].body,
            box1_pose_init[0],
            box1_pose_init[1],
        )
        robot_.control(robot_ja_init, set_instantly=True)

        return score

    def _discretize_surface(self, env_):
        """
        Discretizes side surfaces of box into meshgrid of points

        Returns:
            NDArray: Array of size (n,3) with transformed meshgrid points
        """
        half_extents = self.box_size / 2

        x1 = half_extents[0]
        x2 = -half_extents[0]
        y_min, y_max = -half_extents[1], half_extents[1]
        z_min, z_max = -half_extents[2], half_extents[2]

        # Define the number of points in each dimension
        num_points_y = 10
        num_points_z = 10

        # Generate the meshgrid
        y_values = np.linspace(y_min, y_max, num_points_y)
        z_values = np.linspace(z_min, z_max, num_points_z)
        y_mesh, z_mesh = np.meshgrid(y_values, z_values)

        y_coordinates = y_mesh.flatten()
        z_coordinates = z_mesh.flatten()
        x1_coordinates = np.ones(y_coordinates.shape) * x1
        x2_coordinates = np.ones(y_coordinates.shape) * x2

        y_coordinates = np.concatenate([y_coordinates, y_coordinates])
        z_coordinates = np.concatenate([z_coordinates, z_coordinates])
        x_coordinates = np.concatenate([x1_coordinates, x2_coordinates])
        points = np.concatenate(
            [
                x_coordinates.reshape(x_coordinates.size, 1),
                y_coordinates.reshape(y_coordinates.size, 1),
                z_coordinates.reshape(z_coordinates.size, 1),
            ],
            axis=1,
        )

        # Apply transformations
        box_pos, box_orn = p.getBasePositionAndOrientation(self.boxes[1].body)
        r = Rotation.from_matrix(get_rotation_matrix(box_orn, env_))
        points_rotated = r.apply(points)

        points_translated = np.empty(points_rotated.shape)
        points_translated[:, 0] = points_rotated[:, 0] + box_pos[0]
        points_translated[:, 1] = points_rotated[:, 1] + box_pos[1]
        points_translated[:, 2] = points_rotated[:, 2] + box_pos[2]

        # Reshape back to meshgrid
        mesh1 = points_translated[(num_points_y * num_points_z) :, :].reshape(
            num_points_y, num_points_z, 3
        )
        mesh2 = points_translated[: (num_points_y * num_points_z), :].reshape(
            num_points_y, num_points_z, 3
        )
        map = np.zeros((mesh1.shape[0], mesh1.shape[1], 1))
        map1 = np.concatenate([mesh1, map], axis=2)
        map2 = np.concatenate([mesh2, map], axis=2)

        # Calculate normal
        x0, y0, z0 = mesh1[0, 0]
        x1, y1, z1 = mesh1[0, 1]
        x2, y2, z2 = mesh1[1, 0]

        ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]  # first vector
        vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]  # sec vector

        u_cross_v = [
            uy * vz - uz * vy,
            uz * vx - ux * vz,
            ux * vy - uy * vx,
        ]  # cross product

        normal = np.array(u_cross_v)
        normal /= np.linalg.norm(normal)

        # Visualize the transformed meshgrid
        # fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(
        #     map2[:, :, 0],
        #     map2[:, :, 1],
        #     map2[:, :, 2],
        #     marker=".",
        #     color="#27566d",
        # )
        # ax.scatter(
        #     map1[:, :, 0],
        #     map1[:, :, 1],
        #     map1[:, :, 2],
        #     marker=".",
        #     color="#ffa600",
        # )
        # ax.set_xlabel("X-axis")
        # ax.set_ylabel("Y-axis")
        # ax.set_zlabel("Z-axis")
        # ax.axes.set_xlim3d(left=0.1, right=0.4)
        # ax.axes.set_ylim3d(bottom=0.25, top=0.55)
        # ax.axes.set_zlim3d(bottom=0.7, top=1.0)
        # plt.show()

        if get_euler(box_orn, env_)[1] > 0:
            return map2, normal
        else:
            return map1, -normal

    def _get_pc_box_origin(self):
        """
        Returns point cloud of generic box of correct size, centered at the origin

        Returns:
            o3d.geometry.PointCloud
        """
        # create box mesh and sample points on surface
        box_mesh = o3d.geometry.TriangleMesh.create_box(
            width=self.box_size[0], height=self.box_size[1], depth=self.box_size[2]
        )
        half_extents = self.box_size / 2
        centered_box_mesh = copy.deepcopy(box_mesh).translate(-half_extents)

        # voxel_grid = o3d.geometry.VoxelGrid.create_dense(
        #     width=self.box_size[0],
        #     height=self.box_size[1],
        #     depth=self.box_size[2],
        #     voxel_size=0.01,
        #     origin=-half_extents,
        #     color=[1.0, 0.7, 0.0],
        # )
        # o3d.visualization.draw_geometries(
        #     [centered_box_mesh, voxel_grid], mesh_show_wireframe=True
        # )

        # points = []
        # for voxel in voxel_grid.get_voxels():
        #     points.append(voxel_grid.get_voxel_center_coordinate(voxel.grid_index))

        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = o3d.utility.Vector3dVector(points)
        # new_pcd.colors = o3d.utility.Vector3dVector(
        #     np.random.uniform(0, 1, size=(len(points), 3))
        # )

        # convert to point cloud
        n = 1000
        pcd = centered_box_mesh.sample_points_uniformly(number_of_points=n)

        return pcd

    def _get_pc_box(self, env_, idx: int):
        """
        Returns point cloud of box in current configuration

        Args:
            env_ (BulletSceneInterface): Bullet environment object
            idx (int): Box index (first, second, or third box)

        Returns:
            o3d.geometry.PointCloud
        """
        pcd = self._get_pc_box_origin()

        # apply transformations
        box_pos, box_orn = p.getBasePositionAndOrientation(self.boxes[idx].body)
        pcd_translated = copy.deepcopy(pcd).translate(np.asarray(box_pos))
        pcd_rotated = copy.deepcopy(pcd_translated).rotate(
            get_rotation_matrix(box_orn, env_)
        )

        # estimate normals for each point
        pcd_rotated.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        normals = np.asarray(pcd_rotated.normals)
        return pcd_rotated, normals

    def moveto(
        self,
        robot_,
        ee_pose=None,
        joint_angles=None,
        velocity_control=False,
        forces=None,
    ):
        """
        Move robot to a given ee_pose or joint angles. If both are given, ee_pose is used.

        Args:
            robot_ (Robot): Robot body object
            ee_pose (tuple[pose, orientation]): Tuple with ee_pose and ee_orientation
            joint_angles ():
        """
        robot_ = self.robot

        if ee_pose is not None:
            joint_angles = robot_.ik(
                self.robot.end_effector,
                target_pos=ee_pose[0],
                target_orient=ee_pose[1],
                use_current_joint_angles=True,
            )
        if joint_angles is None:
            return False

        robot_.control(joint_angles, forces=forces, velocity_control=velocity_control)
        curr_time = time.time()
        while (
            np.linalg.norm(
                robot_.get_joint_angles(robot_.controllable_joints) - joint_angles
            )
            > 0.065
        ):
            robot_.env.step_simulation(realtime=True)
            if time.time() - curr_time > 5:
                return False
        return True

    def pick_and_place(self, idx: int, random_rotate: bool = False):
        """
        Finds best grasp for book object, moves to grasp, and places object on shelf

        Args:
            idx (int): Box index (first, second, or third box)
            random_rotate (bool): Default False, will not rotate box about Y upon placement
        """
        robot_ = self.robot

        # MOVE to above box
        box_min, box_max = self.boxes[idx].get_AABB()
        above_box_pose = np.copy(self.box_grasp_pose)
        above_box_pose[0] = np.mean([box_min[0], box_max[0]])
        above_box_pose[2] += 0.05
        self.moveto(robot_, ee_pose=(above_box_pose, self.box_grasp_orientation))

        # MOVE to box and grasp
        box_grasp_pose = np.copy(self.box_grasp_pose)
        box_grasp_pose[0] = np.mean([box_min[0], box_max[0]])
        self.moveto(robot_, ee_pose=(box_grasp_pose, self.box_grasp_orientation))
        robot_.set_gripper_position([0] * 2, force=50000)
        robot_.env.step_simulation(steps=100, realtime=False)

        # MOVE upwards to pick
        pos, ori = robot_.get_link_pos_orient(robot_.end_effector)
        self.moveto(robot_, ee_pose=(pos + [0, 0, 0.2], ori))

        # MOVE the box in front of the shelf
        pos = self.box_positions[idx]
        _, curr_ori = robot_.get_link_pos_orient(robot_.end_effector)
        ori = np.array([0, -np.pi / 2, 0])
        curr_ori_euler = self.env.sim.getEulerFromQuaternion(curr_ori)
        new_orn = np.array(curr_ori_euler) + ori
        self.moveto(robot_, ee_pose=(pos, new_orn))

        # MOVE into shelf
        goal_position, goal_orientation = self.robot.get_link_pos_orient(
            robot_.end_effector
        )
        goal_position[1] += 0.2
        self.moveto(robot_, ee_pose=(goal_position, goal_orientation))
        robot_.env.step_simulation(steps=100, realtime=False)

        # MOVE down in shelf to place box
        goal_position, goal_orientation = robot_.get_link_pos_orient(
            robot_.end_effector
        )
        goal_position[2] -= 0.08

        if random_rotate:
            roll = (
                np.random.uniform(-np.pi / 3, -np.pi / 6)
                if np.random.choice([True, False])
                else np.random.uniform(np.pi / 6, np.pi / 3)
            )  # random angle between (-pi/3, -pi/6) or (pi/6, pi/3)
            yaw = np.random.uniform(-np.pi / 6, np.pi / 6)
            cur_euler = get_euler(goal_orientation, robot_.env)
            cur_euler[2] += yaw
            joint_angles = robot_.ik(
                self.robot.end_effector,
                target_pos=goal_position,
                target_orient=get_quaternion(cur_euler, robot_.env),
                use_current_joint_angles=True,
            )
            joint_angles[-1] += roll  # apply roll directly to J7 (wrist)
            self.moveto(robot_, joint_angles=joint_angles)
        else:
            self.moveto(robot_, ee_pose=(goal_position, goal_orientation))
        robot_.env.step_simulation(steps=100, realtime=False)

        # OPEN gripper
        robot_.set_gripper_position([1] * 2, force=5000)
        robot_.env.step_simulation(steps=100, realtime=False)

        # MOVE out of shelf
        goal_position, goal_orientation = robot_.get_link_pos_orient(
            robot_.end_effector
        )
        goal_position[1] -= 0.15
        self.moveto(robot_, ee_pose=(goal_position, goal_orientation))
        robot_.env.step_simulation(steps=100, realtime=False)

        # MOVE to home position
        joint_target = robot_.arm_home_angles_
        self.moveto(robot_, joint_angles=joint_target)
