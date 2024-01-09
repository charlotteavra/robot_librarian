import numpy as np
from heapq import heappop, heappush
import pybullet as p
from bookshelf import Bookshelf
import objects_handler as oh


class Node(object):
    def __init__(self, pose, idx):
        self.pose = np.array(pose)
        self.joint_angles = None
        self.idx = idx
        self.x = pose[0]
        self.y = pose[1]
        self.z = pose[2]
        self.g_value = 0
        self.h_value = 0
        self.f_value = 0
        self.parent = None

    def __lt__(self, other):
        return self.f_value < other.f_value

    def __eq__(self, other):
        return (self.pose == other.pose).all()


class AStar(object):
    def __init__(self, robot_, map, normal, bookshelf):
        self.robot = robot_
        self.map = map
        self.normal = normal
        self.bookshelf = bookshelf
        self.goal_h_value = 0.1

    def get_successor(self, node):
        """
        Returns 8 neighbors of current node (or maximum possible if edge/corner node)
        """
        successor_list = []
        i, j = node.idx

        idx_list = [
            [i + 1, j + 1],
            [i, j + 1],
            [i - 1, j + 1],
            [i - 1, j],
            [i - 1, j - 1],
            [i, j - 1],
            [i + 1, j - 1],
            [i + 1, j],
        ]

        for idx_ in idx_list:
            if idx_[0] > 0 and idx_[1] > 0:
                try:
                    if self.map[idx_[0], idx_[1], 3] == 0:
                        try:
                            pose_ = self.map[idx_[0], idx_[1], :][:3]
                            self.map[idx_[0], idx_[1], 3] = -1
                            successor_list.append(Node(pose_, idx_))
                        except:
                            continue
                except:
                    continue

        return successor_list

    def heuristic(self, current):
        """
        Computes cost-to-go
        """
        robot_ = self.robot

        # Get current joint angles and box configs to reset to
        robot_ja_init = robot_.get_joint_angles(robot_.controllable_joints)
        box0_pose_init = p.getBasePositionAndOrientation(self.bookshelf.boxes[0].body)
        box1_pose_init = p.getBasePositionAndOrientation(self.bookshelf.boxes[1].body)

        offset_point = current.pose + 0.015 * self.normal
        orn = [-np.pi / 2, p.getEulerFromQuaternion(box1_pose_init[1])[1], 0]
        cur_ee_pose = (offset_point, orn)

        # Visualize
        # oh.Shape(
        #     robot_.env,
        #     oh.Sphere(radius=0.005),
        #     static=False,
        #     mass=1,
        #     position=cur_ee_pose[0],
        #     rgba=[np.random.uniform(0, 1) for _ in range(3)] + [1],
        # )

        # Define goal condition
        ideal_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])

        # Check for IK solution of None or in collision
        robot_joint_angles_ = robot_.ik(
            robot_.end_effector,
            target_pos=cur_ee_pose[0],
            target_orient=cur_ee_pose[1],
            use_current_joint_angles=True,
        )
        current.joint_angles = robot_joint_angles_
        if robot_joint_angles_ is None or self.bookshelf._robot_in_collision(
            robot_joint_angles_, robot_, self.bookshelf.obstacles
        ):
            score = (
                np.arccos((2 * np.dot(ideal_quat, box1_pose_init[1]) ** 2) - 1)
            ) / np.pi  # box is not moved so score is based on current box config
            return score

        robot_.control(robot_joint_angles_, set_instantly=True)
        self.bookshelf._execute_push(robot_, cur_ee_pose, self.normal, realTime=False)

        box1_pose_fin = p.getBasePositionAndOrientation(self.bookshelf.boxes[1].body)
        score = (np.arccos((2 * np.dot(ideal_quat, box1_pose_fin[1]) ** 2) - 1)) / np.pi

        # return boxes and robot to original position
        p.resetBasePositionAndOrientation(
            self.bookshelf.boxes[0].body,
            box0_pose_init[0],
            box0_pose_init[1],
        )
        p.resetBasePositionAndOrientation(
            self.bookshelf.boxes[1].body,
            box1_pose_init[0],
            box1_pose_init[1],
        )
        robot_.control(robot_ja_init, set_instantly=True)

        print(score)
        return score

    def plan(self, start_idx):
        # initialize start node and goal node class
        start_node = Node(self.map[start_idx[0], start_idx[1], :][:3], start_idx)

        # calculate h and f value of start_node
        start_node.h_value = self.heuristic(start_node)
        start_node.f_value = start_node.g_value + start_node.h_value

        # Initially, only the start node is known.
        open_list = []
        closed_list = np.array([])
        heappush(open_list, start_node)

        # while open_list is not empty
        while len(open_list):
            # Current is the node in open_list that has the lowest f value
            # This operation can occur in O(1) time if open_list is a min-heap or a priority queue
            current = heappop(open_list)
            closed_list = np.append(closed_list, current)

            self.map[current.idx[0], current.idx[1], 3] = -1

            if current.h_value < self.goal_h_value:
                box1_pose_init = p.getBasePositionAndOrientation(
                    self.bookshelf.boxes[1].body
                )
                cur_orn = [
                    -np.pi / 2,
                    p.getEulerFromQuaternion(box1_pose_init[1])[1],
                    0,
                ]
                print("============ GOAL NODE FOUND ============")
                return current.pose, cur_orn, current.joint_angles

            for successor in self.get_successor(current):
                """
                1. pass current node as parent of successor node
                2. calculate g, h, and f value of successor node
                    (1) d(current, successor) is the weight of the edge from current to successor
                    (2) g(successor) = g(current) + d(current, successor)
                    (3) h(successor) can be computed by calling the heuristic method
                    (4) f(successor) = g(successor) + h(successor)
                """
                successor.parent = current
                successor.g_value = current.g_value + self.heuristic(successor)
                successor.h_value = self.heuristic(successor)
                successor.f_value = successor.g_value + successor.h_value

                heappush(open_list, successor)

        # If the loop is exited without return, goal node not found
        print("============ GOAL NODE NOT FOUND ============")
        return None
