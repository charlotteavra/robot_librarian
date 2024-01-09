import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet as p
from collections import namedtuple

assets_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../assets"
)

CLEARANCE = 0.0
GUI = True
CLIENT = 0
TABLE_COLOUR = [0.5, 0.5, 0.5]
STATIC_COLOUR = [0.698, 0.133, 0.133]
MOVEABLE_COLOUR = [0.0, 0.0, 1.0]
OOI_COLOUR = [1.0, 0.843, 0.0]
HZ = 140.0

FALL_POS_THRESH = 0.35  # float('inf')
FALL_VEL_THRESH = 1.0  # float('inf')
CONTACT_THRESH = 1e-9
PUSH_END_THRESH = 0.02

YCB_OBJECTS = {
    2: "002_master_chef_can",
    3: "003_cracker_box",
    4: "004_sugar_box",
    5: "005_tomato_soup_can",
    6: "006_mustard_bottle",
    7: "007_tuna_fish_can",
    8: "008_pudding_box",
    9: "009_gelatin_box",
    10: "010_potted_meat_can",
    11: "011_banana",
    19: "019_pitcher_base",
    21: "021_bleach_cleanser",
    24: "024_bowl",
    25: "025_mug",
    35: "035_power_drill",
    36: "036_wood_block",
}


#########
# Panda #
#########

PANDA_GROUPS = {
    "panda": [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
    "hand": ["panda_finger_joint1", "panda_finger_joint2"],
}


##########
# Joints #
##########


def get_num_joints(body, sim):
    if sim is None:
        return p.getNumJoints(body)
    else:
        return sim.sim.getNumJoints(body)


def get_joints(body, sim=None):
    return list(range(get_num_joints(body, sim)))


JointInfo = namedtuple(
    "JointInfo",
    [
        "jointIndex",
        "jointName",
        "jointType",
        "qIndex",
        "uIndex",
        "flags",
        "jointDamping",
        "jointFriction",
        "jointLowerLimit",
        "jointUpperLimit",
        "jointMaxForce",
        "jointMaxVelocity",
        "linkName",
        "jointAxis",
        "parentFramePos",
        "parentFrameOrn",
        "parentIndex",
    ],
)


def get_joint_info(body, joint, sim):
    if sim is None:
        return JointInfo(*p.getJointInfo(body, joint))
    else:
        return JointInfo(*sim.getJointInfo(body, joint))


def get_joint_name(body, joint, sim):
    return get_joint_info(body, joint, sim).jointName.decode("UTF-8")


def joint_from_name(body, name, sim=None):
    for joint in get_joints(body, sim=sim):
        if get_joint_name(body, joint, sim) == name:
            return joint
    raise ValueError(body, name)


def joints_from_names(body, names, sim=None):
    return tuple(joint_from_name(body, name, sim) for name in names)


JointState = namedtuple(
    "JointState",
    [
        "jointPosition",
        "jointVelocity",
        "jointReactionForces",
        "appliedJointMotorTorque",
    ],
)


def get_joint_state(body, joint, sim):
    if sim is None:
        return JointState(*p.getJointState(body, joint))
    else:
        return JointState(*sim.getJointState(body, joint))


def get_joint_position(body, joint, sim=None):
    return get_joint_state(body, joint, sim).jointPosition


def get_joint_velocity(body, joint, sim=None):
    return get_joint_state(body, joint, sim).jointVelocity


def get_joint_reaction_force(body, joint, sim=None):
    return get_joint_state(body, joint, sim).jointReactionForces


def get_joint_torque(body, joint, sim=None):
    return get_joint_state(body, joint, sim).appliedJointMotorTorque


def get_joint_positions(body, joints, sim=None):  # joints=None):
    return tuple(get_joint_position(body, joint, sim) for joint in joints)


def get_joint_velocities(body, joints, sim=None):
    return tuple(get_joint_velocity(body, joint, sim) for joint in joints)


#########
# Links #
#########

get_links = get_joints  # Does not include BASE_LINK


def get_all_links(body, sim=None):
    return [BASE_LINK] + list(get_links(body, sim=sim))


def get_link_parent(body, link, sim):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link, sim).parentIndex


def get_all_link_parents(body, sim):
    return {link: get_link_parent(body, link, sim) for link in get_links(body, sim=sim)}


def get_all_link_children(body, sim):
    children = {}
    for child, parent in get_all_link_parents(body, sim).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link, sim):
    children = get_all_link_children(body, sim)
    return children.get(link, [])


def get_link_descendants(body, link, sim, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link, sim):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, sim, test=test))
    return descendants


def get_link_subtree(body, link, sim=None, **kwargs):
    return [link] + get_link_descendants(body, link, sim, **kwargs)


LinkState = namedtuple(
    "LinkState",
    [
        "linkWorldPosition",
        "linkWorldOrientation",
        "localInertialFramePosition",
        "localInertialFrameOrientation",
        "worldLinkFramePosition",
        "worldLinkFrameOrientation",
    ],
)


def get_link_state(body, link, sim=None):
    if sim is None:
        return LinkState(*p.getLinkState(body, link))
    else:
        return LinkState(*sim.getLinkState(body, link))


BodyInfo = namedtuple("BodyInfo", ["base_name", "body_name"])


def get_body_info(body, sim):
    if sim is None:
        return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))
    else:
        return BodyInfo(*sim.getBodyInfo(body, physicsClientId=CLIENT))


def get_base_name(body, sim=None):
    return get_body_info(body, sim).base_name.decode(encoding="UTF-8")


def get_link_name(body, link, sim=None):
    if link == BASE_LINK:
        return get_base_name(body, sim)
    return get_joint_info(body, link, sim).linkName.decode("UTF-8")


def link_from_name(body, name, sim=None):
    if name == get_base_name(body, sim):
        return BASE_LINK
    for link in get_joints(body, sim):
        if get_link_name(body, link, sim) == name:
            return link
    raise ValueError(body, name)


##########
# Angles #
##########

FLOATS = [float, np.float64, np.float32]


def normalize_angle(a):
    if np.any(np.fabs(a) > 2 * np.pi):
        if type(a) in FLOATS:
            a = np.fmod(a, 2 * np.pi)
        else:
            r = np.where(np.fabs(a) > 2 * np.pi)
            a[r[0]] = np.fmod(a[r[0]], 2 * np.pi)
    while np.any(a < -np.pi):
        if type(a) in FLOATS:
            a += 2 * np.pi
        else:
            r = np.where(a < -np.pi)
            a[r[0]] += 2 * np.pi
    while np.any(a > np.pi):
        if type(a) in FLOATS:
            a -= 2 * np.pi
        else:
            r = np.where(a > np.pi)
            a[r[0]] -= 2 * np.pi
    return a


def shortest_angle_diff(af, ai):
    return normalize_angle(af - ai)


def shortest_angle_dist(af, ai):
    return np.fabs(shortest_angle_diff(af, ai))


def get_euler(quaternion, env):
    return (
        np.array(quaternion)
        if len(quaternion) == 3
        else np.array(env.sim.getEulerFromQuaternion(np.array(quaternion)))
    )


def get_quaternion(euler, env):
    return (
        R.from_matrix(euler).as_quat()
        if np.array(euler).ndim > 1
        else (
            np.array(euler)
            if len(euler) == 4
            else np.array(env.sim.getQuaternionFromEuler(np.array(euler)))
        )
    )


def get_rotation_matrix(quaternion, env):
    return np.array(
        env.sim.getMatrixFromQuaternion(get_quaternion(quaternion, env))
    ).reshape((3, 3))


def get_axis_angle(quaternion, env):
    q = get_quaternion(quaternion, env)
    sqrt = np.sqrt(1 - q[-1] ** 2)
    return np.array([q[0] / sqrt, q[1] / sqrt, q[2] / sqrt]), 2 * np.arccos(q[-1])


def get_difference_quaternion(q1, q2, env):
    return env.sim.getDifferenceQuaternion(
        get_quaternion(q1, env), get_quaternion(q2, env)
    )


def quaternion_product(q1, q2, env):
    # Return Hamilton product of 2 quaternions
    return env.sim.multiplyTransforms(
        [0, 0, 0], get_quaternion(q1, env), [0, 0, 0], q2
    )[1]


def multiply_transforms(p1, q1, p2, q2, env):
    return env.sim.multiplyTransforms(
        p1, get_quaternion(q1, env), p2, get_quaternion(q2, env)
    )


def rotate_point(point, quaternion, env):
    return env.sim.multiplyTransforms(
        [0, 0, 0], get_quaternion(quaternion, env), point, [0, 0, 0, 1]
    )[0]


########
# Misc #
########


class bcolors:
    PINK = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    HIGHLIGHT = "\033[7m"
    STRIKE = "\033[9m"
