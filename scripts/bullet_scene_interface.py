import time
import numpy as np
import pybullet_data
from screeninfo import get_monitors
import pybullet as p
from pybullet_utils import bullet_client as bc

from util import Util
from objects_handler import *
from bodies.body import Body


class BulletSceneInterface:
    def __init__(
        self,
        time_step=1.0 / HZ,
        gravity=None,
        render=True,
        gpu_rendering=False,
        shadows=False,
        seed=int(time.time()),
        deformable=False,
    ):
        if gravity is None:
            gravity = [0, 0, -9.81]

        self.time_step = time_step
        self.last_sim_time = None
        self.gravity = np.array(gravity)
        self.shadows = shadows
        self.id = None
        self.sim = None
        self.render = render
        self.gpu_rendering = gpu_rendering
        self.view_matrix = None
        self.deformable = deformable
        self.seed(seed)
        self.directory = assets_directory
        self.visual_items = []

        self.body = None
        self.info = {}

        if self.render:
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except:
                self.width = 1920
                self.height = 1080
            self.sim = bc.BulletClient(
                connection_mode=p.GUI,
                options="--background_color_red=0.8 "
                "--background_color_green=0.9 "
                "--background_color_blue=1.0 "
                "--width=%d --height=%d" % (self.width, self.height),
            )
        else:
            self.sim = bc.BulletClient(connection_mode=p.DIRECT)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.id = self.sim._client
        self.util = Util(self.sim)

        self.setup_sim()

    def seed(self, seed=1001):
        np.random.seed(seed)
        self.np_random = None

    def disconnect(self):
        del self.sim

    def setup_sim(self):
        self.sim.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        ground_id = self.sim.loadURDF("plane.urdf")
        self.body = Body(ground_id, self)
        self.info["ground"] = self.body
        self.sim.setTimeStep(self.time_step)
        # self.sim.setRealTimeSimulation(0) # Disable real time simulation so that the
        # simulation only advances when we call stepSimulation
        self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_GUI, False)
        self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_TINY_RENDERER, True)
        self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_RGB_BUFFER_PREVIEW, False)
        self.sim.configureDebugVisualizer(
            self.sim.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False
        )
        self.sim.configureDebugVisualizer(
            self.sim.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False
        )
        self.sim.configureDebugVisualizer(self.sim.COV_ENABLE_SHADOWS, self.shadows)
        self.sim.configureDebugVisualizer(
            self.sim.COV_ENABLE_WIREFRAME, False
        )  # True for wireframe

        if self.render and self.gpu_rendering:
            self.util.enable_gpu()
        self.last_sim_time = time.time()

    def reset(self):
        if self.deformable:
            self.sim.resetSimulation(self.sim.RESET_USE_DEFORMABLE_WORLD)
        else:
            self.sim.resetSimulation()
        if self.gpu_rendering:
            self.util.enable_gpu()
        self.set_gui_camera()
        self.sim.setTimeStep(self.time_step)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        # self.sim.setRealTimeSimulation(0)
        self.sim.setGravity(self.gravity[0], self.gravity[1], self.gravity[2])
        ground_id = self.sim.loadURDF("plane.urdf")
        self.body = Body(ground_id, self)
        self.info["ground"] = self.body
        self.last_sim_time = time.time()

    def set_gui_camera(self, look_at_pos=(0, 0, 0.75), distance=1, yaw=0, pitch=-30):
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=look_at_pos,
            physicsClientId=self.id,
        )

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def stepSimulation(self):
        self.sim.stepSimulation()

    def visualize_coordinate_frame(
        self,
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
        alpha=1.0,
        replace_old_cf=None,
    ):
        transform = lambda pos: self.sim.multiplyTransforms(
            position, get_quaternion(orientation, self), pos, [0, 0, 0, 1]
        )[0]

        x = Line(
            self,
            start=transform([0, 0, 0]),
            end=transform([0.2, 0, 0]),
            rgba=[1, 0, 0, alpha],
            replace_line=None if replace_old_cf is None else replace_old_cf[0],
        )
        y = Line(
            self,
            start=transform([0, 0, 0]),
            end=transform([0, 0.2, 0]),
            rgba=[0, 1, 0, alpha],
            replace_line=None if replace_old_cf is None else replace_old_cf[1],
        )
        z = Line(
            self,
            start=transform([0, 0, 0]),
            end=transform([0, 0, 0.2]),
            rgba=[0, 0, 1, alpha],
            replace_line=None if replace_old_cf is None else replace_old_cf[2],
        )
        return x, y, z

    def clear_visual_item(self, items):
        if items is None:
            return
        if type(items) in (list, tuple):
            for item in items:
                self.sim.removeBody(item.body)
                # p.removeUserDebugItem(item, physicsClientId=env.id)
                for i in range(len(self.visual_items)):
                    if self.visual_items[i] == item:
                        del self.visual_items[i]
                        break
        else:
            self.sim.removeBody(items.body)
            for i in range(len(self.visual_items)):
                if self.visual_items[i] == items:
                    del self.visual_items[i]
                    break

    def clear_all_visual_items(self):
        for item in self.visual_items:
            self.sim.removeBody(item.body)
        self.visual_items = []

    def clear_all_debug_items(self):
        self.sim.removeAllUserDebugItems()

    def step_simulation(self, steps=1, realtime=True):
        for _ in range(steps):
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.id) # Enable rendering
            self.stepSimulation()
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=env.id) # Disable rendering, this allows us to create and delete objects without object flashing
            if realtime and self.render:
                self.slow_time()

    def compute_collision_detection(self):
        self.sim.performCollisionDetection()


def redraw(env):
    env.sim.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)


def get_keys():
    specials = {
        p.B3G_ALT: "alt",
        p.B3G_SHIFT: "shift",
        p.B3G_CONTROL: "control",
        p.B3G_RETURN: "return",
        p.B3G_LEFT_ARROW: "left_arrow",
        p.B3G_RIGHT_ARROW: "right_arrow",
        p.B3G_UP_ARROW: "up_arrow",
        p.B3G_DOWN_ARROW: "down_arrow",
    }
    # return {chr(k) if k not in specials else specials[k] : v for k, v in p.getKeyboardEvents().items()}
    return [
        chr(k) if k not in specials else specials[k]
        for k in p.getKeyboardEvents().keys()
    ]


class Camera:
    def __init__(
        self,
        env,
        camera_pos=(0.5, -0.5, 1.5),
        look_at_pos=(0, 0, 0.75),
        fov=60,
        camera_width=1920 // 4,
        camera_height=1080 // 4,
    ):
        self.env = env
        self.fov = fov
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(
            camera_pos, look_at_pos, [0, 0, 1], physicsClientId=self.env.id
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov,
            self.camera_width / self.camera_height,
            0.01,
            100,
            physicsClientId=self.env.id,
        )

    def set_camera_rpy(self, look_at_pos=(0, 0, 0.75), distance=1.5, rpy=(0, -35, 40)):
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            look_at_pos,
            distance,
            rpy[2],
            rpy[1],
            rpy[0],
            2,
            physicsClientId=self.env.id,
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            self.fov,
            self.camera_width / self.camera_height,
            0.01,
            100,
            physicsClientId=self.env.id,
        )

    def get_rgba_depth(
        self, light_pos=(0, -3, 1), shadow=False, ambient=0.8, diffuse=0.3, specular=0.1
    ):
        w, h, img, depth, segmentation_mask = p.getCameraImage(
            self.camera_width,
            self.camera_height,
            self.view_matrix,
            self.projection_matrix,
            lightDirection=light_pos,
            shadow=shadow,
            lightAmbientCoeff=ambient,
            lightDiffuseCoeff=diffuse,
            lightSpecularCoeff=specular,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.env.id,
        )
        img = np.reshape(img, (h, w, 4))
        depth = np.reshape(depth, (h, w))
        segmentation_mask = np.reshape(segmentation_mask, (h, w))
        return img, depth, segmentation_mask

    def get_point_cloud(self, body=None):
        # get a depth image
        rgba, depth, segmentation_mask = self.get_rgba_depth()
        rgba = rgba.reshape((-1, 4))
        depth = depth.flatten()
        segmentation_mask = segmentation_mask.flatten()

        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))

        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1 : 1 : 2 / self.camera_height, -1 : 1 : 2 / self.camera_width]
        y *= -1.0
        x, y, z = x.reshape(-1), y.reshape(-1), depth
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)

        # Filter point cloud to only include points on the target body
        pixels = pixels[segmentation_mask == body.body]
        z = z[segmentation_mask == body.body]
        rgba = rgba[segmentation_mask == body.body]

        # filter out "infinite" depths
        pixels = pixels[z < 0.99]
        rgba = rgba[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3:4]
        points = points[:, :3]

        return points, rgba / 255
