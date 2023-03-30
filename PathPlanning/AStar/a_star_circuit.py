"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math

import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation

from PIL import Image, ImageOps


show_animation = True


class AStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return (
                str(self.x)
                + ","
                + str(self.y)
                + ","
                + str(self.cost)
                + ","
                + str(self.parent_index)
            )

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1,
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1,
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost
                + self.calc_heuristic(goal_node, open_set[o]),
            )
            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(
                    self.calc_grid_position(current.x, self.min_x),
                    self.calc_grid_position(current.y, self.min_y),
                    "xc",
                )
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    "key_release_event",
                    lambda event: [exit(0) if event.key == "escape" else None],
                )
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id,
                )
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)
        ]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # obstacle map generation
        self.obstacle_map = [
            [False for _ in range(self.y_width)] for _ in range(self.x_width)
        ]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        return motion


def convert_im_to_grid(filepath):
    # Open, grayscale, resize image
    im = Image.open(filepath)
    im = ImageOps.grayscale(im)
    im = im.resize((1024, 1024))

    # Convert to numpy array
    grid = np.asarray(im)

    # Convert all in-between gray to a single value
    grid = np.where(np.logical_or(grid == 0, grid == 255), grid, 100)

    # Map from colors to occupancy grid
    #   Black (0) -> free space (0)
    #   Gray (other) -> obstacle (100)
    #   While (255) -> unknown (-1)
    mapping = {0: 0, 100: 100, 255: -1}
    print(set(grid.flatten()))
    grid = np.vectorize(mapping.get)(grid)

    return grid


def main():
    print(__file__ + " start!!")

    with open("./map_and_pose/occupancy_grid2.npy", "rb") as f:
        grid = np.load(f)
    with open("./map_and_pose/map_metadata.npy", "rb") as f:
        # resolution, width, height
        map_metadata = np.load(f)
    with open("./map_and_pose/map_origin.npy", "rb") as f:
        # position.x, position.y, position.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w
        map_origin = np.load(f)
    with open("./map_and_pose/pose.npy", "rb") as f:
        # position.x, position.y, position.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w
        pose = np.load(f)

    # Unpack
    map_origin_x, map_origin_y, *_ = map_origin
    map_resolution, map_width, map_height = map_metadata

    # start position
    sx = pose[0]
    sy = pose[1]
    rotation = Rotation.from_quat(pose[3:])
    _, _, yaw = rotation.as_euler("xyz", degrees=False)

    # manually set start pose for testing
    sx = -2.7
    sy = -8.7
    yaw = 0

    # set goal position
    grid_size = 0.25
    robot_radius = 0.5
    scaling_factor = robot_radius * 2.25
    gx = sx - (math.cos(yaw)) * scaling_factor
    gy = sy - (math.sin(yaw)) * scaling_factor

    midpoint_x = (sx + gx) / 2
    midpoint_y = (sy + gy) / 2

    run = sx - gx  # horizontal distance from start to goal
    rise = sy - gy  # vertical distance from start to goal

    def point_to_index_x(point):
        return int((point - map_origin_x) / map_resolution)

    def point_to_index_y(point):
        return int((point - map_origin_y) / map_resolution)

    midpoint_x_idx = point_to_index_x(midpoint_x)
    midpoint_y_idx = point_to_index_y(midpoint_y)

    def draw_vertical():
        i = 1  # start at 1 so that second part of line can start at 0
        while True:
            if (
                midpoint_y_idx + i >= len(grid[0])
                or grid[midpoint_x_idx][midpoint_y_idx + i] == 100
            ):
                break
            grid[midpoint_x_idx][midpoint_y_idx + i] = 100
            i += 1
        i = 0
        while True:
            if (
                midpoint_y_idx - i == 0
                or grid[midpoint_x_idx][midpoint_y_idx - i] == 100
            ):
                break
            grid[midpoint_x_idx][midpoint_y_idx - i] = 100
            i += 1

    def draw_horizontal():
        i = 1  # start at 1 so that second part of line can start at 0
        while True:
            if (
                midpoint_x_idx + i >= len(grid)
                or grid[midpoint_x_idx + i][midpoint_y_idx] == 100
            ):
                break
            grid[midpoint_x_idx + i][midpoint_y_idx] = 100
            i += 1
        i = 0
        while True:
            if (
                midpoint_x_idx - i == 0
                or grid[midpoint_x_idx - i][midpoint_y_idx] == 100
            ):
                break
            grid[midpoint_x_idx - i][midpoint_y_idx] = 100
            i += 1

    if run > rise:
        draw_vertical()
    else:
        draw_horizontal()

    # Construct obstacles from grid
    ob = np.argwhere(grid == 100)
    downsample_factor = 20
    ox, oy = (
        list(ob[::downsample_factor, 0] * map_resolution + map_origin_x),
        list(ob[::downsample_factor, 1] * map_resolution + map_origin_y),
    )

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.plot(midpoint_x, midpoint_y, "xr")
        plt.grid(True)
        plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    # save waypoints to a file
    np.savetxt("../rx.numpy", np.array(rx))
    np.savetxt("../ry.numpy", np.array(ry))

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == "__main__":
    main()
