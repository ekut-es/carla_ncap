# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random
import math


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_RETURN
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import cv2
except ImportError:
    raise RuntimeError('cannot import cv2, make sure cv2 package is installed')

# debugging
import time

# ==============================================================================
# -- constants -----------------------------------------------------------------
# ==============================================================================

BOUNDING_BOX_RENDER_DISTANCE = 100

BB_COLOR = (248, 64, 24)
vehicle_color_dict = {
    "pedestrian" : [60,20,220],
    "rider" : [0,0,255],
    "car" : [142,0,0],
    "truck" : [70,0,0],
    "bus" : [100,60,0],
    "motorcycle" : [230,0,0],
    "bicycle" : [32,11,119]
}


# ==============================================================================
# -- Helpful functions ---------------------------------------------------------
# ==============================================================================

def calculate_border(min, max):
    """
    calculate the maximum amount the bbox borders should be able to shrink
    """

    return int((max - min) / 10)

def find_min_max_points(points):
    """
    gets a non empty list of points and returns the smallest and largest x and y values
    """
    if not points or len(points) == 0:
        raise ValueError("The list of points should not be empty!")

    min_x = max_x = points[0][0]
    min_y = max_y = points[0][1]

    for p in points:
        if p[0] < min_x:
            min_x = p[0]
        if p[0] > max_x:
            max_x = p[0]
        if p[1] < min_y:
            min_y = p[1]
        if p[1] > max_y:
            max_y = p[1]

    return min_x, max_x, min_y, max_y

def find_leftmost_index(array, value, min_x, max_x, min_y, max_y):
    """
    returns the leftmost index where a two dimensional color image array has the value value
    """

    # check the max shrink border for the value or vehicle rider
    for x in range(min_x, max_x - calculate_border(min_x, max_x)):
        for y in range(min_y, max_y - calculate_border(min_y, max_y)):
            if np.all(array[y, x] == value) or np.all(array[y, x] == vehicle_color_dict["rider"]):
                # found leftmost pixel where vehicle begins
                #print("cols: " + str(x) + " | rows: " + str(y))
                #print("array: " + str(array[y, x]) + " | value: " + str(value))
                return x

    return min_x

def find_rightmost_index(array, value, min_x, max_x, min_y, max_y):
    """
    returns the rightmost index where a two dimensional color image array has the value value
    """

    # check the max shrink border for the value or vehicle rider
    for x in range(max_x, min_x + calculate_border(min_x, max_x), -1):
        for y in range(min_y, max_y - calculate_border(min_y, max_y)):
            if np.all(array[y, x] == value) or np.all(array[y, x] == vehicle_color_dict["rider"]):
                # found rightmost pixel where vehicle begins
                return x

    return max_x

def find_topmost_index(array, value, min_x, max_x, min_y, max_y):
    """
    returns the topmost index where a two dimensional color image array has the value value
    """

    # check the max shrink border for the value or vehicle rider
    for y in range(min_y, max_y - calculate_border(min_y, max_y)):
        for x in range(min_x, max_x - calculate_border(min_x, max_x)):
            if np.all(array[y, x] == value) or np.all(array[y, x] == vehicle_color_dict["rider"]):
                # found topmost pixel where vehicle begins
                return y

    return min_y

def find_bottommost_index(array, value, min_x, max_x, min_y, max_y):
    """
    returns the bottommost index where a two dimensional color image array has the value value
    """

    # check the max shrink border for the value or vehicle rider
    for y in range(max_y, min_y + calculate_border(min_y, max_y), -1):
        for x in range(min_x, max_x - calculate_border(min_x, max_x)):
            if np.all(array[y, x] == value) or np.all(array[y, x] == vehicle_color_dict["rider"]):
                # found bottommost pixel where vehicle begins
                return y

    return max_y

def find_extreme_indices(array, value, min_x, max_x, min_y, max_y):
    """
    returns the extreme indices of a tow dimensional color image array where it has the value value
    """

    height, width, _ = array.shape

    min_x_inside = min_x >= 0 and min_x < width
    max_x_inside = max_x >= 0 and max_x < width
    min_y_inside = min_y >= 0 and min_y < height
    max_y_inside = max_y >= 0 and max_y < height

    # case completely outside
    if not (min_x_inside or max_x_inside) or not (min_y_inside or max_y_inside):
        return [min_x, max_x, min_y, max_y]


    if not min_x_inside:
        min_x = 0
    if not max_x_inside:
        max_x = width - 1
    if not min_y_inside:
        min_y = 0
    if not max_y_inside:
        max_y = height - 1

    leftmost = find_leftmost_index(array, value, min_x, max_x, min_y, max_y)
    rightmost = find_rightmost_index(array, value, min_x, max_x, min_y, max_y)
    topmost = find_topmost_index(array, value, min_x, max_x, min_y, max_y)
    bottommost = find_bottommost_index(array, value, min_x, max_x, min_y, max_y)

    return [leftmost, rightmost, topmost, bottommost]

def extreme_indices_are_inside(array_shape, min_x, max_x, min_y, max_y):
    """
    checks if the extreme indices min_x, ..., max_y are inside the shape of an image
    """

    height, width, _ = array_shape

    return min_x >= 0 and min_x < width and max_x >= 0 and max_x < width and min_y >= 0 and min_y < height and max_y >= 0 and max_y < height


# ==============================================================================
# -- ClientSideBoundingBoxes -------------------------------------------------
# ==============================================================================


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    # drawing
    @staticmethod
    def draw_2d_bounding_box_label(world, display, camera_transform, bbox, vehicle):
        """
        Draws a label with information of the car under the bounding box.
        """

        min_x = bbox[0]
        max_x = bbox[1]
        min_y = bbox[2]
        max_y = bbox[3]

        # select closest side of bounding box and calculate center point
        center_x = min_x + 0.5 * (max_x - min_x)
        center = [int(center_x), int(max_y)]

        # calculate the distance from self to camera
        distance_to_camera = vehicle.get_transform().location.distance(camera_transform.location)
        font_size = min(int(500 / distance_to_camera), 27)

        # draw label under bounding box
        if world.get_actor(vehicle.id).type_id.startswith('vehicle.kawasaki.ninja'):
            name = "motorcycle"
        elif world.get_actor(vehicle.id).type_id.startswith('vehicle.'):
            name = world.get_actor(vehicle.id).attributes['base_type']
        else:
            name = "pedestrian"

        # debugging
        name = str(vehicle.id)

        font = pygame.font.Font(None, font_size)

        text_surface = font.render(name, True, (0, 0, 0))
        text_rect = text_surface.get_rect()
        text_rect.midtop = center[:2]

        # Create a surface slightly larger than the text surface
        # and fill it with the background color
        bg_surface = pygame.Surface((text_rect.width, text_rect.height))
        bg_surface.fill(BB_COLOR)

        # Blit the text surface onto the background surface
        bg_surface.blit(text_surface, (0, 0))

        # Blit the background surface onto the screen
        display.blit(bg_surface, text_rect.topleft)

    @staticmethod
    def draw_2d_bounding_boxes(world, display, vehicels_with_bounding_boxes, show_bbox_labels, camera_transform, width, height):
        """
        Draws 2d bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((width, height))
        bb_surface.set_colorkey((0, 0, 0))

        for vehicle, bbox in vehicels_with_bounding_boxes:
            # draw 2d box lines
            min_x = bbox[0]
            max_x = bbox[1]
            min_y = bbox[2]
            max_y = bbox[3]
            #print("vehicle id: " + str(vehicle.id))
            #print("min_x: " + str(min_x) + "\tmax_x: " + str(max_x) + "\tmin_y: " + str(min_y) + "\tmax_y: " + str(max_y))
            # top
            pygame.draw.line(bb_surface, BB_COLOR, (min_x, max_y), (max_x, max_y))
            # bottom
            pygame.draw.line(bb_surface, BB_COLOR, (min_x, min_y), (max_x, min_y))
            # left
            pygame.draw.line(bb_surface, BB_COLOR, (min_x, min_y), (min_x, max_y))
            # right
            pygame.draw.line(bb_surface, BB_COLOR, (max_x, min_y), (max_x, max_y))

            # draw label
            if show_bbox_labels:
                ClientSideBoundingBoxes.draw_2d_bounding_box_label(world, display, camera_transform, bbox, vehicle)

        display.blit(bb_surface, (0, 0))

    # bounding boxes
    @staticmethod
    def convert_to_2d_vehicle_bboxes(vbboxes):
        """
        this function converts a 3d vehicle bbox list to a 2d vehicle bbox list
        """

        converted_vbboxes = []
        for (vehicle, bbox) in vbboxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]

            min_x, max_x, min_y, max_y = find_min_max_points(points)
            converted_vbboxes.append((vehicle, [min_x, max_x, min_y, max_y]))

        return converted_vbboxes

    @staticmethod
    def get_vehicle_segmentation_color(world, vehicle):
        """
        returns the segmentation color of the vehicle
        """

        actor = world.get_actor(vehicle.id)

        if actor.type_id.startswith('walker.pedestrian'):
            return vehicle_color_dict["pedestrian"]

        base_type = actor.attributes['base_type']

        if actor.type_id.startswith('vehicle.kawasaki.ninja') or base_type == "motorcycle":
            return vehicle_color_dict["motorcycle"]
        elif base_type == "car":
            return vehicle_color_dict["car"]
        elif base_type == "truck":
            return vehicle_color_dict["truck"]
        elif base_type == "bus" or base_type == "van" or base_type == "train":
            return vehicle_color_dict["bus"]
        elif base_type == "bicycle":
            return vehicle_color_dict["bicycle"]

        # default to vehicle rider
        return vehicle_color_dict["rider"]

    @staticmethod
    def shrink_2d_bboxes(world, vehicle_boxes, sem_img):
        """
        shrink bounding boxes to fit the visual vehicle
        """

        smaller_boxes = []
        for vehicle, box in vehicle_boxes:
            min_x = box[0]
            max_x = box[1]
            min_y = box[2]
            max_y = box[3]

            # the bbox should be inside the semantic image
            if min_y >= max_y or min_x >= max_x:
                smaller_boxes.append((vehicle, box))       
                continue
            

            # get segmentation color for vehicle
            vehicle_color = ClientSideBoundingBoxes.get_vehicle_segmentation_color(world, vehicle)
            new_box = find_extreme_indices(sem_img, vehicle_color, min_x, max_x - 1, min_y, max_y - 1)

            smaller_boxes.append((vehicle, new_box))

        return np.asarray(smaller_boxes)

    @staticmethod
    def filter_small_bboxes_vehicles_map(world, original_vehicles_bboxes, vehicles_bboxes, sem_img):
        """
        filters the list of vehicles bounding boxes
        the current bounding box should be larger than a percentage of the
        original one and the bounding box should be a minimum width and height
        """

        # min width, height and percentage
        #min_width = 10
        min_height = 15
        percentage = 0.45

        # check that the original and shrunken vehicles_bboxes list are the same size
        if len(original_vehicles_bboxes) != len(vehicles_bboxes):
            print("Fatal shrink filter error!\nOld and new vbbox lists should be same length!")
            return vehicles_bboxes

        # main loop over all vbbox elements
        filtered_vehicle_boxes = []
        for i in range(len(vehicles_bboxes)):
            v_old, b_old = original_vehicles_bboxes[i]
            v_new, b_new = vehicles_bboxes[i]

            height_old = b_old[3] - b_old[2]
            height_new = b_new[3] - b_new[2]

            height_difference = height_new / height_old

            filtered_vehicle_boxes.append((v_new, b_new))



        return filtered_vehicle_boxes

    @staticmethod
    def filter_hidden_bboxes_vehicles_map(world, vehicles_bboxes, sem_img):
        """
        filters the list of vehicles bounding boxes
        the vehicle should be the most visible element in the bbox
        """

        # assistance functions
        def vehicle_is_most_frequent_color(counts, colors, vehicle_color):
            """
            checks that the most frequent and the second most frequent colors
            are either the vehicle color or the color of a pedestrian
            because two wheeled vehicles consist of two colors
            """

            threshold = 1 / 5

            # count the total number of pixels
            total = sum(counts)
            # count the number of vehicle and rider pixels as one
            vehicle_index = np.where((colors == vehicle_color).all(axis=-1))
            rider_index = np.where((colors == vehicle_color_dict["rider"]).all(axis=-1))

            if vehicle_index[0].size == 0:
                if rider_index[0].size == 0:
            
                    return False
                else:
                    # percentage of vehicle rider in image has to be higher than threshold
                    p = counts[rider_index[0]][0] / total

                    return p > threshold
            else:
                if rider_index[0].size == 0:
                    # percentage of vehicle in image has to be higher than threshold
                    p = counts[vehicle_index[0]][0] / total
                    return p > threshold
                
                else:

                    # total percentage of vehicle and vehicle rider in image has to be higher than threshold
                    vehicle_and_rider_count = counts[vehicle_index[0]][0] + counts[rider_index[0]][0]
                    p = vehicle_and_rider_count / total

                    return p > threshold

            return False

        # main loop over all vbbox elements
        filtered_vehicle_boxes = []
        for vehicle, box in vehicles_bboxes:
            # extract bbox extremes
            min_x = box[0]
            max_x = box[1]
            min_y = box[2]
            max_y = box[3]

            # extract only the bbox in the semantic image
            sem_seg_box = sem_img[min_y:max_y, min_x:max_x, ]
            pixels = sem_seg_box.reshape(-1, 3)

            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            # get the color the vehicle should have
            vehicle_color = ClientSideBoundingBoxes.get_vehicle_segmentation_color(world, vehicle)

            filtered_vehicle_boxes.append((vehicle, box))

        return filtered_vehicle_boxes

    @staticmethod
    def get_2d_bounding_boxes_vehicles_map(world, vehicles, camera_transform, camera_calibration, seg_img):
        """
        Creates 2D bounding boxes based on carla vehicle snapshot list and camera.
        """

        vehicles_with_bounding_boxes = [
            (vehicle, ClientSideBoundingBoxes.get_3d_bounding_box(world, vehicle, camera_transform, camera_calibration))
            for vehicle in vehicles
            if vehicle.get_transform().location.distance(camera_transform.location) < BOUNDING_BOX_RENDER_DISTANCE
            ]

        # filter objects behind camera
        vehicles_with_bounding_boxes = [(v, bb) for (v, bb) in vehicles_with_bounding_boxes if all(bb[:, 2] > 0)]

        # convert to 2d bounding boxes
        vehicles_with_bounding_boxes_2d = ClientSideBoundingBoxes.convert_to_2d_vehicle_bboxes(vehicles_with_bounding_boxes)

        # visually shrink bounding boxes
        vehicles_with_bounding_boxes_shrink = ClientSideBoundingBoxes.shrink_2d_bboxes(world, vehicles_with_bounding_boxes_2d, seg_img)

        # filter by size
        vehicles_with_bounding_boxes_filter = ClientSideBoundingBoxes.filter_small_bboxes_vehicles_map(world, vehicles_with_bounding_boxes_2d, vehicles_with_bounding_boxes_shrink, seg_img)

        # filter hidden vehicles
        vehicles_with_bounding_boxes_filter = ClientSideBoundingBoxes.filter_hidden_bboxes_vehicles_map(world, vehicles_with_bounding_boxes_filter, seg_img)

        return vehicles_with_bounding_boxes_filter

    @staticmethod
    def get_3d_bounding_box(world, vehicle, camera_transform, camera_calibration):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords, extent = ClientSideBoundingBoxes._create_bb_points(world, vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(world, bb_cords, extent, vehicle, camera_transform)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera_calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(world, vehicle):
        """
        Returns 3D bounding box for a vehicle.
        takes extnet bug for 2 wheeled vehicles into account
        """

        cords = np.zeros((8, 4))
        extent = world.get_actor(vehicle.id).bounding_box.extent
        type_id = world.get_actor(vehicle.id).type_id
        if type_id == "vehicle.harley-davidson.low_rider":
            extent.x = 2.350175619125366/2.0
            extent.y = 0.7662330269813538/2.0
            extent.z = 0.6534097790718079 + 0.2
        elif type_id == "vehicle.kawasaki.ninja":
            extent.x = 2.043684244155884/2.0
            extent.y = 0.7969123125076294/2.0
            extent.z = 0.5996276140213013 + 0.2
        elif type_id == "vehicle.yamaha.yzf":
            extent.x = 2.1907684803009033/2.0
            extent.y = 0.7662330269813538/2.0
            extent.z = 0.6148329377174377 + 0.1
        elif type_id == "vehicle.diamondback.century":
            extent.x = 1.6562436819076538/2.0
            extent.y = 0.42141881585121155/2.0
            extent.z = 0.7479862570762634 + 0.2
        elif type_id == "vehicle.gazelle.omafiets":
            extent.x = 1.843441367149353/2.0
            extent.y = 0.4674844741821289/2.0
            extent.z = 0.7356970310211182 + 0.2
        elif type_id == "vehicle.bh.crossbike":
            extent.x = 1.5093227624893188/2.0
            extent.y = 0.8659406304359436/2.0
            extent.z = 0.6382263898849487 + 0.1

        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

        return cords, extent

    # misc
    @staticmethod
    def _vehicle_to_sensor(world, cords, extent, vehicle, sensor_transform):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(world, cords, extent, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor_transform)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(world, cords, extent, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        takes bug for 2 wheeled vehicles into account
        """

        vehicle_type_ids = {
            "vehicle.harley-davidson.low_rider",
            "vehicle.kawasaki.ninja",
            "vehicle.yamaha.yzf",
            "vehicle.diamondback.century",
            "vehicle.gazelle.omafiets",
            "vehicle.bh.crossbike"
        }

        bb = world.get_actor(vehicle.id).bounding_box
        transform = vehicle.get_transform()
        type_id = world.get_actor(vehicle.id).type_id

        # bugfix two wheeled vehicles
        if world.get_actor(vehicle.id).type_id in vehicle_type_ids:
            bb.location.x = 0.0
            bb.location.y = 0.0
            if type_id == "vehicle.harley-davidson.low_rider":
                bb.location.z = bb.location.z + 0.2
            elif type_id == "vehicle.kawasaki.ninja":
                bb.location.z = bb.location.z + 0.2
            elif type_id == "vehicle.diamondback.century":
                bb.location.z = bb.location.z + 0.2
            elif type_id == "vehicle.gazelle.omafiets":
                bb.location.z = bb.location.z + 0.2
            else:
                bb.location.z = bb.location.z + 0.1

        realworldPosition = bb.location

        bb_transform = carla.Transform(realworldPosition)

        # transform
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(transform)
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor_transform):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor_transform)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    @staticmethod
    def convert_to_tensor_format(world, vehicles_with_bounding_boxes, camera_transform, width, height):
        """
        formats the data for the rcnn training
        """
        vehicles = []
        bounding_boxes = []
        labels = []
        distances = []

        # separate bounding_boxes and vehicles
        for v, bbox in vehicles_with_bounding_boxes:
            # bboxes
            min_x, max_x, min_y, max_y = bbox
            vehicles.append(v)
            bounding_boxes.append([min_x, min_y, max_x, max_y])

            # labels
            if world.get_actor(v.id).type_id.startswith('vehicle.kawasaki.ninja'):
                # bug: missing base type for this motorcycle
                labels.append("motorcycle")
            if world.get_actor(v.id).type_id.startswith('vehicle.'):
                labels.append(world.get_actor(v.id).attributes['base_type'])
            else:
                labels.append("pedestrian")

            # distances
            distances.append(v.get_transform().location.distance(camera_transform.location))

        d = {}
        d['boxes'] = bounding_boxes
        d['labels'] = labels
        d['distances'] = distances

        return d
