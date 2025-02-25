import glob
import os
import sys


try:
    sys.path.append(glob.glob('~/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import random
import numpy as np
import math
import cv2
from carla import ColorConverter as cc
from PIL import Image
from bounding_boxes import ClientSideBoundingBoxes

BASE_DIR = "./ncap_data/"

car_img = None
car_seg_img = None
imgs = [None] * 12
seg_imgs = [None] * 12
point_clouds = [None] * 12
car_img_running = True
car_seg_img_running = True
car_lidar_running = True
imgs_running = [True] * 12
seg_imgs_running = [True] * 12
point_clouds_running = [True] * 12

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(2000.0)

    # Get the world and blueprint library
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # custom world settings
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.10
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    # set weather and time (noon = 90.0 | afternoon = 15.0 | night = -90.0)
    clear_noon_weather = carla.WeatherParameters.ClearNoon
    clear_noon_weather.sun_altitude_angle = 90.0
    world.set_weather(carla.WeatherParameters.ClearNoon)

    run_scenarios(world, blueprint_library)

def run_scenarios(world, blueprint_library):
    """
    runs all the defined ncap scenarios
    returns a list of collision times for each scenario
    """

    # init --------------------------------------------------------------------
    collisions = []

    is_pedestrian = [
        False,
        False,
        False,
        False,
        False,
        False,

        # pedestrians
        True,
        True,
        True,
        True
    ]

    # Get the blueprints
    bicycle_bp = blueprint_library.find('vehicle.bh.crossbike'),
    pedestrian_bp_1 = blueprint_library.find('walker.pedestrian.0001'),
    pedestrian_bp_2 = blueprint_library.find('walker.pedestrian.0002'),
    pedestrian_bp_3 = blueprint_library.find('walker.pedestrian.0028'),
    pedestrian_bps = [
        pedestrian_bp_1,
        pedestrian_bp_2,
        pedestrian_bp_3,
    ]
    car_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
    collision_sensor_bp = world.get_blueprint_library().find('sensor.other.collision')

    # Configure camera blueprints
    image_size_x = 1920
    image_size_y = 1080
    fov = 90
    # Configure the RGB camera sensor blueprint
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(image_size_x))
    camera_bp.set_attribute('image_size_y', str(image_size_y))
    camera_bp.set_attribute('fov', str(fov))

    # Configure the segmentation camera sensor blueprint
    segmentation_camera_bp = blueprint_library.find('sensor.camera.instance_segmentation')
    segmentation_camera_bp.set_attribute('image_size_x', str(image_size_x))
    segmentation_camera_bp.set_attribute('image_size_y', str(image_size_y))
    segmentation_camera_bp.set_attribute('fov', str(fov))

    # Configure LiDAR Blueprint
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('dropoff_general_rate', '0.25')
    lidar_bp.set_attribute('dropoff_intensity_limit', '0.8')
    lidar_bp.set_attribute('dropoff_zero_intensity', '0.4')
    lidar_bp.set_attribute('points_per_second', str(91000))
    lidar_bp.set_attribute('rotation_frequency', str(10.0))
    lidar_bp.set_attribute('channels', str(52.0))
    lidar_bp.set_attribute('horizontal_fov', str(70.0))
    lidar_bp.set_attribute('lower_fov', str(-12.5))
    lidar_bp.set_attribute('upper_fov', str(12.5))
    lidar_bp.set_attribute('range', str(250.0))

    # infrastructure cameras 1-12
    camera_spawn_points = [
    carla.Transform(carla.Location(x=-113.92990234,y=11.14681396,z=7.46746948), carla.Rotation(pitch=-26.641678,yaw=-138.750793,roll=0.000041)),
    carla.Transform(carla.Location(x=-115.05329102,y=-8.62421326,z=7.26872009), carla.Rotation(pitch=-24.932026,yaw=133.204376,roll=0.000043)),
    carla.Transform(carla.Location(x=-138.35486328,y=-9.36459351,z=8.44570190), carla.Rotation(pitch=-32.190121,yaw=43.696766,roll=0.000032)),
    carla.Transform(carla.Location(x=-138.94128906,y=11.56463989,z=7.96590149), carla.Rotation(pitch=-34.268242,yaw=-44.540871,roll=0.000085)),
    carla.Transform(carla.Location(x=-113.04909180,y=1.05454895,z=6.93961121), carla.Rotation(pitch=-25.697315,yaw=-179.776215,roll=0.001461)),
    carla.Transform(carla.Location(x=-126.75587891,y=-9.79303162,z=7.15608337), carla.Rotation(pitch=-33.467541,yaw=89.523506,roll=0.001062)),
    carla.Transform(carla.Location(x=-139.21208008,y=1.22161018,z=7.18213867), carla.Rotation(pitch=-31.077557,yaw=-2.000458,roll=0.000061)),
    carla.Transform(carla.Location(x=-126.48766602,y=12.58492798,z=7.33829163), carla.Rotation(pitch=-17.146780,yaw=-89.614693,roll=0.001481)),
    carla.Transform(carla.Location(x=-113.04909180,y=1.05454895,z=6.93961121), carla.Rotation(pitch=-21.833300,yaw=-0.733825,roll=0.001470)),
    carla.Transform(carla.Location(x=-126.75587891,y=-9.79303162,z=7.15608337), carla.Rotation(pitch=-34.077076,yaw=-89.601097,roll=0.001091)),
    carla.Transform(carla.Location(x=-139.21208008,y=1.22161018,z=7.18213867), carla.Rotation(pitch=-18.281820,yaw=179.412247,roll=0.000064)),
    carla.Transform(carla.Location(x=-126.48766602,y=12.58492798,z=7.33829163), carla.Rotation(pitch=-30.366514,yaw=88.375465,roll=0.001476)),
    ]

    calibration = np.identity(3)
    calibration[0, 2] = image_size_x / 2.0
    calibration[1, 2] = image_size_y / 2.0
    calibration[0, 0] = calibration[1, 1] = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))

    infrastructure_cameras = []
    infrastructure_segmentation_cameras = []
    infrastructure_lidars = []
    for i in range(len(camera_spawn_points)):
        # spawn cameras
        cam = world.spawn_actor(
            camera_bp,
            camera_spawn_points[i]
        )
        seg_cam = world.spawn_actor(
            segmentation_camera_bp,
            camera_spawn_points[i]
        )
        lidar_sensor = world.spawn_actor(
            lidar_bp,
            camera_spawn_points[i]
        )

        # listen to cameras
        cam.listen(lambda image, index=i: process_img(image, index))
        seg_cam.listen(lambda image, index=i: process_seg_img(image, index))
        lidar_sensor.listen(lambda pcl, index=i: process_pcl(pcl, index))

        # set calibration
        cam.calibration = calibration
        seg_cam.calibration = calibration
        lidar_sensor.calibration = calibration

        # save cams in lists
        infrastructure_cameras.append(cam)
        infrastructure_segmentation_cameras.append(seg_cam)
        infrastructure_lidars.append(lidar_sensor)

    # Define the locations
    car_spawn_point_1 = carla.Transform(
        carla.Location(x=-50,y=-4.16,z=0.1),
        carla.Rotation(pitch=0, yaw=180, roll=0)
    )
    car_spawn_point_2 = carla.Transform(
        carla.Location(x=-85.51011719,y=-4.24960175,z=0.1),
        carla.Rotation(pitch=0, yaw=180, roll=0)
    )
    car_spawn_point_3 = carla.Transform(
        carla.Location(x=-131.84451172,y=-58.17173340,z=0.1),
        carla.Rotation(pitch=0, yaw=90, roll=0)
    )
    car_spawn_point_4 = carla.Transform(
        carla.Location(x=-177.54132812,y=6.48148132,z=0.1),
        carla.Rotation(pitch=0, yaw=0, roll=0)
    )
    car_spawn_point_5 = carla.Transform(
        carla.Location(x=-120.55846680,y=64.54118164,z=0.1),
        carla.Rotation(pitch=0, yaw=270, roll=0)
    )
    collision_locations = [
        carla.Location(x=-117.08, y=-4.16, z=0.1),
        carla.Location(x=-121.51011719, y=-4.24960175, z=0.1),
        carla.Location(x=-131.84451172, y=-5.81532410, z=0.1),
        carla.Location(x=-131.84451172, y=-5.81532410, z=0.1),
        carla.Location(x=-134.63119141, y=6.48148132, z=0.1),
        carla.Location(x=-120.55846680, y=7.64744873, z=0.1),

        # pedestrians
        carla.Location(x=-114.93057617, y=-4.16, z=0.1),
        carla.Location(x=-131.84451172, y=-9.44565674, z=0.1),
        carla.Location(x=-138.33119141, y=6.48148132, z=0.1),
        carla.Location(x=-120.55846680, y=10.64744873, z=0.1),
    ]
    vru_spawn_points = [
        carla.Transform(
            carla.Location(x=-117.08,y=-50,z=0.1),
            carla.Rotation(pitch=0, yaw=90, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-99.51011719,y=-4.24960175,z=0.1),
            carla.Rotation(pitch=0, yaw=180, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-167.99681641,y=-5.81532410,z=0.1),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-131.84451172,y=-43.17173340,z=0.1),
            carla.Rotation(pitch=0, yaw=90, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-134.63119141,y=55.24582031,z=0.1),
            carla.Rotation(pitch=0, yaw=270, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-81.39397461,y=10.64744873,z=0.1),
            carla.Rotation(pitch=0, yaw=180, roll=0)
        ),


        # pedestrians
        carla.Transform(
            carla.Location(x=-114.93057617,y=-15.43779785,z=1.0),
            carla.Rotation(pitch=0, yaw=90, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-144.74497070,y=-9.44565674,z=1.0),
            carla.Rotation(pitch=0, yaw=0, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-138.33119141,y=19.16471558,z=1.0),
            carla.Rotation(pitch=0, yaw=270, roll=0)
        ),
        carla.Transform(
            carla.Location(x=-107.39397461,y=10.64744873,z=1.0),
            carla.Rotation(pitch=0, yaw=180, roll=0)
        ),
    ]
    car_spawn_points = [
        car_spawn_point_1,
        car_spawn_point_2,
        car_spawn_point_3,
        car_spawn_point_3,
        car_spawn_point_4,
        car_spawn_point_5,

        # pedestrians
        car_spawn_point_1,
        car_spawn_point_3,
        car_spawn_point_4,
        car_spawn_point_5,
    ]

    # Set the desired speed in km/h
    bicycle_speed_kmh = 15
    pedestrian_speed_kmh = 5
    pedestrian_speed_ms = pedestrian_speed_kmh / 3.6
    vru_speeds_kmh = [
        bicycle_speed_kmh,
        bicycle_speed_kmh,
        bicycle_speed_kmh,
        bicycle_speed_kmh,
        bicycle_speed_kmh,
        bicycle_speed_kmh,

        # pedestrians
        pedestrian_speed_kmh,
        pedestrian_speed_kmh,
        pedestrian_speed_kmh,
        pedestrian_speed_kmh,
    ]
    vru_speeds_ms = [s / 3.6 for s in vru_speeds_kmh]
    car_speeds_kmh = range(20, 65, 5)
    car_speeds_ms = [speed / 3.6 for speed in car_speeds_kmh]

    # will set car lights to on
    light_state = carla.VehicleLightState(carla.VehicleLightState.Position |
                                          carla.VehicleLightState.LowBeam |
                                          carla.VehicleLightState.HighBeam)

    # run 10 scenarios --------------------------------------------------------
    n = 0 # counter for frames
    for i in range(0, 10):
        print(f"running scenario {i + 1}")
        # select vru (bicycle or semirandom pedestrian)
        if is_pedestrian[i]:
            vru_bp = random.choice(pedestrian_bps)
        else:
            vru_bp = bicycle_bp

        # Spawn the vru and the car at the given locations
        vru = world.spawn_actor(vru_bp[0], vru_spawn_points[i])
        car = world.spawn_actor(car_bp, car_spawn_points[i])
        car_actor_id = car.id
        vru_actor_id = vru.id

        # set lights to on for night scenarios
        if not is_pedestrian[i]:
            vru.set_light_state(light_state)
        car.set_light_state(light_state)

        if not is_pedestrian[i]:
            # only appy handbrake to vru if it is not a pedestrian
            vru.apply_control(carla.VehicleControl(hand_brake=True))
        car.apply_control(carla.VehicleControl(hand_brake=True))

        # move the camera to the correct position on the car
        bbox_extent_car = car.bounding_box.extent
        camera_transform = carla.Transform(carla.Location(x=0.8*bbox_extent_car.x, y=0.0, z=1.3*bbox_extent_car.z))

        # spawn car cameras
        car_camera = world.spawn_actor(camera_bp, camera_transform, attach_to=car, attachment_type=carla.AttachmentType.Rigid)
        car_segmentation_camera = world.spawn_actor(segmentation_camera_bp, camera_transform, attach_to=car, attachment_type=carla.AttachmentType.Rigid)
        car_lidar_sensor = world.spawn_actor(lidar_bp, camera_transform, attach_to=car, attachment_type=carla.AttachmentType.Rigid)
        car_camera.listen(lambda image: process_img(image, -1))
        car_segmentation_camera.listen(lambda image: process_seg_img(image, -1))
        car_lidar_sensor.listen(lambda pcl: process_pcl(pcl, -1))
        car_camera.calibration = calibration
        car_segmentation_camera.calibration = calibration
        car_lidar_sensor.calibration = calibration

        # Attach collision sensors to the car
        collision_car = world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=car)

        # collision callback
        def on_collision(event):
            nonlocal collision_occurred
            collision_occurred = True
        collision_car.listen(on_collision)

        # Distance to collision point
        distance_vru = vru_spawn_points[i].location.distance(collision_locations[i])
        distance_car = car_spawn_points[i].location.distance(collision_locations[i])

        world.tick()
        for car_speed_ms in car_speeds_ms:
            # Calculate travel time
            time_vru = distance_vru / vru_speeds_ms[i]
            time_car = distance_car / car_speed_ms

            # calculate start offset for the faster vehicle
            offset_location, car_drives_first = calculate_offset_location(car_spawn_points[i].location, vru_spawn_points[i].location, collision_locations[i], time_car, time_vru)

            # debugging
            print(f"Running scenario with car speed {car_speed_ms * 3.6:.1f} km/h")

            # set vru velocity
            vru_velocity_vectors = [
                carla.Vector3D(0, vru_speeds_ms[i], 0),
                carla.Vector3D(-vru_speeds_ms[i], 0, 0),
                carla.Vector3D(vru_speeds_ms[i], 0, 0),
                carla.Vector3D(0, vru_speeds_ms[i], 0),
                carla.Vector3D(0, -vru_speeds_ms[i], 0),
                carla.Vector3D(-vru_speeds_ms[i], 0, 0),

                # pedestrians
                carla.Vector3D(0, vru_speeds_ms[i], 0),
                carla.Vector3D(vru_speeds_ms[i], 0, 0),
                carla.Vector3D(0, -vru_speeds_ms[i], 0),
                carla.Vector3D(-vru_speeds_ms[i], 0, 0),
            ]
            vru_direction_vectors = [
                {'x' : 0, 'y': 0, 'z': 0},
                {'x' : 0, 'y': 0, 'z': 0},
                {'x' : 0, 'y': 0, 'z': 0},
                {'x' : 0, 'y': 0, 'z': 0},
                {'x' : 0, 'y': 0, 'z': 0},
                {'x' : 0, 'y': 0, 'z': 0},

                # pedestrians
                {'x' : 0, 'y': 1, 'z': 0},
                {'x' : 1, 'y': 0, 'z': 0},
                {'x' : 0, 'y': -1, 'z': 0},
                {'x' : -1, 'y': 0, 'z': 0},
            ]
            vru_velocity = vru_velocity_vectors[i]
            if not is_pedestrian[i]:
                vru.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                control = carla.WalkerControl()
                control.speed = 0
                control.direction.x = 0
                control.direction.y = 0
                control.direction.z = 0
                vru.apply_control(control)

            # set car velocity
            car_velocity_vectors = [
                carla.Vector3D(-car_speed_ms, 0, 0),
                carla.Vector3D(-car_speed_ms, 0, 0),
                carla.Vector3D(0, car_speed_ms, 0),
                carla.Vector3D(0, car_speed_ms, 0),
                carla.Vector3D(car_speed_ms, 0, 0),
                carla.Vector3D(0, -car_speed_ms, 0),

                # pedestrians
                carla.Vector3D(-car_speed_ms, 0, 0),
                carla.Vector3D(0, car_speed_ms, 0),
                carla.Vector3D(car_speed_ms, 0, 0),
                carla.Vector3D(0, -car_speed_ms, 0),
            ]
            car_velocity = car_velocity_vectors[i]
            car.set_target_velocity(carla.Vector3D(0, 0, 0))

            if car_drives_first:
                car.apply_control(carla.VehicleControl(hand_brake=False))

                while not vehicle_reached_location(car_spawn_points[i].location, offset_location, car.get_transform().location):
                    car.set_target_velocity(car_velocity)
                    n = tick(world, i, n, car_camera, car_lidar_sensor, infrastructure_cameras, infrastructure_lidars, calibration, car_actor_id, vru_actor_id, car_speed_ms, vru_speeds_ms[i])

                if not is_pedestrian[i]:
                    vru.apply_control(carla.VehicleControl(hand_brake=False))
                    vru.set_target_velocity(vru_velocity)
                else:
                    control = carla.WalkerControl()
                    control.speed = pedestrian_speed_ms
                    control.direction.x = vru_direction_vectors[i]['x']
                    control.direction.y = vru_direction_vectors[i]['y']
                    control.direction.z = vru_direction_vectors[i]['z']
                    vru.apply_control(control)
            else:
                if not is_pedestrian[i]:
                    vru.apply_control(carla.VehicleControl(hand_brake=False))

                while not vehicle_reached_location(vru_spawn_points[i].location, offset_location, vru.get_transform().location):
                    if not is_pedestrian[i]:
                        vru.set_target_velocity(vru_velocity)
                    else:
                        control = carla.WalkerControl()
                        control.speed = pedestrian_speed_ms
                        control.direction.x = vru_direction_vectors[i]['x']
                        control.direction.y = vru_direction_vectors[i]['y']
                        control.direction.z = vru_direction_vectors[i]['z']
                        vru.apply_control(control)
                    n = tick(world, i, n, car_camera, car_lidar_sensor, infrastructure_cameras, infrastructure_lidars, calibration, car_actor_id, vru_actor_id, car_speed_ms, vru_speeds_ms[i])

                car.apply_control(carla.VehicleControl(hand_brake=False))
                car.set_target_velocity(car_velocity)

            # run the remaining
            collision_occurred = False

            while not collision_occurred and not (vehicle_reached_location(car_spawn_points[i].location, collision_locations[i], car.get_transform().location) and vehicle_reached_location(vru_spawn_points[i].location, collision_locations[i], vru.get_transform().location)):
                # reapply target velocities to maintain constant speed
                if not is_pedestrian[i]:
                    vru.set_target_velocity(vru_velocity)
                car.set_target_velocity(car_velocity)

                n = tick(world, i, n, car_camera, car_lidar_sensor, infrastructure_cameras, infrastructure_lidars, calibration, car_actor_id, vru_actor_id, car_speed_ms, vru_speeds_ms[i])

            # Check for collisions
            if collision_occurred:
                print(f"Collision detected at car speed {car_speed_ms * 3.6:.1f} km/h")
                collisions.append((i, vru_speeds_ms[i], car_speed_ms, time.time()))
            elif vehicle_reached_location(car_spawn_points[i].location, collision_locations[i], car.get_transform().location) and vehicle_reached_location(vru_spawn_points[i].location, collision_locations[i], vru.get_transform().location):
                print("Passed colission point")
            else:
                print("what")

            reset_vehicle(vru, vru_spawn_points[i])
            reset_vehicle(car, car_spawn_points[i])

            # short delay before another reset (bugfix)
            for short_delay in range(20):
                n = tick(world, i, n, car_camera, car_lidar_sensor, infrastructure_cameras, infrastructure_lidars, calibration, car_actor_id, vru_actor_id, car_speed_ms, vru_speeds_ms[i])

            # Reset positions and velocities for the next scenario
            reset_vehicle(vru, vru_spawn_points[i])
            reset_vehicle(car, car_spawn_points[i])

            # short delay between scenarios (bugfix)
            for short_delay in range(10):
                n = tick(world, i, n, car_camera, car_lidar_sensor, infrastructure_cameras, infrastructure_lidars, calibration, car_actor_id, vru_actor_id, car_speed_ms, vru_speeds_ms[i])

        # destroy actors after final scenario
        collision_car.destroy()
        vru.destroy()
        car.destroy()
        car_camera.stop()
        car_camera.destroy()
        car_segmentation_camera.stop()
        car_segmentation_camera.destroy()
        car_lidar_sensor.stop()
        car_lidar_sensor.destroy()
        print(f"completed scenario {i + 1}")

    # finally destroy infrastructure cameras
    for cam in infrastructure_cameras:
        cam.stop()
        cam.destroy()
    for cam in infrastructure_segmentation_cameras:
        cam.stop()
        cam.destroy()
    for lidar_sens in infrastructure_lidars:
        lidar_sens.stop()
        lidar_sens.destroy()
    print("Scenarios completed")

    return collisions

# helpful functions -----------------------------------------------------------
def tick(world, scenario_index, frame_index, car_camera, car_lidar, infrastructure_cameras, infrastructure_lidars, calibration, car_actor_id, vru_actor_id, car_speed, vru_speed):
    """
    does a world tick and saves all images to disk
    """
    global car_img_running
    global imgs_running
    global car_seg_img_running
    global seg_imgs_running
    global car_point_cloud_running
    global point_clouds_running

    # wait till images finished saving
    timeout = 0
    while car_img_running or any(imgs_running) or car_seg_img_running or any(seg_imgs_running):
        # create a timeout of 10 seconds
        if timeout >= 50:
            print("timeout when waiting for images")
            print(f"car_img_running: {car_img_running}, car_seg_img_running: {car_seg_img_running}, imgs_running: {imgs_running}, seg_imgs_running: {seg_imgs_running}")

            world.tick()
            return frame_index + 1
        time.sleep(0.1)
        timeout = timeout + 1

    save_images(scenario_index, frame_index)
    save_point_clouds(scenario_index, frame_index)

    ground_truth_data = get_ground_truth_data_list(world, car_camera, infrastructure_cameras, calibration)

    save_data_list(ground_truth_data, scenario_index, frame_index)
    save_world_data(world, BASE_DIR + "WorldData/" + str(scenario_index) + "_" + str(frame_index) + ".txt", car_actor_id, vru_actor_id, car_speed, vru_speed)
    del car_camera
    del infrastructure_cameras
    world.tick()

    # increase frame counter
    return frame_index + 1

def reset_vehicle(vehicle, spawn_transform):
    """
    reset car and vru for new scenario
    """
    vehicle.set_transform(spawn_transform)
    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    if not is_pedestrian(vehicle):
        vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        vehicle.apply_control(carla.VehicleControl(hand_brake=True))
    else:
        control = carla.WalkerControl()
        control.speed = 0
        control.direction.y = 0
        control.direction.x = 0
        control.direction.z = 0
        vehicle.apply_control(control)

def is_pedestrian(actor):
    # Check if the actor's blueprint ID starts with "walker.pedestrian."
    return actor.type_id.startswith("walker.pedestrian")

def calculate_offset_location(car_spawn_location, vru_spawn_location, collision_location, time_car, time_vru):
    """
    calculates the offset location. From that position both vehicles shoud
    reach the collision location in the same time.
    """

    # small offset for the vehicle because the location should be in the front not the center
    # calculate whether x or y is further away
    dist_x = collision_location.x - car_spawn_location.x
    dist_y = collision_location.y - car_spawn_location.y

    # move car_spawn_location 2 further away in the direction of the larger distance
    if abs(dist_x) > abs(dist_y):
        new_car_spawn_location = carla.Location(x=car_spawn_location.x - 2.5 * math.copysign(1, (dist_x)), y=car_spawn_location.y, z=car_spawn_location.z)
    else:
        new_car_spawn_location = carla.Location(x=car_spawn_location.x, y=car_spawn_location.y - 2.5 * math.copysign(1, (dist_y)), z=car_spawn_location.z)

    #print(f"time_car: {time_car}, time_vru: {time_vru}")
    #print(f"new_car_spawn_location: {new_car_spawn_location}, vru_spawn_location: {vru_spawn_location}, collision_location: {collision_location}")

    if time_car < time_vru:
        ratio = (time_vru - time_car) / time_vru
        offset_location = vru_spawn_location + (collision_location - vru_spawn_location) * ratio
        car_drives_first = False
    else:
        ratio = (time_car - time_vru) / time_car
        offset_location = new_car_spawn_location + (collision_location - new_car_spawn_location) * ratio
        car_drives_first = True

    #print(f"offset_location: {offset_location}, car_drives_first: {car_drives_first}")
    return offset_location, car_drives_first

def vehicle_reached_location(spawn_location, goal_location, current_location):
    """
    checks if the given location is reached or already passed
    returns the check as Boolean
    """

    # bugfix: before first tick where vehicle is spawned its location is (0, 0, 0)
    if current_location.x == 0 and current_location.y == 0 and current_location.z == 0:
        current_location = spawn_location

    def distance(l1, l2):
        # calculates the distance between two carla locations
        return math.sqrt((l1.x - l2.x) ** 2 + (l1.y - l2.y) ** 2 + (l1.z - l2.z) ** 2)

    dist_spawn_goal = distance(goal_location, spawn_location)
    dist_spawn_current = distance(current_location, spawn_location)

    # debugging
    #print(f"spawn_location: {spawn_location}, goal_location: {goal_location}, current_location: {current_location}")
    #print(f"dist spawn goal: {dist_spawn_goal}")
    #print(f"dist spawn current: {dist_spawn_current}")
    return dist_spawn_current >= dist_spawn_goal

def get_ground_truth_data_list(world, car_camera, infrastructure_cameras, calibration):
    """
    gets the ground truth data for every single camera
    and returns it as a List
    """

    global seg_imgs
    global car_seg_img

    data = []

    data.append(get_ground_truth_data(world, car_camera, car_seg_img, calibration))

    for i in range(len(infrastructure_cameras)):
        data.append(get_ground_truth_data(world, infrastructure_cameras[i], seg_imgs[i], calibration))

    return data

def get_ground_truth_data(world, cam, seg_img, calibration):
    """
    gets the ground truth data for a specific camera and frame
    """

    # get world, vehicle and pedestrian snapshots
    world_snapshot = world.get_snapshot()

    vehicle_snapshots = []

    for snap in world_snapshot:
        type_id = world.get_actor(snap.id).type_id
        if type_id.startswith('vehicle.') or type_id.startswith('walker.pedestrian'):
            vehicle_snapshots.append(snap)

    vehicles_with_bounding_boxes = ClientSideBoundingBoxes.get_2d_bounding_boxes_vehicles_map(world, vehicle_snapshots, cam.get_transform(), calibration, seg_img)

    return ClientSideBoundingBoxes.convert_to_tensor_format(world, vehicles_with_bounding_boxes, cam.get_transform(), 1920, 1080)
def save_point_clouds(scenario_index,frame_index):
    global car_point_cloud
    global point_clouds

    car_point_cloud.tofile(BASE_DIR + "car/" + str(scenario_index) + "_" + str(frame_index) + ".bin")

    for i in range(len(point_clouds)):
        point_clouds[i].tofile(BASE_DIR + str(i) + "/" + str(scenario_index) + "_" + str(frame_index) + ".bin")
        


def save_images(scenario_index, frame_index):
    """
    save all the infracam and the carcam images

    """

    global car_img
    global imgs

    # save images and labels
    save_array_image(car_img, BASE_DIR + "car/" + str(scenario_index) + "_" + str(frame_index) + ".png")
    for i in range(len(imgs)):
        save_array_image(imgs[i], BASE_DIR + str(i) + "/" + str(scenario_index) + "_" + str(frame_index) + ".png")

def save_array_image(image_data, filename):
    """
    Save an image array to disk as a PNG file
    """

    # Convert NumPy array back to an image
    image_data_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_data_rgb.astype('uint8'), 'RGB')
    img.save(filename)

def save_data_list(ground_truth_data, scenario_index, frame_index):
    """
    saves the ground truth data for each camera
    """

    # split car and infrastructure camera data
    ground_truth_car_data, *ground_truth_infra_data = ground_truth_data

    save_cam_data(ground_truth_car_data, BASE_DIR + "car/" + str(scenario_index) + "_" + str(frame_index) + ".txt")
    for i in range(len(ground_truth_infra_data)):
        save_cam_data(ground_truth_infra_data[i], BASE_DIR + str(i) + "/" + str(scenario_index) + "_" + str(frame_index) + ".txt")

def save_cam_data(d, filename):
    """
    saves the ground truth data of a specific camera into a file
    """

    with open(filename, 'a') as file:
        for i in range(min(len(d['labels']), len(d['boxes']), len(d['distances']))):
            file.write(str(d['labels'][i]) + " " + str(d['boxes'][i][0]) + " " + str(d['boxes'][i][1]) + " " + str(d['boxes'][i][2]) + " " + str(d['boxes'][i][3]) + " " + str(d['distances'][i]) + "\n")

def save_world_data(world, filename, car_actor_id, vru_actor_id, car_speed, vru_speed):
    """
    saves data like the distance between the car and the vru
    """

    car_actor = world.get_actor(car_actor_id)
    vru_actor = world.get_actor(vru_actor_id)

    distance = car_actor.get_transform().location.distance(vru_actor.get_transform().location)

    with open(filename, 'a') as file:
        file.write(str(distance) + " " + str(car_speed) + " " + str(vru_speed))

def process_img(image, index):
    """
    gets an image and camera index and saves the image to the appropriate variable
    """

    global imgs
    global imgs_running
    global car_img
    global car_img_running

    conv_img = image.convert(cc.Raw)
    image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_data = image_data.reshape((image.height, image.width, 4))

    # Drop the alpha channel
    image_data = image_data[:, :, :3]

    if index == -1:
        # car camera
        car_img = image_data
        car_img_running = False
    else:
        # infrastructure camera
        imgs[index] = image_data
        imgs_running[index] = False

def process_seg_img(image, index):
    """
    gets a segmentation image and camera index and saves the image to the appropriate variable
    """

    global seg_imgs
    global seg_imgs_running
    global car_seg_img
    global car_seg_img_running

    conv_img = image.convert(cc.CityScapesPalette)
    image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_data = image_data.reshape((image.height, image.width, 4))

    # Drop the alpha channel
    image_data = image_data[:, :, :3]

    if index == -1:
        # car camera
        car_seg_img = image_data
        car_seg_img_running = False
    else:
        # infrastructure camera
        seg_imgs[index] = image_data
        seg_imgs_running[index] = False

def process_pcl(pcl, index):
    """
    gets a segmentation image and camera index and saves the image to the appropriate variable
    """

    global point_clouds
    global point_clouds_running
    global car_point_cloud
    global car_point_cloud_running

   
    points = np.copy(np.frombuffer(pcl.raw_data, dtype=np.dtype('f4')))
    point_cloud = np.reshape(points, (int(points.shape[0] / 4), 4))

    if index == -1:
        # car camera
        car_point_cloud = point_cloud
        car_point_cloud_running = False
    else:
        # infrastructure camera
        point_clouds[index] = point_cloud
        point_clouds_running[index] = False

if __name__ == '__main__':
    main()
