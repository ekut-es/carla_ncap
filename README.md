# EuroNCAP Dataset

## Sensor Data and 2D Labels
Directories 0-11 are the infrastructure sensor units and car is the vehicle under Test.
Each directory contains camera images, LiDAR point clouds and information for each frame.
The information is saved in the following format.
```
actor_type min_x min_y max_x max_y dist
```
Where `actor_type` is one of

```
[car, motorcycle, bycicle, pedestrian]
```
`min_x`, `min_y`, `max_x`, `max_y` is the bounding Box of the vehicle
and `dist` is the distance of the camera to the actor.

## World Data
The directory WorldData contains information about the world for each frame.
```
dist car_speed vru_speed
```
Where `dist` is the distance between the VRU and car in meters and `car_speed` and `vru_speed` are the speeds of the actors in m/s.
## Eval
Eval is not part of the simulation. It contains the predicted bounding boxes for each frame.
```
cam_index actor_type min_x min_y max_x max_y confidence
```
Where `cam_index` is the camera in the scene for which the prediction was made. It is one of
```
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, car]
```
`actor_type`, `min_x`, `min_y`, `max_x`, `max_y`  are the same as in **Cam Data** and `confidence` is the models' confidence score for that prediction.