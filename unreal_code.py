#import setup_path
import airsim
import cv2
import numpy as np 
import pprint
import time
from planner import lseg_based_nav
# connect to the AirSim simulator


if __name__ == '__main__':
    #client = airsim.VehicleClient()
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()
    car_controls.steering = 0
    car_controls.throttle = 0
    car_controls.brake = 0
    client.setCarControls(car_controls)

    debug_mode = False

    lseg_planner = lseg_based_nav(planning_horizon=5, max_action=-0.9)
    previous_action = np.zeros(2)
    throttle = 0.2
    ref_speed = 2
    vehicles_poses = []
    while True:
        begin_time = time.time()
        car_state = client.getCarState()

        pose = car_state.kinematics_estimated.position
        pose = [pose.x_val, pose.y_val, pose.z_val]
        vehicles_poses.append(pose)
        angles = car_state.kinematics_estimated.orientation
        angles = [angles.x_val, angles.y_val, angles.z_val]
        print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]+np.math.pi/2))
        #airsim.ImageRequest(1, airsim.ImageType.Segmentation, True, True)
        responses = client.simGetImages([
            # png format
            # uncompressed RGB array bytes
            airsim.ImageRequest(1, airsim.ImageType.Scene, False, False),
            # floating point uncompressed image
            airsim.ImageRequest(1, airsim.ImageType.DepthPlanar, True)])

        #segImage = responses[0]
        rawImage = responses[0]
        depthImage = responses[1]
        if not rawImage:
            continue
        png = np.fromstring(rawImage.image_data_uint8, dtype=np.uint8)
        png = png.reshape(responses[0].height, responses[0].width, 3)
        throttle = 0.4 * (ref_speed - car_state.speed)

        depthImage = airsim.list_to_2d_float_array(depthImage.image_data_float, depthImage.width, depthImage.height)
        ##print(depthImage.shape) -> 144, 256
        # depth_instensity = np.array(256 * depthImage / 0x0fff,
        #                             dtype=np.uint8)
        ##segmap = airsim.list_to_2d_float_array(segImage.image_data_float, segImage.width, segImage.height)

          # dx, dy, dbearing, vel, pitch, steering_angle, throttle = observations
        observations = [pose[0], pose[1], angles[2] ,car_state.speed, angles[0],
                        previous_action[0], previous_action[1]]
        actions = lseg_planner.planning(png, depthImage, 'road', measurements=observations, unreal_mode=True, threshold=0.1,throttle_command=throttle)
        car_controls.steering = float(actions[0].detach().cpu().numpy())

        car_controls.brake = np.maximum(0.5*(car_state.speed - ref_speed),0)
        car_controls.steering
        car_controls.throttle = throttle
        client.setCarControls(car_controls)
        previous_action[0] = actions[0]
        previous_action[1] = car_controls.throttle
        print('steering: ',actions[0])
        if debug_mode:
            cv2.imshow("AirSim_RGB", png)
            cv2.imshow("AirSim_depth", depthImage)
            ##cv2.imshow("AirSim_segmap", segmap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # print("x={}, y={}, z={}".format(pose[0], pose[1], pose[2]))
            # print("pitch={}, roll={}, yaw={}".format(angles[0], angles[1], angles[2]))
            print('Observation: ',observations)
            print('actions: ',actions)
        time.sleep(0.05)

        #print('actions: ', actions)
        # elif cv2.waitKey(1) & 0xFF == ord('c'):
        #     client.simClearDetectionMeshNames(camera_name, image_type)
        # elif cv2.waitKey(1) & 0xFF == ord('a'):
        #     client.simAddDetectionFilterMeshName(camera_name, image_type, "Cylinder*")

        np.save('./trajectory_sep_28.npy',np.array(vehicles_poses))
    cv2.destroyAllWindows()
