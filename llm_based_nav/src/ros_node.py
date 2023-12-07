#!/usr/bin/env python
#import setup_path
import cv2
import numpy as np 
import pprint
import time
from src.Navigation_using_foundation_models_and_speech_instruction.planner import lseg_based_nav
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# connect to the AirSim simulator
br = CvBridge()
rawImage = None
depthImage = None
flag_init = 0
def callback_image(data):
    global rawImage
    global flag_init
    rawImage = br.imgmsg_to_cv2(data)
    if(flag_init ==0):
        flag_init = 1


def callback_depth(data):
    global depthImage
    global flag_init
    depthImage = br.imgmsg_to_cv2(data)
    if (flag_init == 1):
        flag_init = 2

    print(depthImage.shape)
if __name__ == '__main__':


    debug_mode = True

    preferred_terrains = ['road','ice']
    preferred_terrains_speeds = [('road',3),('road',3),('ice',1.5)]
    landmarks = ['car', 'animal']

    lseg_planner = lseg_based_nav(planning_horizon=5, max_action=-0.9, given_instruction = (preferred_terrains,
                                                                                            landmarks,
                                                                                          preferred_terrains_speeds))
    rospy.init_node('llm_based_nav', anonymous=True)
    rospy.Subscriber("/oak_front/color/image_raw", Image, callback_image)
    rospy.Subscriber("/oak_front/depth/image_raw", Image, callback_depth)

    rospy.Rate(100)

    throttle = 0.2
    while True:

        if (flag_init>1):
            observations = [0, 0, 0 ,0, 0,
                          0, 0]

            if rawImage  is not None:
                actions, ref_speed, seg_map = lseg_planner.planning(rawImage , depthImage,  'human', measurements=observations,
                                                       unreal_mode=False, threshold=0.8,throttle_command=throttle)
            else:
                input('PNG is none!')

            if debug_mode:
                cv2.imshow("Image_RGB", rawImage )
                cv2.imshow("Image_depth", depthImage)
                cv2.imshow("segmaps", seg_map)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        #np.save('./trajectory_oct_19.npy',np.array(vehicles_poses))
    cv2.destroyAllWindows()
