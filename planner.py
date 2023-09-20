# we don't have the image-based planner in this code! and the MPC takes a set of desired waypoints as input for planning
from evotorch import Problem
from evotorch.algorithms import SNES, CEM, CMAES
from evotorch.logging import StdOutLogger, PandasLogger
import torch
import numpy as np
from mpc_planner.system_identification import MHE_MPC
from mpc_planner.mpc_controller import mpc_planner
from lseg import LSegNet
import torch.nn.functional as F

# for lseg
import clip
import cv2

class lseg_based_nav:
    def __init__(self):
        planning_horizon = 10
        num_of_actions = planning_horizon  # one for the velocity
        self.planner = mpc_planner(planning_horizon, num_of_actions)
        # 5.8262448167737955e+02
        self.cam_parameters = {
            'FX_DEPTH' : 500,
            'FY_DEPTH' : 80,
            'CX_DEPTH' : 3.1304475870804731e+02,
            'CY_DEPTH' : 2.3844389626620386e+02}

        self.net = LSegNet(
            backbone="clip_vitl16_384",
            features=256,
            crop_size=480,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )

        self.lseg_initialization()
    def planning(self, current_image, current_depth_map, preferred_terrain):
        imagePlane_point = self.image_processing(current_image, preferred_terrain)
        realWorld_point = self.point_to_3d_world(current_depth_map, imagePlane_point)
        planned_actions = self.planner.plan(realWorld_point) #TODO check this
        return planned_actions
    def point_to_3d_world(self, depth_image, point):
        w = int(point[1])
        h = int(point[0])
        z = depth_image[h][w]/1000 # the raw data is in mm
        x = (w - self.cam_parameters['CX_DEPTH']) * z / self.cam_parameters['FX_DEPTH']
        y = (h - self.cam_parameters['CY_DEPTH']) * z / self.cam_parameters['FY_DEPTH']
        return [y,x]
    def lseg_initialization(self):
        # Initialize the model

        # Load pre-trained weights
        self.net.load_state_dict(torch.load('/home/barbados/lseg-minimal/examples/checkpoints/lseg_minimal_e200.ckpt'))
        self.net.eval()
        self.net.cuda()

        # Preprocess the text prompt
        clip_text_encoder = self.net.clip_pretrained.encode_text
        self.label_classes = ['grass', 'stairs', 'other']
        # Cosine similarity module
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            # Extract and normalize text features
            prompt = [clip.tokenize(lc).cuda() for lc in self.label_classes]
            text_feat_list = [clip_text_encoder(p) for p in prompt]
            self.text_feat_norm_list = [
                torch.nn.functional.normalize(tf) for tf in text_feat_list
            ]
    def image_processing(self, img, preferred_class='grass'):
            image_size = (640,480)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            img = torch.from_numpy(img).float() / 255.0
            img = img[..., :3]  # drop alpha channel, if present
            img = img.cuda()
            img = img.permute(2, 0, 1)  # C, H, W
            img = img.unsqueeze(0)  # 1, C, H, W
            # Extract per-pixel CLIP features (1, 512, H // 2, W // 2)
            img_feat = self.net.forward(img)
            # Normalize features (per-pixel unit vectors)
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
            # Compute cosine similarity across image and prompt features
            similarities = []
            for _i in range(len(self.label_classes)):
                similarity = self.cosine_similarity(
                    img_feat_norm, self.text_feat_norm_list[_i].unsqueeze(-1).unsqueeze(-1)
                )
                similarities.append(similarity)

            similarities = torch.stack(
                similarities, dim=0
            )  # num_classes, 1, H // 2, W // 2
            similarities = similarities.squeeze(1)  # num_classes, H // 2, W // 2
            similarities = similarities.unsqueeze(0)  # 1, num_classes, H // 2, W // 2

            desired_class = self.label_classes.index(preferred_class)
            class_scores = torch.max(similarities, 1)[1].type('torch.FloatTensor')  # 1, H // 2, W // 2
            uncertainties = torch.var(similarities, 1)
            kernels_list = [100,50,20,10,5]

            for item in kernels_list:
                # this clustered thing has 1,...,... shape!
                clustered_data = F.avg_pool2d(class_scores, kernel_size=item)
                clustered_uncertainties = torch.max_pool2d(uncertainties, kernel_size=item)
                lower_bound = int(0.6 * clustered_data.shape[1])  # to make sure that it's on the ground not air
                higher_bound = int(0.9 * clustered_data.shape[1])
                if torch.sum(clustered_data[0][lower_bound:higher_bound,:] == desired_class)>0:
                    break
            clustered_data = clustered_data[0].detach().cpu()
            clustered_uncertainties = clustered_uncertainties[0].detach().cpu()
            selected_point = [] # the third element is the uncertainty

            row_scale, col_scale = image_size[1]/clustered_data.shape[0],image_size[0]/clustered_data.shape[1]
            for row in range(lower_bound,higher_bound): # don't come to the bottom too much to be able to reach that point!
                for col in range(clustered_data.shape[1]):
                    if clustered_data[row,col] == desired_class:
                        selected_point.append([row*row_scale,col*col_scale, clustered_uncertainties[row,col]])

            selected_point = np.array(selected_point)
            candidate_point = np.argmin(abs(selected_point[:,:2]-selected_point[:,:2].mean(0)).sum(1))
            return selected_point[candidate_point,:2]

if __name__ == '__main__':
    image_path = '/home/barbados/rc_car_new_data_mcgill/image_data/image_2023-06-13-17-23-24_0001.jpg'
    depth_path = '/home/barbados/rc_car_new_data_mcgill/depth_data/depth_2023-06-13-17-23-24_0001.npy'

    img = cv2.imread(image_path)
    depth_image = np.load(depth_path,allow_pickle=True)

    lseg_planner = lseg_based_nav()
    lseg_planner.planning(img, depth_image, 'grass')

