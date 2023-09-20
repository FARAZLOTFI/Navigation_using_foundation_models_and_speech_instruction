import clip
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch
import os
from lseg import LSegNet
import time
import cv2
import numpy as np
def get_new_pallete(num_colors):
    """Generate a color pallete given the number of colors needed. First color is always black."""
    pallete = []
    for j in range(num_colors):
        lab = j
        r, g, b = 0, 0, 0
        i = 0
        while lab > 0:
            r |= ((lab >> 0) & 1) << (7 - i)
            g |= ((lab >> 1) & 1) << (7 - i)
            b |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
        pallete.append([r, g, b])
    return torch.tensor(pallete).float() / 255.0


if __name__ == "__main__":

    # Initialize the model
    net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
    # Load pre-trained weights
    net.load_state_dict(torch.load('/home/barbados/lseg-minimal/examples/checkpoints/lseg_minimal_e200.ckpt'))
    net.eval()
    net.cuda()

    # Preprocess the text prompt
    clip_text_encoder = net.clip_pretrained.encode_text
    # prompts = ["other"]  # begin with the catch-all "other" class
    # label_classes = set()
    # for _c in args.segclasses.split(","):
    #     if _c != "other":
    #         label_classes.add(_c)
    # label_classes = list(label_classes)
    # label_classes.insert(0, "other")
    # print(f"Classes of interest: {label_classes}")
    # if len(label_classes) == 1:
    #     raise ValueError("Need more than 1 class")
    label_classes = ['road', 'other']
    # Cosine similarity module
    cosine_similarity = torch.nn.CosineSimilarity(dim=1)
    images_folder = '/home/barbados/offroad_terrain_dataset_kaggle/archive/TrainingImages/TrainingImages/OriginalImages/'
    images_list = os.listdir(images_folder)
    images_list.sort()
    with torch.no_grad():

        # Extract and normalize text features
        prompt = [clip.tokenize(lc).cuda() for lc in label_classes]
        text_feat_list = [clip_text_encoder(p) for p in prompt]
        text_feat_norm_list = [
            torch.nn.functional.normalize(tf) for tf in text_feat_list
        ]
        for image in images_list:
        # Load the input image
            image_size = (640, 480)
            img = cv2.imread(images_folder + image)#str(args.query_image))
            img_shape = img.shape
            row_fac = 3
            col_fac = 1.5
            img = img[-row_fac*image_size[1]:, int(img_shape[1]/2 - col_fac*image_size[0]):int(img_shape[1]/2 + col_fac*image_size[0]),:]
            begin_time = time.time()
            print(f"Original image shape: {img.shape}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, image_size)
            img = torch.from_numpy(img).float() / 255.0
            img = img[..., :3]  # drop alpha channel, if present
            img = img.cuda()
            img = img.permute(2, 0, 1)  # C, H, W
            img = img.unsqueeze(0)  # 1, C, H, W
            print(f"Image shape: {img.shape}")

            # Extract per-pixel CLIP features (1, 512, H // 2, W // 2)
            img_feat = net.forward(img)
            # Normalize features (per-pixel unit vectors)
            img_feat_norm = torch.nn.functional.normalize(img_feat, dim=1)
            print(f"Extracted CLIP image feat: {img_feat_norm.shape}")

            # Compute cosine similarity across image and prompt features
            similarities = []
            for _i in range(len(label_classes)):
                similarity = cosine_similarity(
                    img_feat_norm, text_feat_norm_list[_i].unsqueeze(-1).unsqueeze(-1)
                )
                similarities.append(similarity)

            similarities = torch.stack(
                similarities, dim=0
            )  # num_classes, 1, H // 2, W // 2
            similarities = similarities.squeeze(1)  # num_classes, H // 2, W // 2
            similarities = similarities.unsqueeze(0)  # 1, num_classes, H // 2, W // 2
            class_scores = torch.max(similarities, 1)[1]  # 1, H // 2, W // 2
            class_scores = class_scores[0].detach()

            output_mask = np.expand_dims(class_scores.cpu().numpy(),-1).astype('float32')
            ################to be saved############################
            saved_img = np.ones(shape=(img_shape[0],img_shape[1]))
            saved_img[-row_fac * image_size[1]:, int(img_shape[1] / 2 - col_fac*image_size[0]):int(img_shape[1] / 2 + col_fac*image_size[0])] \
                = cv2.resize(output_mask, (image_size[0]*row_fac,image_size[1]*row_fac))

            #######################################################
            cv2.imwrite('./results/'+image[:-4]+'.png', saved_img * 255)
            print(f"class scores: {class_scores.shape}")

            # pallete = get_new_pallete(len(label_classes))
            #
            # # img size // 2 for height and width dims
            # disp_img = torch.zeros(int(image_size[1]/2), int(image_size[0]/2), 3)
            # for _i in range(len(label_classes)):
            #     disp_img[class_scores == _i] = pallete[_i]
            # rawimg = cv2.imread(images_folder + image)#str(args.query_image))
            # img_shape = rawimg.shape
            # rawimg = rawimg[-2 * image_size[1]:, int(img_shape[1] / 2 - image_size[0]):int(img_shape[1] / 2 + image_size[0]), :]
            #
            # rawimg = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
            # rawimg = cv2.resize(rawimg, (int(image_size[0]/2), int(image_size[1]/2)))
            # rawimg = torch.from_numpy(rawimg).float() / 255.0
            # rawimg = rawimg[..., :3]  # drop alpha channel, if present
            #
            # disp_img = 0.5 * disp_img + 0.5 * rawimg
            #
            # cv2.imshow('result',disp_img.detach().cpu().numpy())
            # cv2.waitKey(1)
            # plt.imshow(disp_img.detach().cpu().numpy())
            # plt.legend(
            #     handles=[
            #         mpatches.Patch(
            #             color=(
            #                 pallete[i][0].item(),
            #                 pallete[i][1].item(),
            #                 pallete[i][2].item(),
            #             ),
            #             label=label_classes[i],
            #         )
            #         for i in range(len(label_classes))
            #     ]
            # )
            # plt.show()
            # input(' ')
            print('time: ',time.time() - begin_time)