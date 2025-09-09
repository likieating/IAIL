import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img
from models.multi_attention import MAT
# from convs.xception_dmp import xception
from convs.xception import xception
def main():
    # model = xception(2)
    path = "/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/dmp/0111-17-50-45-019_OSMA_split1_resnet32_1993_B0_Inc1/task2/val_acc87.24_model.pth"

    #path = "/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/dmp/0309-14-55-25-751_OSMA_split1_xception_1993_B0_Inc1/task2/val_acc82.99_model.pth"
    model=xception(2)
    #path="/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/dfil_xception/0112-21-16-36-244_OSMA_split1_resnet32_1993_B0_Inc1/task2/val_acc85.63_model.pth"
    path="/home/tangshuai/dmp-bk/logs/benchmark/OSMA/split1/dfil_xception/0305-11-10-28-859_OSMA_split1_xception_1993_B0_Inc1/task2/val_acc88.05_model.pth"
    state = torch.load(path, map_location=torch.device("cuda:2"))
    model.load_state_dict(state,strict=False)
    target_layers = [model.conv4]
    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # load image
    #img_path = "/home/tangshuai/dmp-bk/grad_cam/dataa/01_02__talking_against_wall__YVGY8LOK_frame_55_face_0.png"
    #img_path="/home/tangshuai/dmp-bk/grad_cam/dataa/01_02__talking_angry_couch__YVGY8LOK_frame_58_face_0.png"
    #img_path = "/home/tangshuai/dmp-bk/grad_cam/dataa/01_27__outside_talking_still_laughing__ZYCZ30C0_frame_10_face_0.png"
    #img_path = "/home/tangshuai/dmp-bk/grad_cam/dataa/02_03__podium_speech_happy__QH3Y0IG0_frame_14_face_0.png"
    #img_path = "/data/ssd2/tangshuai/DFIL299/DFD/test/fake/01_20__kitchen_still__D8GWGO2A_frame_8_face_0.png"
    img_path="/home/tangshuai/dmp-bk/grad_cam/dataa/02_03__podium_speech_happy__QH3Y0IG0_frame_0_face_0.png"

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    # img = center_crop_img(img, 224)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 1  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()