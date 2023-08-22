import torch
import warnings
import os
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
from MGT import MGT
from data import TestDataset, YCrCb2RGB

warnings.filterwarnings("ignore")
EPSILON = 1e-5
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def main():
    # GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型路径
    model_path = './Model/MGT/MGT_epoch_100.pth'
    # Dataset
    Vis_RGB = False  # 可见光图像是否为RGB图像
    testset_type = "MSRS"  # MSRS/TNO
    if testset_type == 'MSRS':
        Vis_RGB = True

    # 加载/初始化模型
    model = MGT(Ex_depths=3, Fusion_depths=3, Re_depths=3)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 加载数据集
    test_path = os.path.join('./Dataset/test/', testset_type)
    print('Loading test dataset from {}.'.format(test_path))
    testset = TestDataset(test_path, Vis_RGB)
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    test_tqdm = tqdm(testloader, total=len(testloader))

    # 创建融合结果保存路径
    fused_image_save_path = os.path.join('./result/', testset_type)
    if os.path.exists(fused_image_save_path) is not True:
        os.makedirs(fused_image_save_path)

    # 开始测试
    print('Staring testing on {}'.format(device))
    if Vis_RGB:
        for vis_y_image, vis_cb_image, vis_cr_image, inf_image, name in test_tqdm:
            _, _, H, W = vis_y_image.shape
            vis_y_image = vis_y_image.to(device)
            inf_image = inf_image.to(device)
            with torch.no_grad():
                img_fusion = model(inf_image, vis_y_image)
                # img_fusion = (img_fusion - torch.min(img_fusion)) / (
                #         torch.max(img_fusion) - torch.min(img_fusion) + EPSILON)
                img_fusion = img_fusion.cpu()
                img_fusion = YCrCb2RGB(img_fusion[0], vis_cb_image[0], vis_cr_image[0])
                img_fusion = img_fusion * 255
                out_path = fused_image_save_path + '/' + name[0]
                cv2.imwrite(out_path, img_fusion.numpy())
    else:
        for vis_image, inf_image, name in test_tqdm:
            _, _, H, W = vis_image.shape
            vis_image = vis_image.to(device)
            inf_image = inf_image.to(device)
            with torch.no_grad():
                img_fusion = model(inf_image, vis_image)
                # img_fusion = (img_fusion - torch.min(img_fusion)) / (
                #         torch.max(img_fusion) - torch.min(img_fusion) + EPSILON)
                img_fusion = img_fusion[0].cpu().numpy().transpose(1, 2, 0)
                img_fusion = img_fusion * 255
                out_path = fused_image_save_path + '/' + name[0]
                cv2.imwrite(out_path, img_fusion)

    print('Done.')


if __name__ == '__main__':
    main()
