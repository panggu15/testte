import os
import torch
from torch.utils.data import DataLoader
from dicom_nii_2d_dataset import DicomNii2DDataset
from score import evaluate_model

import albumentations as A

import segmentation_models_pytorch as smp

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data")
parser.add_argument('--model_path', type=str, default="model.pth")
args = parser.parse_args()

UNET_RESIZE = 256

transform = A.Compose(
    [
        A.Resize(height=UNET_RESIZE, width=UNET_RESIZE),
        A.Normalize(),
    ]
)

BATCH_SIZE = 16

# 검증 시에는 shuffle=False
test_dataset = DicomNii2DDataset(os.path.join(args.data_path, "breast"), transform)
test_dataloader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

model = smp.Unet(
    encoder_name="resnext50_32x4d",  # Encoder로 resnext50_32x4d 사용
    encoder_weights="imagenet",  # 사전 학습된 가중치 사용
    in_channels=1,  # 입력 채널 수 (흑백 이미지)
    classes=1  # 출력 채널 수 (이진 분할이므로 1)
).to(device)

# state_dict 로드
model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

def evaluate(model, dataloader, device):
    """
    모델 평가
    """

    model.eval()

    #
    inference_gds = 0.0
    inference_miou = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.float().to(device)
            images = images.unsqueeze(1)  # bhw to b1hw

            masks = masks.float().to(device)
            masks = masks.unsqueeze(1)  # bhw to b1hw

            # 모델 예측
            outputs = model(images)

            # 점수 계산
            gds, miou = evaluate_model(outputs, masks, device)
            inference_gds += gds
            inference_miou += miou

        length = len(dataloader)

        # gds
        epoch_gds = inference_gds / length

        # miou
        epoch_miou = inference_miou / length

    print(f"GDS: {epoch_gds:.8f}, mIoU: {epoch_miou:.8f}")
    return epoch_gds, epoch_miou

# 평가
evaluate(model, test_dataloader, device)
