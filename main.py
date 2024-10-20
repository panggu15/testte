import os
import random
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dicom_nii_2d_dataset import DicomNii2DDataset
import albumentations as A
from albumentations import Compose

import segmentation_models_pytorch as smp
from score import evaluate_model

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data")
args = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42) # Seed 고정

UNET_RESIZE = 256

train_transform = A.Compose([A.ShiftScaleRotate(p = 0.5),
                           A.HorizontalFlip(0.5),
                           A.VerticalFlip(0.5),
                           A.RandomBrightnessContrast(0.5),
                           A.GaussNoise(p = 0.5),
                           A.GaussianBlur(p=0.5),
                           A.ElasticTransform(),
                           A.Resize(height=UNET_RESIZE, width=UNET_RESIZE),
                           A.Normalize()
                          ])

test_transform = A.Compose(
    [
        A.Resize(height=UNET_RESIZE, width=UNET_RESIZE),
        A.Normalize(),
    ]
)
import copy
def train(model, num_epochs, dataloader, valid_loader, optimizer, criterion, device, scheduler=None):
    """
    모델 학습
    """

    train_losses = []  # 손실 값
    train_gds = []  # gds
    train_miou = []  # miou
    
    best_score = 0
    best_model = None

    # Training loop
    for epoch in range(num_epochs):
        # 모델을 학습 모드로 전환
        model.train()

        # 손실 초기화
        running_loss = 0.0

        #
        running_gds = 0.0
        running_miou = 0.0

        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.float().to(device)
            images = images.unsqueeze(1)  # bhw to b1hw

            masks = masks.float().to(device)
            masks = masks.unsqueeze(1)  # bhw to b1hw

            # 기울기 초기화
            optimizer.zero_grad()

            # 모델 예측
            outputs = model(images)

            # 손실 계산
            loss = criterion(outputs, masks)

            # 역전파
            loss.backward()

            # 가중치 업데이트
            optimizer.step()

            # 손실 누적
            running_loss += loss.item()

            # 점수 계산
            gds, miou = evaluate_model(outputs, masks, device)
            running_gds += gds
            running_miou += miou

        length = len(dataloader)

        # 에포크 손실 출력
        epoch_loss = running_loss / length
        train_losses.append(epoch_loss)

        # gds
        epoch_gds = running_gds / length
        train_gds.append(epoch_gds)

        # miou
        epoch_miou = running_miou / length
        train_miou.append(epoch_miou)
        
        valid_gds, valid_miou = evaluate(model, valid_loader, device)
        
        if best_score < valid_gds:
            best_score = valid_gds
            best_model = copy.deepcopy(model)
        
        if scheduler is not None:
            scheduler.step(epoch_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, GDS: {epoch_gds:.8f}, mIoU: {epoch_miou:.8f}, val_GDS: {valid_gds:.8f}, val_mIoU: {valid_miou:.8f}"
        )

    return train_losses, train_gds, train_miou, best_score, best_model

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

# 하이퍼파라미터(Hyperparameters) 설정
num_epochs = 10
learning_rate = 3e-4
BATCH_SIZE = 32

# 경로에 맞게 수정 필요
# ./drive/MyDrive/2024/maithon_2024/smart_health_care2/train/breast
train_dataset = DicomNii2DDataset(os.path.join(args.data_path, "breast"), train_transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)

valid_dataset = DicomNii2DDataset(os.path.join(args.data_path, "breast"), test_transform)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

model = smp.Unet(
    encoder_name="resnext50_32x4d",  # Encoder로 resnext50_32x4d 사용
    encoder_weights="imagenet",  # 사전 학습된 가중치 사용
    in_channels=1,  # 입력 채널 수 (흑백 이미지)
    classes=1  # 출력 채널 수 (이진 분할이므로 1)
).to(device)

criterion = smp.losses.DiceLoss(mode='binary')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

train_losses, train_gds, train_miou, best_score, best_model = train(
    model, num_epochs, train_dataloader, valid_dataloader, optimizer, criterion, device, scheduler
)


from datetime import datetime


today_date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

# 팀 이름 수정
team = "알파코"

# 학습이 완료된 모델의 state_dict(모델 가중치와 매개변수 정보)을 파일로 저장
torch.save(
    best_model.state_dict(),  # 저장할 모델의 상태 딕셔너리
    f"{today_date_str}_{team}_model_complete_state_dict_{num_epochs:04}.pth",
)
