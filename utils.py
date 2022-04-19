import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mean_std(data_dir_list):

    """
    이미지 정규화하기 위해 평균과 표준편차를 구하는 함수

    :param data_dir_list: 데이터 주소 리스트
    :return: 데이터셋의 채널별 평균과 표준편차

    """

    meanRGB = np.array([0,0,0], np.float64)
    stdRGB = np.array([0,0,0], np.float64)

    for i in data_dir_list:

        x_img = cv2.imread(i)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

        meanRGB += np.mean(x_img, axis=(0,1))
        stdRGB += np.std(x_img, axis=(0,1))

    meanRGB /= len(data_dir_list)
    stdRGB /= len(data_dir_list)

    print("평균 : ", meanRGB[0], meanRGB[1], meanRGB[2])
    print("표준편차 : ", stdRGB[0], stdRGB[1], stdRGB[2])

    return meanRGB, stdRGB

def imshow(img,mean=0,std=1):
    """
    :param img: 시각화하고자 하는 이미지
    :param mean: 이미지 정규화 시 사용한 평균값. 1차원 리스트 또는 1차원 배열타입으로 입력
    :param std: 이미지 정규화 시 사용한 표준편차값. 1차원 리스트 또는 1차원 배열타입으로 입력

    """

    img = img.numpy().transpose((1, 2, 0))

    mean = np.array(mean)
    std = np.array(std)

    img = std * img + mean
    img = np.clip(img, 0, 1)

    plt.imshow(img)
    plt.show()

def check_dataloader(dataloader,mode=0):

    """
    dataloader가 잘 작동하는지, Feature와 Label이 맞게 나오는지 확인하는 함수

    :param dataloader:
    :param mode: 0이면 dataloader가 이미지데이터와 정형데이터를 반환할때, 1이면 dataloader가 이미지데이터와 이미지데이터를 반환할때

    """

    images, labels= next(iter(dataloader))
    print("image shape : ", images.shape)
    print("label shape : ", labels.shape)
    plt.figure(figsize=(16, 18))

    if mode==0:
        imshow(images[0])
        print(labels[0])

    else:
        imshow(images[0])
        imshow(labels[0])
        
def rand_bbox(size, lam):
    """
    argumentation cutmix사용할때 쓰는 
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2 
