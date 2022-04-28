import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mean_std(data_path):

    """
    이미지 정규화하기 위해 평균과 표준편차를 구하는 함수
    :param data_path: 데이터 주소 리스트, ImageFolder사용 시에는 데이터 주소 String
    :return: 데이터셋의 채널별 평균과 표준편차
    """

    meanRGB = np.array([0,0,0], np.float64)
    stdRGB = np.array([0,0,0], np.float64)

    for i in data_path:

        x_img = cv2.imread(i)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        x_img = x_img.astype(np.float64)/255.0

        meanRGB += np.mean(x_img, axis=(0,1))
        stdRGB += np.std(x_img, axis=(0,1))

    meanRGB /= len(data_path)
    stdRGB /= len(data_path)

    print("평균 : ", meanRGB[0], meanRGB[1], meanRGB[2])
    print("표준편차 : ", stdRGB[0], stdRGB[1], stdRGB[2])
    

    # # imageFolder로 데이터 불러오기
    # train_ds = datasets.ImageFolder(root = data_path, transform=transforms.ToTensor())
    # meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x,_ in train_ds]
    # stdRGB = [np.std(x.numpy(), axis=(1,2)) for x,_ in train_ds]

    # # 이미지 각 mean과 std의 전체적인 평균 값
    # meanR = np.mean([m[0] for m in meanRGB])
    # meanG = np.mean([m[1] for m in meanRGB])
    # meanB = np.mean([m[2] for m in meanRGB])

    # stdR = np.mean([s[0] for s in stdRGB])
    # stdG = np.mean([s[1] for s in stdRGB])
    # stdB = np.mean([s[2] for s in stdRGB])

    # print(meanR, meanG, meanB)
    # print(stdR, stdG, stdB)
    
    
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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device) # 주어진 수 내 랜덤하게 자연수 생성
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def predict(model, test_loader, device):
    model.eval()
    model_pred = []
    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)

            pred_logit = model(img)
            pred_logit = pred_logit.squeeze(1).detach().cpu()

            model_pred.extend(pred_logit.tolist())
            
    return model_pred
