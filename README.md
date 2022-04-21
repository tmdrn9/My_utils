# My_utils
딥러닝 시 자주 사용하는 코드,함수, 그리고 클래스들을 모아놓은 저장소

✔ 프로그래밍 언어 : Python

✔ 라이브러리 : Pytorch, albumentations

----

## Dataset.py

ClassficationDataset : x는 이미지 데이터, y는 정형데이터일때 쓰는 데이터셋

ImageDataset : x,y 둘 다 이미지 데이터일 경우에 쓰는 데이터셋


````
# Dataset 사용예시
train_dataset = ImageDataset(train_x,train_y,mytransform)
````

<br>

추가적으로 augmentation시에는 torchvision.transforms보다는 albumentations 라이브러리를 애용합니다

가장 대표적인 이유로는 x_img와 y_img에 같은 함수를 적용할 수 있기 때문입니다

![image](https://miro.medium.com/max/1750/1*5uLc6odMwOVO4OVyLUjigA.jpeg)

````
augmentation = albumentations.Compose([
      #원하는 함수들
      ],additional_targets={'target_image':'image'})
````

위와 같은 코드로 변수를 만들어 ImageDataset의 transform매개변수의 인자로 넣어주면 위 그림과 같이 x_img와 y_img에 같은 함수가 적용됩니다


두번째 이유로는 속도가 더 빠릅니다 [**[참고 사이트]**](https://pseudo-lab.github.io/Tutorial-Book/chapters/object-detection/Ch3-preprocessing.html)


## utils.py

get_mean_std : Nomalize하기 위해 평균과 표준편차를 계산해주는 함수

imshow : 이미지의 shape이 C,W,H일때 시각화해주는 함수

check_dataloader : dataloader가 잘 작동하는지, Data와 Label이 맞게 나오는지 확인하는 함수

rand_bbox : cutmix하기 위해 필요한 함수

````
# cutmix 사용 예시

for e in range(0, n_epochs):
    ###################
    # train the model #
    ###################
    model.train()
    for data, labels in tqdm(train_dataloader):
        # move tensors to GPU if CUDA is available
        data, labels = data.to(device), labels.to(device)
        
        if np.random.random()>0.5:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(data.size()[0]).to(device)
            target_a = labels
            target_b = labels[rand_index]            
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
            data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
            logits = model(data)
            loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)

        else :
            logits = model(data)
            loss = criterion(logits, labels)

````

## multiTaskLearningNet.py

multi task learning 시 불러서 사용하는 네트워크 모듈

-- 수정중--
