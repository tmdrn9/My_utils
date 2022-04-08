# My_utils
딥러닝 시 자주 사용하는 함수들을 코드들을 모듈로 정리해놓는 곳

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
