import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import skimage.data
import selectivesearch

import resnet

## RCNN 모델을 만들고 학습하고 테스트하는 클래스
class R_CNN :
    def __init__(self, input_shape, output_class) :
        self.input_shape = input_shape
        self.output_class = output_class
        self.background_class = 1

    ## model 만들기 
    def BuildModel(self, model_number) :
        if model_number == 50 :
            self.model = resnet.ResnetBuilder.build_resnet_50(self.input_shape, self.output_class)
            return True
        elif model_number == 101 :
            self.model = resnet.ResnetBuilder.build_resnet_101(self.input_shape, self.output_class)

    ## model 훈련
    def TrainModel(self, input_images_dir) :

        ## 1. image directory에서 이미지들 불러오기
        ## 2. (필요시) 픽셀값 0 ~ 1 normalize
        ## 3. selective search 사용해서 region proposal (2 천개)
        ## 4. IoU값 구하기
        ## 5. ground truth의 bounding box와 비교해서 IoU가 0.7 이상인 region에 대해서 추출
        ## 6. ground truth의 bounding box와 비교해서 IoU가 0.7 미만인 region의 대해서 추출
        ## 7. positive class와 negative class의 비율 조절
        ## 8. ResNet 모델 학습
        return None

    ## image test
    def TestModel(self, input_image) :
        return None

    ## model 저장
    def SaveModel(self) :
        return None