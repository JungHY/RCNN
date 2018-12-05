import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import skimage.data
import selectivesearch

from mrcnn import model as modellib

## Bounding Box regression model
class BBOX :
    def __init__(self, input_shape) :
        """
        CNN의 출력 fixed-length feature vector를 입력으로 Bounding Box를 표현하는 (x, y, w, h)를 출력
        input_shape = keras.layers.CNN(특정 레이어).output_shape
        """
        self.input = keras.layers.Input(shape=input_shape)

    def train(self, input_feature_vectors, ground_truth_bboxs) :
        """
        Bounding Box Regressor model 학습
        optimizer, loss, metrics 지정
        """
    def build(self, ) :
        return None

    def feedforward(self, input_feature_vector) :
        return None
    


## RCNN 모델을 만들고 학습하고 테스트하는 클래스
class R_CNN :
    def __init__(self, input_shape, output_class) :
        self.input_shape = input_shape
        ### + 1 for background class
        self.output_class = output_class + 1
        self.input_image = Input(shape=input_shape, name="input_image")

    def getIoU(self, scRegion, gtRegion) :
        """
        selective search 알고리즘의 region과 ground truth region 간의 IoU를 계산
        """
        ## bb = [x1, y1, x2, y2]
        scRegion_rect = scRegion['rect']
        bb1 = [scRegion_rect[0], scRegion_rect[1], scRegion_rect[2], scRegion_rect[3]]
        bb2 = [gtRegion[0], gtRegion[1], gtRegion[0]+gtRegion[2], gtRegion[1]+gtRegion[3]]

        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top :
            return 0.0

        intersection_erea = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        iou = intersection_erea / float(bb1_area + bb2_area - intersection_erea)
        assert iou >= 0.0
        assert iou <= 1.0

        return iou



    def scRegionProposal(self, input_image, gtRegions) :
        """
        input_image를 입력으로 selective search알고리즘으로 2000개의 candidate region 탐색 후
        ground truth와 비교하여 background인지 object인지 판별
        """

        ### selective search output
        ### [ {'labels': [0.1], 'rect':(0, 0, 15, 25), 'size':260}, --- ]
        scRegions = selectivesearch.selective_search(input_image)

        ### candidate regions 와 background regions
        proposalRegions_idxs = []
        backgroundRegions_idxs = []

        ### selective search region과 ground truth region의 IoU를 기준으로
        ### 0.7 이상이면 object가 있는 region으로 판단하고
        ### 0.2 미만이면 backgroun region으로 판단
        region_idxs = 0
        for scRegion in scRegions :
            for gtRegion in gtRegions :
                if self.getIoU(scRegion, gtRegion) >= 0.7 :
                    proposalRegions_idxs.append(region_idxs)
                elif self.getIoU(scRegion, gtRegion) < 0.2 :
                    backgroundRegions_idxs.append(region_idxs)
            region_idxs += 1

        proposalRegions = [scRegions[i] for i in proposalRegions_idxs]
        backgrounRegions = [scRegions[i] for i in backgroundRegions_idxs]

        return proposalRegions, backgrounRegions




    ## model 만들기 
    def BuildRCNN(self, architecture) :
        C1,C2,C3,C4,C5 = modellib.resnet_graph(input_image=self.input_image, architecture=architecture, stage5=True, train_bn=True)
    

    ## model 훈련
    def TrainRCNN(self, input_images_dir) :

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
    def TestRCNN(self, input_image) :
        return None

    ## model 저장
    def SaveModel(self) :
        return None

    def LoadModel(self) :
        return None