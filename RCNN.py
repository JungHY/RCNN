import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import skimage.data
import selectivesearch
import os
import random
from random import shuffle

from mrcnn import model as modellib

RANDOM_SEED = 10

## Bounding Box regression model
class BBOX :
    def __init__(self, input) :
        """
        CNN의 출력 fixed-length feature vector를 입력으로 Bounding Box를 표현하는 (x, y, w, h)를 출력
        input = CNN의 출력 feature vectors
        output = bbox(x, y, w, h)
        """
        self.input = input
        self.build(self.input)

    def train(self, input_feature_vectors, ground_truth_bboxs, epoch=10, batch_size=10) :
        """
        Bounding Box Regressor model 학습
        optimizer, loss, metrics 지정
        """

        self.model.fit(input_feature_vectors, ground_truth_bboxs, epoch=epoch, batch_size=batch_size)


    def build(self, input ) :

        self.O1 = Flatten()(input)
        self.O2 = Dense(100, activation='relu')(self.O1)
        self.O3 = Dense(32, activation='relu')(self.O2)
        self.output = Dense(4, activation='relu')(self.O3)

        self.model = keras.models.Model(inputs=input, outputs=self.output)
        ### model compile
        self.model.compile(loss='mean_squared_error', optimizer='sgd')

    def feedforward(self, input_feature_vector) :
        """
        CNN의 출력 fixed-length feature vector를 입력으로 Bounding Box를 출력
        """
        bbox = self.model.predict(input_feature_vector)

        return bbox

    def getOutput(self) :

        return self.output

    def save(self, name) :

        self.model.save(name + '.h5')

    
class CLASSIFIER :
    def __init__(self, input, class_num) :
        """
        CNN의 출력 fixed-length feature vector를 입력으로 proposal region이 어떤 class에 속하는지 출력
        input = CNN의 출력 feature vectors
        output = class 분류 값
        class_num = class의 개수
        """
        self.input = input
        self.class_num = class_num

    def build(self, input, class_num) :
        """
        classifier model build
        """

        self.O1 = Flatten()(input)
        self.O2 = Dense(100, activation='relu')(self.O1)
        self.O3 = Dense(32, activation='relu')(self.O2)
        self.output = Dense(class_num+1, activation='softmax')

        self.model = keras.models.Model(inputs=input, outputs=self.output)
        self.model.compile(loss='categorical_crossentropy', optimizer='sgd')

    def train(self, input_feature_vectors, ground_truth_classes, epoch=10, batch_size=10) :
        """
        Proposal Region Classifier model 학습
        """
        
        self.model.fit(input_feature_vectors, ground_truth_classes, epoch=epoch, batch_size=batch_size)

    def feedforward(self, input_feature_vector) :
        
        output_class = self.model.predict(input_feature_vector)

        return output_class

    def getOutput(self) :
        return self.output

    def save(self, name) :
        
        self.model.save(name + '.h5')


## RCNN 모델을 만들고 학습하고 테스트하는 클래스
class R_CNN :
    def __init__(self, warp_shape, output_class) :
        self.warp_shape = warp_shape
        ### + 1 for background class
        self.output_class = output_class + 1
        self.input_image = Input(shape=warp_shape, name="input_image")

        ### build CNN model & compile model
        self.CNN_layers = self.BuildRCNN("resnet101")
        ### build BBOX model & compile model
        self.BBOX_model = BBOX(self.CNN_layers[4])
        ### build Classifier model & compile model
        self.CLS_model = CLASSIFIER(self.CNN_layers[4], self.output_class)
        ### build Total model
        self.TOTAL_model = keras.models.Model(inputs=self.input_image, outputs=[self.BBOX_model.getOutput(), self.CLS_model.getOutput()])
        self.TOTAL_model.compile(loss='mean_squared_error', optimizer='sgd')

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



    def scRegionProposal(self, input_image, gtRegions, gtClasses) :
        """
        input_image를 입력으로 selective search알고리즘으로 2000개의 candidate region 탐색 후
        ground truth와 비교하여 background인지 object인지 판별
        """

        ### selective search output
        ### [ {'labels': [0.1], 'rect':(0, 0, 15, 25), 'size':260}, --- ]
        _, scRegions = selectivesearch.selective_search(input_image)

        ### candidate regions 와 background regions
        proposalRegions_idxs = []
        backgroundRegions_idxs = []
        proposalBBoxes_idxs = []

        ### selective search region과 ground truth region의 IoU를 기준으로
        ### 0.7 이상이면 object가 있는 region으로 판단하고
        ### 0.2 미만이면 backgroun region으로 판단
        region_idxs = 0
        for scRegion in scRegions :
            bbox_idxs = 0
            for gtRegion in gtRegions :
                if self.getIoU(scRegion, gtRegion) >= 0.7 :
                    proposalRegions_idxs.append(region_idxs)
                    proposalBBoxes_idxs.append(bbox_idxs)
                elif self.getIoU(scRegion, gtRegion) < 0.2 :
                    backgroundRegions_idxs.append(region_idxs)
                bbox_idxs += 1
            region_idxs += 1

        proposalRegions = [scRegions[i] for i in proposalRegions_idxs]
        backgrounRegions = [scRegions[i] for i in backgroundRegions_idxs]
        proposalBBoxes = [gtRegions[i] for i in proposalBBoxes_idxs]
        proposalRegionsclasses = [gtClasses[i] for i in proposalRegions_idxs]

        return proposalRegions, backgrounRegions, proposalRegionsclasses, proposalBBoxes

    def RegionWarp(self, input_image, region) :

        ### input_image 원본 이미지
        ### warping 할 영역 (x, y, h, w)
        ### warping 할 크기 warp_shape
        x, y, w, h = region

        region_image = input_image[y:y+h, x:x+w]
        warp_region_image = skimage.transform.warp(region_image, skimage.transform.SimilarityTransform, output_shape=self.warp_shape)

        return warp_region_image


    ## model 만들기 
    def BuildRCNN(self, architecture) :
        self.C1,self.C2,self.C3,self.C4,self.C5 = modellib.resnet_graph(input_image=self.input_image, architecture=architecture, stage5=True, train_bn=True)

        CNN_layers = [self.C1, self.C2, self.C3, self.C4, self.C5]
        self.F1 = Flatten()(self.C5)
        self.D1 = Dense(128, activation='relu')(self.F1)
        self.D2 = Dense(self.output_class, activation='softmax')(self.D1)

        self.CNNmodel = keras.models.Model(inputs=self.input_image, outputs=self.D2)
        self.CNNmodel.compile(loss='categorical_crossentropy', optimizer='sgd')

        self.model = keras.models.Model(inputs=self.input_image, outputs=self.C5)
        self.model.compile(loss='mean_squared_error', optimizer='sgd')
        return CNN_layers

    def preTrain(self, input_images_dir, input_images_classes, gtBBoxes) :
        """
        전체 모델중 CNN의 weight를 미리 학습

        input_images_classes 구조
        [input_image 개수][gtBBox 개수][class]

        gtBBoxes 구조
        [input_image 개수][gtBBox 개수][x, y, w, h]
        """
        input_images_names = os.listdir(input_images_dir)

        image_idx = 0
        for input_image_name in input_images_names :
            input_image = skimage.io.imread(input_image_name)
            for i in range(len(input_images_classes[image_idx])) :
                x, y, w, h = gtBBoxes[image_idx][i]
                region_class = input_images_classes[image_idx][i]
                ### region 추출
                warped_region = self.RegionWarp(input_image, (x,y,w,h))
                ### region 학습
                self.CNNmodel.fit(x=warped_region, y=region_class, epochs=1)
            image_idx += 1

        return None

    def RegionShuffle(self, class_regions, background_regions) :

        shuffled_regions = class_regions + background_regions

        return shuffled_regions


    ## model 훈련
    def TrainCNN(self, input_images_dir, input_images_classes, gtBBoxes) :

        """
        
        input_images_classes 구조
        [input_image 개수][gtBBox 개수][class]
        
        gtBBoxes 구조
        [input_image 개수][gtBBox 개수][x, y, w, h]
        """

        ## 1. image directory에서 이미지들 불러오기
        ## 2. (필요시) 픽셀값 0 ~ 1 normalize
        ## 3. selective search 사용해서 region proposal (2 천개)
        ## 4. IoU값 구하기
        ## 5. ground truth의 bounding box와 비교해서 IoU가 0.7 이상인 region에 대해서 추출
        ## 6. ground truth의 bounding box와 비교해서 IoU가 0.7 미만인 region의 대해서 추출
        ## 7. positive class와 negative class의 비율 조절
        ## 8. ResNet 모델 학습


        ## 

        self.preTrain(input_images_dir, input_images_classes, gtBBoxes)

        input_images_names = os.listdir(input_images_dir)

        idx = 0
        for input_image_name in input_images_names :
            ## image open
            input_image = skimage.io.imread(input_image_name)
            ## proposal region, backgroun 추출
            objectclassRegions, backgroundclassRegions, objectRegionclasses, objectclassBBoxes  = self.scRegionProposal(input_image, gtBBoxes[idx], gtBBoxes[idx])
            idx += 1
            proposal_regions_num = len(objectclassRegions)
            if len(objectclassRegions)/3 > len(backgroundclassRegions) :
                background_regions_num = len(backgroundclassRegions)
            else :
                background_regions_num = len(objectclassRegions) / 3

            ## object class Region과 background class Region을 비율에 맞게 mix
            merged_regions = objectclassRegions + backgroundclassRegions[:background_regions_num]
            merged_classes = objectRegionclasses + [0 for num in range(background_regions_num)]
            merged_bboxes = objectclassBBoxes + [None for i in range(background_regions_num)]

            ## shuffle
            random.Random(RANDOM_SEED).shuffle(merged_regions)
            random.Random(RANDOM_SEED).shuffle(merged_classes)
            random.Random(RANDOM_SEED).shuffle(merged_bboxes)

            idx = 0
            for region in merged_regions :
                predicted_feature_vectors = self.model.predict(x=region)
                self.BBOX_model.train(input_feature_vectors=predicted_feature_vectors, ground_truth_bboxs=merged_bboxes[idx], epoch=1, batch_size=1)
                self.CLS_model.train(input_feature_vectors=predicted_feature_vectors, ground_truth_classes=merged_classes[idx], epoch=1, batch_size=1)
                idx += 1  

        return None

    ## image test
    def TestRCNN(self, input_image) :

        predicted_classes = []
        predicted_bboxes = []

        if type(input_image) == type('') :
            input_image = skimage.io.imread(input_image)


        _, sc_regions = selectivesearch.selective_search(input_image)
        proposal_regions = []
        for region in sc_regions :
            proposal_regions.append([region['rect'][0], region['rect'][1], region['rect'][2]- region['rect'][0], region['rect'][3] - region['rect'][1] ])

        warped_region = []
        for region in proposal_regions :
            warped_region.append(self.RegionWarp(input_image, region))

        for region in warped_region :
            bbox, class_ = self.TOTAL_model.predict(region)
            predicted_bboxes.append( bbox )
            predicted_classes.append( class_ )

        predict_results = []
        for idx in range(len(predicted_bboxes)) :
            predict_results.append([predicted_bboxes[idx][0], predicted_bboxes[idx][1], predicted_bboxes[idx][2], predicted_bboxes[idx][3], predicted_classes[idx]])

        return predict_results

    ## model 저장
    def SaveModel(self) :
        self.TOTAL_model.save('total_model.h5')

    ### mac book git test