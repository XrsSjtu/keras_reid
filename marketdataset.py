# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

testset_dir = "/home/xurongsen/Dataset/market/bounding_box_test/"
queryset_dir = "/home/xurongsen/Dataset/market/query/"
trainset_dir = "/home/xurongsen/Dataset/market/bounding_box_train/"
# testset_dir = "D:/xrs_sjtu/icat/boat/Re-identification/DataSet/DukeMTMC-reID/DukeMTMC-reID/bounding_box_test/"
# queryset_dir = "D:/xrs_sjtu/icat/boat/Re-identification/DataSet/DukeMTMC-reID/DukeMTMC-reID/query/"
# trainset_dir = "D:/xrs_sjtu/icat/boat/Re-identification/DataSet/DukeMTMC-reID/DukeMTMC-reID/bounding_box_train/"

def format_id(id):
    if(id<10):
        return "000"+str(id)
    if(id<100):
        return "00"+str(id)
    if(id<1000):
        return "0"+str(id)
    else:
        return str(id)

def get_img():
    query_imgs = []
    query_labels = []
    query_list = os.listdir(queryset_dir)
    for i in range(0,len(query_list)):   #读文件夹下的所有图
        img_dir = os.path.join(queryset_dir,query_list[i])
        if os.path.isfile(img_dir):
            person_id = os.path.splitext(query_list[i])[0].split('_')[0]  #取每张图的ID
            query_img = cv2.imread(img_dir)
            query_img = cv2.resize(query_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            query_img = np.reshape(query_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)

            query_imgs.append(query_img)
            query_labels.append(int(person_id))
    #
    #     #print(person_id,photo_id)

    test_imgs = []
    test_labels = []
    test_list = os.listdir(testset_dir)
    for j in range(0,len(test_list)):
        img_dir = os.path.join(testset_dir,test_list[j])
        if os.path.isfile(img_dir):
            person_id = os.path.splitext(test_list[j])[0].split('_')[0]
            test_img = cv2.imread(img_dir)
            test_img = cv2.resize(test_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            test_img = np.reshape(test_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)

            test_imgs.append(test_img)
            test_labels.append(int(person_id))
    #         #print(person_id,photo_id)
    return np.asarray(query_imgs, np.float32),np.asarray(test_imgs, np.float32),\
           np.asarray(query_labels, np.int32),np.asarray(test_labels, np.int32)
    #return np.asarray(query_imgs, np.float32),np.asarray(query_labels, np.int32)
    #return np.asarray(test_imgs, np.float32),np.asarray(test_labels, np.int32)
def get_triplet_data():
    train_imgs = []
    train_labels = []

def get_triplet_hard_data(SN,PN):  # PN表示一个batch有多少人，SN表示一个人有多少图
    train_imgs = []
    train_labels = []
    train_list = os.listdir(trainset_dir)
    id_list = []

    for i in range(0,len(train_list)):
        person_id = os.path.splitext(train_list[i])[0].split('_')[0]
        id_list.append(int(person_id))

    id_list = list(set(id_list)) #去重

    pIDset = np.random.choice(id_list,PN,False)
    # print("pIDset:",pIDset)
    for pID in pIDset:
        img_list = [img for img in train_list if img.startswith(format_id(pID))]
        for img_name in np.random.choice(img_list,SN):
            img_dir = os.path.join(trainset_dir,img_name)
            # print("img idr :",img_dir)
            if (os.path.isfile(img_dir)):
                train_img = cv2.imread(img_dir)
                train_img = cv2.resize(train_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)
                train_img = np.reshape(train_img, (IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
                train_imgs.append(train_img)
                train_labels.append(pID)
    return np.asarray(train_imgs, np.float32), np.asarray(train_labels, np.int32)


# query_img,test_img,query_label,test_label=get_img()
# #query_img,query_label = get_img()
# print(len(query_img))
# print(len(test_img))
# print(len(query_label))
# print(len(test_label))
# train_img,train_label = get_triplet_hard_data(3,24)
# print(train_img)
# print(train_label)
# print(len(train_label))