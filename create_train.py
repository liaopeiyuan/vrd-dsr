
# coding: utf-8

# In[1]:


import pandas as pd
from ensemble import GeneralEnsemble
from tqdm import tqdm
from object_detection.utils import label_map_util
import os
import numpy as np
from imageio import imread
import random
import cPickle


# In[6]:
classes = pd.read_csv("challenge-2018-classes-vrd.csv")
dicts = {}
for i in range(len(classes)):
    row=classes.loc[i]
    dicts[row['ImageID']]=i

"""
PATH_TO_LABELS = os.path.join('data', 'oid_bbox_trainable_label_map.pbtxt')
NUM_CLASSES = 546

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
inv_dicts = label_map_util.get_label_map_dict(PATH_TO_LABELS,
                       use_display_name=False,
                       fill_in_gaps_and_background=False)
dicts = {v: k for k, v in inv_dicts.items()}
ROOT = "/home/alexanderliao/data/Kaggle/competitions/google-ai-open-images-visual-relationship-track/challenge2018_test/"
"""

# In[2]:


df=pd.read_csv("challenge-2018-train-vrd.csv")


# In[12]:


df=df.sort_values(by=['ImageID'])
df.loc[1:3,:]


# In[9]:
rels = ['at','on','holds','plays','interacts_with','wears','is','inside_of','under','hits']

ROOT_SEV="/data/train/"
train=[]
currIm = ""
curr_dict= {'img_path':"", 'classes':[], 'boxes':[], 'ix1':[], 'ix2':[], 'rel_classes':[]}
for r in tqdm(range(325000)):
    row = df.loc[r]
    #print(type(row))
    if currIm == "":
        currIm = row['ImageID']
        curr_dict['img_path']=ROOT_SEV+currIm+".jpg"
    if not(row['ImageID']==currIm):
        if not(curr_dict['boxes']==[]):
            train.append(curr_dict)
        currIm=""
        curr_dict= {'img_path':"", 'classes':[], 'boxes':[], 'ix1':[], 'ix2':[], 'rel_classes':[]}
    else:
        if row['LabelName1'] in dicts.keys() and row['LabelName2'] in dicts.keys():
            box1 = [row['XMin1'],row['YMin1'],row['XMax1'],row['YMax1']]
            if box1 in curr_dict['boxes']:
                indx1 = curr_dict['boxes'].index(box1)
            else:
                curr_dict['boxes'].append(box1)
                indx1 = len(curr_dict['boxes'])-1
            box2 = [row['XMin2'],row['YMin2'],row['XMax2'],row['YMax2']]
            if box2 in curr_dict['boxes']:
                indx2 = curr_dict['boxes'].index(box2)
            else:
                curr_dict['boxes'].append(box2)
                indx2 = len(curr_dict['boxes'])-1

            curr_dict['ix1'].append(indx1)
            curr_dict['ix2'].append(indx2)
            curr_dict['rel_classes'].append([rels.index(row["RelationshipLabel"])])
            img1 = dicts[row['LabelName1']]
            img2 = dicts[row['LabelName2']]
            if not(img1 in classes):
                curr_dict['classes'].append(img1)
            if not(img2 in classes):
                curr_dict['classes'].append(img2)
        #print(train)
print(len(train))
train_new=[]
for curr_dict in train:
    #print(curr_dict)
    curr_dict['classes'] = np.array(curr_dict['classes'])
    curr_dict['boxes'] = np.array(curr_dict['boxes'])
    curr_dict['ix1'] = np.array(curr_dict['ix1'])
    curr_dict['ix2'] = np.array(curr_dict['ix2'])
    curr_dict['rel_classes'] = np.array(curr_dict['rel_classes'])
    train_new.append(curr_dict)
cPickle.dump( train_new, open( "train_vrd.pkl", "wb" ) )
