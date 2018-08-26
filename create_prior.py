import pandas as pd
from ensemble import GeneralEnsemble
from tqdm import tqdm
from object_detection.utils import label_map_util
import os
import numpy as np
from imageio import imread
import random
import cPickle

classes = pd.read_csv("challenge-2018-classes-vrd.csv")
print(len(classes))
rel_cand = ['at','on','holds','plays','interacts_with','wears','is','inside_of','under','hits']
triplets = pd.concat([pd.read_csv("challenge-2018-relationship-triplets.csv"),
                      pd.read_csv("challenge-2018-initial-relationship-triplets.csv")])
triplets = triplets.drop_duplicates(['LabelName1','LabelName2','RelationshipLabel'])
print(triplets)
obj1 = []
obj2 = []
rels = []
objs = []

dicts = {}
for i in range(len(classes)):
    row=classes.iloc[i]

    dicts[row['ImageID']]=i

inv_dicts = {v: k for k, v in dicts.items()}

for i in range(len(triplets)):
    row = triplets.iloc[i]
    #print(type(row))
    #print(dicts.keys())
    #print(row['LabelName1'])
    if row['LabelName1'] in dicts.keys() and row['LabelName2'] in dicts.keys():
        #print(1)
        obj1.append(dicts[row['LabelName1']])
        obj2.append(dicts[row['LabelName2']])
        objs.append(str(dicts[row['LabelName1']])+str(dicts[row['LabelName2']]))
        rels.append(rel_cand.index(row['RelationshipLabel']))

prior = np.zeros((len(classes),len(classes),10))

for i in tqdm(range(len(classes))):
    for j in range(len(classes)):
        for r in range(len(rel_cand)):
            if str(i)+str(j) in objs and r == rels[objs.index(str(i)+str(j))]:
                prior[i,j,r]=1

for i in tqdm(range(len(classes))):
    for j in range(len(classes)):
        for r in range(len(rel_cand)):
            sum = np.sum(prior[i,j,:])
            if not(sum==0):
                print(sum)
                prior[i,j,r]=prior[i,j,r]/sum
print(np.sum(prior))
#print(dicts)
#print(inv_dicts)
#print(classes)
cPickle.dump( prior, open( "so_prior_oid.pkl", "wb" ) )
with open('so_prior.pkl', 'rb') as fid:
    so_prior = cPickle.load(fid)
#print(so_prior)
