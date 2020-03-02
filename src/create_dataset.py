#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import os
import pandas as pd
import numpy as np
df = pd.DataFrame(columns=['filename','width','height','class','xmin','ymin','xmax','ymax'])


def get_coords(img1,img,name,df,cl):
    c1,c2,c3 = np.nonzero(img1)
    for i in range(0,c1.shape[0]):
        if cl=='car':
            color = (0,255,0)
        else:
            color = (255,0,0)
        cv2.rectangle(img, (c2[i]-34,c1[i]-34),(c2[i]+34,c1[i]+34) ,color, 2) 
        xmin,ymin,xmax,ymax =c2[i]-34,c1[i]-34,c2[i]+34,c1[i]+34
        w=xmax-xmin
        h= ymax-ymin
        
        r = max(w, h) / 2
        centerx = xmin + w / 2
        centery = ymin + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        car = img[ny:ny+nr, nx:nx+nr]
        try:
            cv2.imwrite('try_{}.png'.format(cl),car)
            df = df.append({'filename':filename,'width':w,'height':h, 'class': cl, 'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax}, ignore_index=True)
        except:
            continue
    #cv2.imwrite(name,img)
    return df,img

cities=['Potsdam_ISPRS']
for city in cities:
    path = '../data/cowc/datasets/ground_truth_sets/{}/'.format(city)
    files = os.listdir(path)
    file_names = [fn for fn in files if 'RGB.png' in fn]
    for filename in file_names:
        print(filename)
        img1 = cv2.imread(path+filename.split('.')[0]+'_Annotated_Cars.png')
        img = cv2.imread(path+filename)  
        df,img = get_coords(img1,img,'{}_cars.png'.format(filename),df,'car')
        img1 = cv2.imread(path+filename.split('.')[0]+'_Annotated_Negatives.png')
        df,img = get_coords(img1,img,'{}_nocars.png'.format(filename),df,'nocar')
        cv2.imwrite(filename,img)
df = df[df['class']=='car']
df.to_csv('../data/labels.csv',index=False)

df = pd.read_csv('../data/labels.csv')

grouped = df.groupby('filename')
grouped.apply(lambda x: len(x)).value_counts()

#split each file into a group in a list

gb = df.groupby('filename')

grouped_list = [gb.get_group(x) for x in gb.groups]

len(grouped_list)

train_index = np.random.choice(len(grouped_list), size=9, replace=False)
test_index = np.setdiff1d(list(range(9)), train_index)
len(train_index), len(test_index)

# take first 200 files
train = pd.concat([grouped_list[i] for i in train_index])
test = pd.concat([grouped_list[i] for i in test_index])

len(train), len(test)

train.to_csv('../data/train_labels.csv', index=None)
test.to_csv('../data/test_labels.csv', index=None)

