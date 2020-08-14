#!/usr/bin/env python
# coding: utf-8

# In[25]:


import os
import cv2
import numpy as np
#os.remove("gender_train/female/female_152.jpg")
#os.remove("gender_train/male/male_152.jpg")


# In[26]:


X=[]
y=[]
path_male="gender_train/male/"
img_names=os.listdir(path_male)
for name in img_names:
    gray=cv2.imread(path_male+name,0)
    X.append(gray.flatten())
    y.append('male')
    
path_female="gender_train/female/"
img_names=os.listdir(path_female)
for name in img_names:
    gray=cv2.imread(path_female+name,0)
    X.append(gray.flatten())
    y.append('female')
    
X=np.array(X)
y=np.array(y)


# In[27]:


X.shape


# In[28]:


from sklearn.decomposition import PCA


# In[31]:


pca=PCA(.99)
new_X=pca.fit_transform(X)
new_X.shape


# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


log=LogisticRegression()
log.fit(new_X,y)


# In[49]:


face_clf=cv2.CascadeClassifier("c:/dataset/haar/haarcascade_frontalface_default.xml")
img=cv2.imread("e:/images/players/logo.jpg",cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_clf.detectMultiScale(gray)
if(len(faces)==0):
    print("Face Not Detected")
else:
    for x,y,w,h in faces:
        face=gray[y:y+h,x:x+w]
        face=cv2.resize(face,(90,90))
        face=face.reshape(1,8100)
        face=pca.transform(face)
        print(log.predict(face))


# In[ ]:




