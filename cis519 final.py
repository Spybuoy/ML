import cv2
import os
import time
import tqdm
import torch
import IPython
import torchvision
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from torchvision import models

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import AlexNet
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder

from PIL import Image
from io import BytesIO
import pandas as pd

X=[]
y=[]
sim=[]
def get_most_common_name(names):
    all_names = set(names);
    most_common = max([names.count(i) for i in all_names]);
    for name in all_names:
        if names.count(name) == most_common:
            return name;
#####################################################################################################################

alexnet = models.alexnet(pretrained=True)

#####################################################################################################################
from gensim.models import KeyedVectors
filename = 'GoogleNewsvectorsnegative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)
####################################################################################################################
# #Downloading set frames to a specified folder in cv2.imwrite() function
df=pd.read_csv("train_data_10000.csv")
tags = df['Tags']
t=[]
for i in range(2415):
  # tags = str(df['Tags'][i])

  try:
    cap = cv2.VideoCapture(f"{i}.f278.webm")
  except ValueError:
    continue
  fps = int(cap.get(cv2.CAP_PROP_FPS))
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  for b in range(10):
      arr_frame=[]
      arr_lap=[]
      for j in range(fps):
          success, frame = cap.read()
          try:
            laplacian = cv2.Laplacian(frame, cv2.CV_64F).var()
        
            arr_lap.append(laplacian)
            arr_frame.append(frame)
          except:
            continue
      try:
        selected_frame = arr_frame[arr_lap.index(max(arr_lap))]
      except ValueError:
        continue

      cv2.imwrite(os.path.join(r"E:\\CIS 519\\project\\New folder\\temp_frames" , f"{b}.jpg"), selected_frame)
  
  #####################################################################################################################
  #Embeddings
  word_embed = []
  with open('imagenet_classes.txt') as f:

    labels = [line.strip() for line in f.readlines()]
  preprocess_image = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
  )])

  #####################################################################################################################
  final=[]
  for a in range(10):
    try:
      image = Image.open(os.path.join(r"E:\\CIS 519\\project\\New folder\\temp_frames" ,f"{a}.jpg"))
    except FileNotFoundError:
      continue
    image_tensor = preprocess_image(image)
    image_tensor = preprocess_image(image)
    input_tensor = torch.unsqueeze(image_tensor, 0)
    alexnet.eval()

    prediction_tensor = alexnet(input_tensor)
    max_value, index_of_max_value = torch.max(prediction_tensor, 1)
  # print(index_of_max_value.numpy())

    predicted_label = labels[index_of_max_value]
    final.append(predicted_label)
    # print(predicted_label)
  # print(f"{i}.f278.webm")
  # print("X : ",final)
  tl= list(str(tags[i]).split("|"))
  # print("y",tl)

  print("######################################################################################")
  tag = get_most_common_name(final)


  tlist = tag.split()

  max_similarity = 0
  for i in range(0,len(tlist)):
    if tlist[i] in model.vocab:
      v_tag = model[tlist[i]]
      for j in range(0,len(tl)):
        if tl[j] in model.vocab: 
          v_object = model[tl[j]]
          similarity = cosine_similarity([v_tag],[v_object])
          if similarity > max_similarity:
            max_similarity = similarity 
            tag_final = tl[j]
            object_final = tlist[i]
  # print(tag_final)
  # print(object_final)
  # print(max_similarity)
  X.append(tag_final)
  y.append(object_final)
  sim.append(max_similarity)

#####################################################################################################################
#Removing the frames from the specified folder
# for i in range(10):
#     try: 
#         os.remove(os.path.join(r"E:\\CIS 519\\project\\New folder\\temp_frames",f"{i}.jpg"))
#     except: pass

# print(X)
# print(y)

df=pd.DataFrame()
df['X']=pd.Series(X)
df['y']=pd.Series(y)
df['max_sim']=pd.Series(sim)
df.to_csv("out.csv")