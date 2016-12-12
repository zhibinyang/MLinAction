import kNN
import os
import re
import sys
from PIL import Image

# A template to decide two different type of pictures

def set_table(a):
    table = []
    table.extend([0]*a)
    table.extend([1]*(256-a))
    return table
#
#   Feature A: the character we use show feature in filename, so the pic names should be like L1.png, L2.png etc.
#   Feature B: Same to Feature A
#   Test: The validation data feature, like T1.png
#
FeatureA='L'
FeatureB='R'
Test = 'T'
training_data = []
answer_data = []
os.chdir(sys.path[0])
for file in os.listdir('kNN'):
    if re.search('[' + FeatureA + FeatureB + ']',file):
        if re.search('L',file):
            answer_data.append(FeatureA)
        else:
            answer_data.append(FeatureB)
        im = Image.open(os.path.join('kNN',file))
        img1 = im.convert("L")
        img2 = img1.point(set_table(140),'1')
        pix2 = img2.load()
        (width,height) = img2.size
        vector = []
        for h in range(height):
            for w in range(width):
                vector.append(pix2[w,h])
        training_data.append(vector)
for file in os.listdir('kNN'):
    if re.search(Test,file):
        print("filename",file)
        im = Image.open(os.path.join('kNN', file))
        img1 = im.convert("L")
        img2 = img1.point(set_table(140), '1')
        pix2 = img2.load()
        (width, height) = img2.size
        vector = []
        for h in range(height):
            for w in range(width):
                vector.append(pix2[w, h])
        kNN.kNN(training_data,answer_data,3,vector)