import numpy as np
from PIL import Image
from feature import *
import os


direct = r"./datasets/original/nonface"
nonface_features = []
for filename in os.listdir(direct):
    img = Image.open(direct + "/" + filename).convert('L')
    img = img.resize((24, 24))
    img = np.array(img)

    img_ndp = NPDFeature(img)
    this_feature = img_ndp.extract()
    nonface_features.append(this_feature)
    print(len(nonface_features))

np.save("./datasets/nonface_features.npy", nonface_features)