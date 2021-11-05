import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
from PIL import Image
import PIL.ImageOps
import numpy as np


x = np.load(r'C:\Users\vedat\OneDrive\Documents\My Coding Stuff\Python\Flask\Class125\Project\image.npz')['arr_0']
y = pd.read_csv(r'C:\Users\vedat\OneDrive\Documents\My Coding Stuff\Python\Flask\Class125\Project\labels.csv')['labels']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 7500, test_size = 2500)

x_train = x_train/255.0
x_test = x_test/255.0

logreg = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train, y_train)

def alphabet_pred_boy(image):
    pil_img = Image.open(image)
    image_bw = pil_img.convert('L')
    image_bw_resize = image_bw.resize((28, 28), Image.ANTIALIAS)
    
    img_bw_rsz_invert = PIL.ImageOps.invert(image_bw_resize)

    pixel_filter = 20
    min_pixel = np.percentile(img_bw_rsz_invert, pixel_filter)

    im_bw_rs_inv_scaled = np.clip(img_bw_rsz_invert-min_pixel, 0, 255)
    
    max_pixel = np.max(img_bw_rsz_invert)

    im_bw_rs_inv_scaled = np.asarray(im_bw_rs_inv_scaled)
    test_sample = np.array(im_bw_rs_inv_scaled).reshape(1, 784)
    test_pred = logreg.predict(test_sample)
    return test_pred[0]
