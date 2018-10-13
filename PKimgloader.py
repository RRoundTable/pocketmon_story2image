import os
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *
import glob
#  예시 : print(glob.glob("/home/adam/*.txt"))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

img_dir='./data/image/*.png'
need_215 = True # set to True for stackGAN
#os._exists(img_dir)
## load images
# with tl.ops.suppress_stdout():  # get image files list
#     imgs_title_list = sorted(tl.files.load_file_list(path=img_dir))
# print(" * %d images found, start loading and resizing ..." % len(imgs_title_list))

imgs_title_list=sorted(glob.glob(img_dir))

s = time.time()

# time.sleep(10)
# def get_resize_image(name):   # fail
#         img = scipy.misc.imread( os.path.join(img_dir, name) )
#         img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
#         img = img.astype(np.float32)
#         return img
# images = tl.prepro.threading_data(imgs_title_list, fn=get_resize_image)
images = []
images_215 = []
for name in imgs_title_list:
    # print("name : ", name)
    img_raw=Image.open(name)
    #img_raw.shape= (215, 215, 4) ? 왜 channel이 4인가 : png는 4개의 channel을 가진다
    img=img_raw.resize((64,64))
    #img = tl.prepro.imresize(img_raw, size=[64, 64])   # (64, 64, 3)
    img=np.asarray(img, dtype=float)
    plt.imshow(img)
    #img = img.astype(np.float32)
    # print(type(img[0][0][0])) #float32

    images.append(img)
    if need_215:
        img = Image.open(name) # (215, 215, 4)
        img = np.asarray(img, dtype=float)

        images_215.append(img)
    # images = np.array(images)
    # images_256 = np.array(images_256)
    print(" * loading and resizing took %ss" % (time.time()-s))


import pickle
def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

# save_all(vocab, '_vocab.pickle')
# save_all((images_train_256, images_train), '_image_train.pickle')
# save_all((images_test_256, images_test), '_image_test.pickle')
# save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
# save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
