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
    img_raw=img_raw.convert('RGB')
    img=img_raw.resize((64,64))
    #img = tl.prepro.imresize(img_raw, size=[64, 64])   # (64, 64, 3)
    img=np.asarray(img, dtype=float)
    #img = img.astype(np.float32)
    # print(type(img[0][0][0])) #float32

    images.append(img)
    if need_215:
        img = Image.open(name) # (215, 215, 4)
        img_raw = img_raw.convert('RGB')
        img = np.asarray(img, dtype=float)

        images_215.append(img)
    # images = np.array(images)
    # images_256 = np.array(images_256)
    # print(" * loading and resizing took %ss" % (time.time()-s))


print("images : ",len(images))
print("images_215 : ",len(images_215))

print(images)
import pickle
def save_all(path, data):
    # save pkl
    f = open(path, 'wb')
    pickle.dump(data, f)
    f.close()


# train/test 분배하기
train=[]
test=[]

train_215=[]
test_215=[]

train_idx=[]
test_idx=[]
for i in range(len(images)):
    if i%2==0:
        train.append(images[i])
        train_215.append(images_215[i])
        train_idx.append(i)
    else :
        test.append(images[i])
        test_215.append(images_215[i])
        test_idx.append(i)


save_all("./data/image_train.pkl",train) # 원격에 저장된다 그렇다면 어떻게 해야하는가 : deployment  : 소프트웨어 전개
save_all("./data/image_train_215.pkl",train_215)
save_all("./data/image_test.pkl",test) # 원격에 저장된다 그렇다면 어떻게 해야하는가 : deployment  : 소프트웨어 전개
save_all("./data/image_test_215.pkl",test_215)

save_all("./data/train_idx.pkl",train_idx   ) # 원격에 저장된다 그렇다면 어떻게 해야하는가 : deployment  : 소프트웨어 전개
save_all("./data/test_idx.pkl",test_idx)
# save_all((images_train_256, images_train), '_image_train.pickle')
# save_all((images_test_256, images_test), '_image_test.pick
# save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
