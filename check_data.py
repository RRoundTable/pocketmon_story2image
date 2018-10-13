
'''
dataloder.py 부분을 확인하기 위해서 만들어놓은 프로그램

dataloader를 어떻게 설계해야하는지 알아보기
'''



import os
import re
import time
import nltk
import re
import string
import tensorlayer as tl
from utils import *

nltk.download('punkt')

dataset = '102flowers'
need_256 = True # set to True for stackGAN



if dataset == '102flowers': # images
    """
    images.shape = [8000, 64, 64, 3]
    captions_ids = [80000, any]
    """
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '102flowers')
    caption_dir = os.path.join(cwd, 'text_c10')
    VOC_FIR = cwd + '/vocab.txt'

    # directory 확인하기
    # print("img dir : ",img_dir) # /tmp/pycharm_project_79/102flowers
    # print("caption_dir : ",caption_dir) # /tmp/pycharm_project_79/text_c10


    ## load captions
    caption_sub_dir = load_folder_list( caption_dir ) # load_folder_list : tensorlayer.file.load_file_list /Return a folder list in a folder by given a folder path.
    captions_dict = {}

    # caption_sub_dir
    # '/tmp/pycharm_project_79/text_c10/class_00011', '/tmp/pycharm_project_79/text_c10/class_00095', '/tmp/pycharm_project_79/text_c10/class_00020', '/tmp/pycharm_project_79/text_c10/class_00078', '/tmp/pycharm_project_79/text_c10/class_00002', '/tmp/pycharm_project_79/text_c10/class_00070', '/tmp/pycharm_project_79/text_c10/class_00023', '/tmp/pycharm_project_79/text_c10/class_00102', '/tmp/pycharm_project_79/text_c10/class_00017', '/tmp/pycharm_project_79/text_c10/class_00090', '/tmp/pycharm_project_79/text_c10/class_00049', '/tmp/pycharm_project_79/text_c10/class_00082', '/tmp/pycharm_project_79/text_c10/class_00033', '/tmp/pycharm_project_79/text_c10/class_00052', '/tmp/pycharm_project_79/text_c10/class_00026', '/tmp/pycharm_project_79/text_c10/class_00016', '/tmp/pycharm_project_79/text_c10/class_00063', '/tmp/pycharm_project_79/text_c10/class_00030', '/tmp/pycharm_project_79/text_c10/class_00046', '/tmp/pycharm_project_79/text_c10/class_00057', '/tmp/pycharm_project_79/text_c10/class_00014', '/tmp/pycharm_project_79/text_c10/class_00072', '/tmp/pycharm_project_79/text_c10/class_00079', '/tmp/pycharm_project_79/text_c10/class_00040', '/tmp/pycharm_project_79/text_c10/class_00092', '/tmp/pycharm_project_79/text_c10/class_00038', '/tmp/pycharm_project_79/text_c10/class_00037', '/tmp/pycharm_project_79/text_c10/class_00075', '/tmp/pycharm_project_79/text_c10/class_00050', '/tmp/pycharm_project_79/text_c10/class_00018', '/tmp/pycharm_project_79/text_c10/class_00029', '/tmp/pycharm_project_79/text_c10/class_00064', '/tmp/pycharm_project_79/text_c10/class_00077', '/tmp/pycharm_project_79/text_c10/class_00059', '/tmp/pycharm_project_79/text_c10/class_00099', '/tmp/pycharm_project_79/text_c10/class_00039', '/tmp/pycharm_project_79/text_c10/class_00054', '/tmp/pycharm_project_79/text_c10/class_00051', '/tmp/pycharm_project_79/text_c10/class_00028', '/tmp/pycharm_project_79/text_c10/class_00025', '/tmp/pycharm_project_79/text_c10/class_00101', '/tmp/pycharm_project_79/text_c10/class_00013', '/tmp/pycharm_project_79/text_c10/class_00055', '/tmp/pycharm_project_79/text_c10/class_00015', '/tmp/pycharm_project_79/text_c10/class_00035', '/tmp/pycharm_project_79/text_c10/class_00071', '/tmp/pycharm_project_79/text_c10/class_00005', '/tmp/pycharm_project_79/text_c10/class_00069', '/tmp/pycharm_project_79/text_c10/class_00084', '/tmp/pycharm_project_79/text_c10/class_00074', '/tmp/pycharm_project_79/text_c10/class_00065', '/tmp/pycharm_project_79/text_c10/class_00073', '/tmp/pycharm_project_79/text_c10/class_00007', '/tmp/pycharm_project_79/text_c10/class_00081', '/tmp/pycharm_project_79/text_c10/class_00010', '/tmp/pycharm_project_79/text_c10/class_00047', '/tmp/pycharm_project_79/text_c10/class_00024', '/tmp/pycharm_project_79/text_c10/class_00001', '/tmp/pycharm_project_79/text_c10/class_00080', '/tmp/pycharm_project_79/text_c10/class_00094', '/tmp/pycharm_project_79/text_c10/class_00043', '/tmp/pycharm_project_79/text_c10/class_00044', '/tmp/pycharm_project_79/text_c10/class_00041', '/tmp/pycharm_project_79/text_c10/class_00068', '/tmp/pycharm_project_79/text_c10/class_00062', '/tmp/pycharm_project_79/text_c10/class_00061', '/tmp/pycharm_project_79/text_c10/class_00032', '/tmp/pycharm_project_79/text_c10/class_00086', '/tmp/pycharm_project_79/text_c10/class_00093', '/tmp/pycharm_project_79/text_c10/class_00006', '/tmp/pycharm_project_79/text_c10/class_00009', '/tmp/pycharm_project_79/text_c10/class_00058', '/tmp/pycharm_project_79/text_c10/class_00031', '/tmp/pycharm_project_79/text_c10/class_00091', '/tmp/pycharm_project_79/text_c10/class_00083', '/tmp/pycharm_project_79/text_c10/class_00085', '/tmp/pycharm_project_79/text_c10/class_00036', '/tmp/pycharm_project_79/text_c10/class_00012', '/tmp/pycharm_project_79/text_c10/class_00076', '/tmp/pycharm_project_79/text_c10/class_00100', '/tmp/pycharm_project_79/text_c10/class_00060', '/tmp/pycharm_project_79/text_c10/class_00067', '/tmp/pycharm_project_79/text_c10/class_00019', '/tmp/pycharm_project_79/text_c10/class_00021', '/tmp/pycharm_project_79/text_c10/class_00066', '/tmp/pycharm_project_79/text_c10/class_00042', '/tmp/pycharm_project_79/text_c10/class_00048', '/tmp/pycharm_project_79/text_c10/class_00056', '/tmp/pycharm_project_79/text_c10/class_00087', '/tmp/pycharm_project_79/text_c10/class_00053', '/tmp/pycharm_project_79/text_c10/class_00022', '/tmp/pycharm_project_79/text_c10/class_00098', '/tmp/pycharm_project_79/text_c10/class_00003', '/tmp/pycharm_project_79/text_c10/class_00034', '/tmp/pycharm_project_79/text_c10/class_00089', '/tmp/pycharm_project_79/text_c10/class_00096', '/tmp/pycharm_project_79/text_c10/class_00027', '/tmp/pycharm_project_79/text_c10/class_00004', '/tmp/pycharm_project_79/text_c10/class_00008', '/tmp/pycharm_project_79/text_c10/class_00088', '/tmp/pycharm_project_79/text_c10/class_00045', '/tmp/pycharm_project_79/text_c10/class_00097']

    processed_capts = []
    for sub_dir in caption_sub_dir: # get caption file list
        with tl.ops.suppress_stdout(): # tl.ops.suppress_stdout() : contextmanager
            files = tl.files.load_file_list(path=sub_dir, regx='^image_[0-9]+\.txt') # regx : regular expression
            for i, f in enumerate(files):
                file_dir = os.path.join(sub_dir, f)
                key = int(re.findall('\d+', f)[0])
                t = open(file_dir,'r')
                lines = []
                for line in t:
                    line = preprocess_caption(line) # preprocess_caption : 어디서 왔는지 파악하기
                    lines.append(line)
                    processed_capts.append(tl.nlp.process_sentence(line, start_word="<S>", end_word="</S>")) # sentence별로 나누기
                assert len(lines) == 10, "Every flower image have 10 captions"
                captions_dict[key] = lines # 1 image 10 sentence
    # tl.ops.suppress_stdout() 이 끝나야 출력이 진행됨
    print(" * %d x %d captions found " % (len(captions_dict), len(lines)))
    # print(captions_dict[3558])
    # ['the petals on this flower are pink with pink stamen ', 'this flower has petals that are pink with pink stamen',
    #  'the five bright pink petals of this flower join at the receptacle  where darker pink filaments protrude  ending in black anthers ',
    #  'this flower is pink in color  with petals that are wavy ',
    #  'there are wrinkled fuchsia pedals and also slightly darker fuchsia stamen with black tips ',
    #  'this flower has petals that are pink and has red stamen',
    #  'this flower has wide and rounded rose colored petals which are semi sheer ',
    #  'this flower is pink in color  and has petals that are oval shaped and thin ',
    #  'this flower has five bright pink petals with dark pink spots on some of them and dark pink stamen ',
    #  'this flower has five bright pink petals with dark pink spots on some of them and dark pink stamen ']

    ## build vocab
    if not os.path.isfile('vocab.txt'):
        _ = tl.nlp.create_vocab(processed_capts, word_counts_output_file=VOC_FIR, min_word_count=1) # 단어사전 만들기
    else:
        print("WARNING: vocab.txt already exists")
    vocab = tl.nlp.Vocabulary(VOC_FIR, start_word="<S>", end_word="</S>", unk_word="<UNK>")

    # print(vocab.vocab)
    # tensorlayer object # 어떻게 디버깅 할지 확인하기

    ## store all captions ids in list
    captions_ids = []
    try: # python3
        tmp = captions_dict.items()
    except: # python3
        tmp = captions_dict.iteritems()
    for key, value in tmp:
        for v in value:
            # vocab에서 번호 부여하기
            captions_ids.append( [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(v)] + [vocab.end_id])  # add END_ID
            # print(v)              # prominent purple stigma,petals are white inc olor
            # print(captions_ids)   # [[152, 19, 33, 15, 3, 8, 14, 719, 723]]
            # exit()
    captions_ids = np.asarray(captions_ids)
    print(" * tokenized %d captions" % len(captions_ids)) #  * tokenized 81890 captions

    ## check
    img_capt = captions_dict[1][1]
    print("img_capt: %s" % img_capt)
    print("nltk.tokenize.word_tokenize(img_capt): %s" % nltk.tokenize.word_tokenize(img_capt))
    img_capt_ids = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(img_capt)]#img_capt.split(' ')]
    print("img_capt_ids: %s" % img_capt_ids)
    print("id_to_word: %s" % [vocab.id_to_word(id) for id in img_capt_ids])
# img_capt: this flower has bright purple  spiky petals  and greenish sepals below them
# nltk.tokenize.word_tokenize(img_capt): ['this', 'flower', 'has', 'bright', 'purple', 'spiky', 'petals', 'and', 'greenish', 'sepals', 'below', 'them']
# img_capt_ids: [6, 3, 7, 31, 18, 165, 4, 5, 318, 83, 374, 120]
# id_to_word: ['this', 'flower', 'has', 'bright', 'purple', 'spiky', 'petals', 'and', 'greenish', 'sepals', 'below', 'them']



    ## load images
    with tl.ops.suppress_stdout():  # get image files list
        imgs_title_list = sorted(tl.files.load_file_list(path=img_dir, regx='^image_[0-9]+\.jpg'))
    print(" * %d images found, start loading and resizing ..." % len(imgs_title_list)) # * 8189 images found, start loading and resizing ...
    s = time.time()
    # print("imgs_title_list : ",imgs_title_list) #  ['image_00001.jpg', 'image_00002.jpg', 'image_00003.jpg'] 이런 형식

    # time.sleep(10)
    # def get_resize_image(name):   # fail
    #         img = scipy.misc.imread( os.path.join(img_dir, name) )
    #         img = tl.prepro.imresize(img, size=[64, 64])    # (64, 64, 3)
    #         img = img.astype(np.float32)
    #         return img
    # images = tl.prepro.threading_data(imgs_title_list, fn=get_resize_image)
    images = []
    images_256 = []
    for name in imgs_title_list: # 한 이미지의 name 추출
        # print(name)
        img_raw = scipy.misc.imread( os.path.join(img_dir, name) )
        img = tl.prepro.imresize(img_raw, size=[64, 64])    # (64, 64, 3)
        img = img.astype(np.float32)
        images.append(img)
        if need_256: # 256 size가 필요하면
            img = tl.prepro.imresize(img_raw, size=[256, 256]) # (256, 256, 3)
            img = img.astype(np.float32)

            images_256.append(img)
    # images = np.array(images)
    # images_256 = np.array(images_256)
    print(" * loading and resizing took %ss" % (time.time()-s))
#
    n_images = len(captions_dict) # image 개수
    n_captions = len(captions_ids) # caption의 개수
    n_captions_per_image = len(lines) # 10 # image당 text의 개수

    print("n_captions: %d n_images: %d n_captions_per_image: %d" % (n_captions, n_images, n_captions_per_image))

    captions_ids_train, captions_ids_test = captions_ids[: 8000*n_captions_per_image], captions_ids[8000*n_captions_per_image :]
    images_train, images_test = images[:8000], images[8000:]
    if need_256:
        images_train_256, images_test_256 = images_256[:8000], images_256[8000:]
    n_images_train = len(images_train)
    n_images_test = len(images_test)
    n_captions_train = len(captions_ids_train)
    n_captions_test = len(captions_ids_test)
    print("n_images_train:%d n_captions_train:%d" % (n_images_train, n_captions_train))
    print("n_images_test:%d  n_captions_test:%d" % (n_images_test, n_captions_test))

    ## check test image
    # idexs = get_random_int(min=0, max=n_captions_test-1, number=64)
    # temp_test_capt = captions_ids_test[idexs]
    # for idx, ids in enumerate(temp_test_capt):
    #     print("%d %s" % (idx, [vocab.id_to_word(id) for id in ids]))
    # temp_test_img = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # save_images(temp_test_img, [8, 8], 'temp_test_img.png')
    # exit()

    # ## check the first example
    # tl.visualize.frame(I=images[0], second=5, saveable=True, name='temp', cmap=None)
    # for cap in captions_dict[1]:
    #     print(cap)
    # print(captions_ids[0:10])
    # for ids in captions_ids[0:10]:
    #     print([vocab.id_to_word(id) for id in ids])
    # print_dict(captions_dict)

    # ## generate a random batch
    # batch_size = 64
    # idexs = get_random_int(0, n_captions_test, batch_size)
    # # idexs = [i for i in range(0,100)]
    # print(idexs)
    # b_seqs = captions_ids_test[idexs]
    # b_images = images_test[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # print("before padding %s" % b_seqs)
    # b_seqs = tl.prepro.pad_sequences(b_seqs, padding='post')
    # print("after padding %s" % b_seqs)
    # # print(input_images.shape)   # (64, 64, 64, 3)
    # for ids in b_seqs:
    #     print([vocab.id_to_word(id) for id in ids])
    # print(np.max(b_images), np.min(b_images), b_images.shape)
    # from utils import *
    # save_images(b_images, [8, 8], 'temp2.png')
    # # tl.visualize.images2d(b_images, second=5, saveable=True, name='temp2')
    # exit()

import pickle
def save_all(targets, file):
    with open(file, 'wb') as f:
        pickle.dump(targets, f)

save_all(vocab, '_vocab.pickle')
print(vocab)
save_all((images_train_256, images_train), '_image_train.pickle')
save_all((images_test_256, images_test), '_image_test.pickle')
save_all((n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test), '_n.pickle')
save_all((captions_ids_train, captions_ids_test), '_caption.pickle')
