#! /usr/bin/python
# -*- coding: utf8 -*-

""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk
from konlpy.tag import Twitter
from utils import *
from model import *
import model
import matplotlib.pyplot as plt

twitter=Twitter()

###======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
import pickle


# 나의 vocab 가져오기
with open("./data/word2index.pkl", 'rb') as f:
    vocab = pickle.load(f)

# train/test image
with open("./data/image_train.pkl", 'rb') as f:
    images_train = pickle.load(f)
with open("./data/image_test.pkl", 'rb') as f:
    images_test = pickle.load(f)

# test/train index
with open("./data/test_idx.pkl", 'rb') as f:
    captions_ids_train = pickle.load(f)
with open("./data/train_idx.pkl", 'rb') as f:
    captions_ids_test = pickle.load(f)
# caption load
with open("./data/posc2idx.pkl", 'rb') as f:
    posc2idx = pickle.load(f)

# 데이터 확인하기 : 301개의 이미지 데이터까지만 로드
images_train=images_train[:151]
images_test=images_test[:150]
train_idx=captions_ids_train[:151]
test_idx=captions_ids_test[:150]

# image channel 변환 하기
print("channel : ",images_train[0].shape)

ni = int(np.ceil(np.sqrt(batch_size)))
n_images_train=len(images_train)
n_images_test=len(images_test)
n_captions_train=len(train_idx)
n_captions_test=len(test_idx)

for i,story in enumerate(posc2idx):
    posc2idx[i]=[sen for sen in posc2idx[i][:12]] # 12문장이 최소 문장의 갯수 : story끼리 묶지 않기!

# flatten

posc2idx=[sentence for story in posc2idx for sentence in story]


print(".......데이터 확인하기.......")
print("images_train : ",len(images_train)) # 151개의 이미지
print("images_test : ",len(images_test)) # 150개의 이미지
print("vocab: ", len(vocab)) # 76303 의 pos(형태소 개수)
print("posc2idx : ",len(posc2idx)) # 301의 스토리
print("test_idx : " ,len(test_idx)) # 150
print("train_idx : " , len(train_idx)) # 151
print(train_idx)
print(test_idx)
captions_ids_train=[posc2idx[id-1] for id in train_idx]
captions_ids_test=[posc2idx[id+1] for id in test_idx]

n_story=len(posc2idx)
save_dir="./save_dir"

def main_train():
    ###======================== DEFIINE MODEL ===================================###
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise') # generator input

    ## training inference for text-to-image mapping
    net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
    x = net_cnn.outputs
    v = rnn_embed(t_real_caption, is_train=True, reuse=False).outputs
    x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs
    v_w = rnn_embed(t_wrong_caption, is_train=True, reuse=True).outputs

    alpha = 0.2 # margin alpha
    rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
                tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))

    ## training inference for txt2img
    generator_txt2img = model.generator_txt2img_resnet
    discriminator_txt2img = model.discriminator_txt2img_resnet

    net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=True)
    net_fake_image, _ = generator_txt2img(t_z,
                    net_rnn.outputs,
                    is_train=True, reuse=False, batch_size=batch_size)
                    #+ tf.random_normal(shape=net_rnn.outputs.get_shape(), mean=0, stddev=0.02), # NOISE ON RNN
    net_d, disc_fake_image_logits = discriminator_txt2img(
                    net_fake_image.outputs, net_rnn.outputs, is_train=True, reuse=False)
    _, disc_real_image_logits = discriminator_txt2img(
                    t_real_image, net_rnn.outputs, is_train=True, reuse=True)
    _, disc_mismatch_logits = discriminator_txt2img(
                    # t_wrong_image,
                    t_real_image,
                    # net_rnn.outputs,
                    rnn_embed(t_wrong_caption, is_train=False, reuse=True).outputs,
                    is_train=True, reuse=True)

    ## testing inference for txt2img
    net_g, _ = generator_txt2img(t_z,
                    rnn_embed(t_real_caption, is_train=False, reuse=True).outputs,
                    is_train=False, reuse=True, batch_size=batch_size)

    d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits,  tf.zeros_like(disc_mismatch_logits), name='d2')
    d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
    g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')

    ####======================== DEFINE TRAIN OPTS ==============================###
    lr = 0.003
    lr_decay = 0.3      # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100   # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5

    cnn_vars = tl.layers.get_variables_with_name('cnn', True, True)
    rnn_vars = tl.layers.get_variables_with_name('rnn', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    g_vars = tl.layers.get_variables_with_name('generator', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars )
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars )
    # e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(e_loss, var_list=e_vars + c_vars)
    grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
    optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)# optimizer = tf.train.GradientDescentOptimizer(lre)
    rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

    # adam_vars = tl.layers.get_variables_with_name('Adam', False, True)

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    # load the latest checkpoints # error 발생
    net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
    net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
    load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
    load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
    load_and_assign_npz(sess=sess, name=net_d_name, model=net_d)

    ## seed for generation, z and sentence ids
    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
        # sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)]
    n = int(sample_size / ni)

    # 포켓몬 스토리를 input으로 넣기 : 원래 문장으로 넣기
    sample_sentence=["발끝에서 기름이 베어 나와 물 위를 미끄러지듯 걸을 수 있다."]*n+\
                    ["연못이나 호수의 미생물을 먹고 있다."]*n + \
                    ["위험을 감지하면 머리끝에서 물엿같이 달콤한 액체가 나온다."] * n + \
                    ["머리 앞쪽에서 물엿과 비슷한 달콤한 냄새의 액체를 낸다."] * n + \
                    ["소나기가 온 뒤에 웅덩이에 모여들고 있다."] * n + \
                    ["물의 표면을 미끌어 지듯이 가서 머리로부터 달콤한 향기의 꿀을 나오게 한다."] * n + \
                    ["보통은 연못에서 살고 있지만 소나기가 온 뒤에는 마을 안의 물웅덩이에 모습을 드러낸다."] * n + \
                    ["머리 앞쪽에서 물엿과 비슷한 달콤한 냄새의 액체를 낸다. "] * n




    # sample_sentence = captions_ids_test[0:sample_size]
    for i, sentence in enumerate(sample_sentence):
        print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab['<start>']]+[vocab[word[0]] for word in twitter.pos(sentence)] + [vocab['<end>']]    # add END_ID : end -sign을 부여해야 하는가
        # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
        # print(sample_sentence[i])

    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    n_epoch = 10000 # 600
    print_freq = 4
    n_batch_epoch = int(n_images_train / batch_size)
    # exit()
    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()
            ## get matched text
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)

            b_real_caption=[captions_ids_train[id-1] for id in idexs] # 여러개의 스토리 추출 : batch_size = 문장의 개수
           # 문장의 집합으로 바꿔줘야 한다. 하지만 스토리마다 문장길이가 다른 문제를 해결해야한다. -> 문장길이 12로 통일하기

            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post') # matrix 형태로 변환하기
            ## get real image
            # b_real_images = images_train[np.floor(np.asarray(idexs).astype('float') / 12).astype('int')]
            b_real_images=[images_train[id] for id in np.floor(np.asarray(idexs).astype('float') / 12).astype('int')]
            # save_images(b_real_images, [ni, ni], 'samples/step1_gan-cls/train_00.png')
            ## get wrong caption
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)

            b_wrong_caption=[captions_ids_train[id-1] for id in idexs]
            b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post') # matrix 형태로 변환
            ## get wrong image
            idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)

            b_wrong_images=[images_train[id] for id in idexs2]
            ## get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
                # b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)

            b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
            b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train') # chnnel 4인 문제를 해결해야한다.
            ## updates text-to-image mapping
            if epoch < 50:
                errRNN, _ = sess.run([rnn_loss, rnn_optim], feed_dict={
                                                t_real_image : b_real_images, # train
                                                t_wrong_image : b_wrong_images, # train
                                                t_real_caption : b_real_caption,
                                                t_wrong_caption : b_wrong_caption})
            else:
                errRNN = 0

            ## updates D
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                            t_real_image : b_real_images,
                            # t_wrong_image : b_wrong_images,
                            t_wrong_caption : b_wrong_caption,
                            t_real_caption : b_real_caption,
                            t_z : b_z})
            ## updates G
            errG, _ = sess.run([g_loss, g_optim], feed_dict={
                            t_real_caption : b_real_caption,
                            t_z : b_z})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, rnn_loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errRNN))

        # 살펴보기
        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
                                        t_real_caption : sample_sentence, # sample_sentence가 generator의 input으로 들어간다
                                        t_z : sample_seed})

            # print("img_gen : ", img_gen.shape) # img_gen :  (64, 64, 64, 3)

            # img_gen = threading_data(img_gen, prepro_img, mode='rescale')
            print("Save image : ", epoch)
            save_images(img_gen, [ni, ni], 'samples/step1_gan-cls/train_{:02d}.png'.format(epoch)) # error 발생

        ## save model
        if (epoch != 0) and (epoch % 10) == 0:
            tl.files.save_npz(net_cnn.all_params, name=net_cnn_name, sess=sess)
            tl.files.save_npz(net_rnn.all_params, name=net_rnn_name, sess=sess)
            tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            tl.files.save_npz(net_cnn.all_params, name=net_cnn_name+str(epoch), sess=sess)
            tl.files.save_npz(net_rnn.all_params, name=net_rnn_name+str(epoch), sess=sess)
            tl.files.save_npz(net_g.all_params, name=net_g_name+str(epoch), sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name+str(epoch), sess=sess)

        # 일단 사용안함
        # if (epoch != 0) and (epoch % 200) == 0:
        #     sess.run(tf.initialize_variables(adam_vars))
        #     print("Re-initialize Adam")


def main_train_encoder():
    """ for Style Transfer """
    generator_txt2img = model.generator_txt2img_resnet

    ## for training
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False)
    net_fake_image, _ = generator_txt2img(t_z,
                    net_rnn.outputs + tf.random_normal(shape=net_rnn.outputs.get_shape(), mean=0, stddev=0.02), # NOISE ON RNN
                    is_train=True, reuse=False, batch_size=batch_size)
    net_encoder = z_encoder(net_fake_image.outputs, is_train=True, reuse=False)

    ## for evaluation
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    net_z = z_encoder(t_real_image, is_train=False, reuse=True)
    net_g2, _ = generator_txt2img(net_z.outputs, net_rnn.outputs, is_train=False, reuse=True, batch_size=batch_size)

    loss = tf.reduce_mean( tf.square( tf.subtract( net_encoder.outputs, t_z) ))
    e_vars = tl.layers.get_variables_with_name('z_encoder', True, True)

    lr = 0.0002
    lr_decay = 0.5      # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100   # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

    e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=e_vars )


    ###============================ TRAINING ====================================###
    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_encoder_name = os.path.join(save_dir, 'net_encoder.npz')

    if load_and_assign_npz(sess=sess, name=net_g_name, model=net_fake_image) is False:
        raise Exception("Cannot find net_g.npz")
    load_and_assign_npz(sess=sess, name=net_encoder_name, model=net_encoder)

    sample_size = batch_size
    idexs = get_random_int(min=0, max=n_captions_train-1, number=sample_size) # get_random_int seed=100 지우기
    sample_sentence = [captions_ids_train[id] for id in idexs] # captions_ids_train[idexs]  바꾸기
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')
    n_captions_per_image=12
    idexs_=np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')
    sample_image = [images_train[id] for id in idexs_] # n_captions_per_image image당 caption의 수 처리하기
    # print(sample_image.shape, np.min(sample_image), np.max(sample_image), image_size)
    # exit()
    sample_image = threading_data(sample_image, prepro_img, mode='translation')    # central crop first
    save_images(sample_image, [ni, ni], 'samples/step_pretrain_encoder/train__x.png')


    n_epoch = 160 * 100
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)

    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()
            ## get matched text
            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_real_caption = [captions_ids_train[id] for id in idexs]# captions_ids_train[idexs]
            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
            # ## get real image
            # b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
            # ## get wrong caption
            # idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            # b_wrong_caption = captions_ids_train[idexs]
            # b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')
            # ## get wrong image
            # idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
            # b_wrong_images = images_train[idexs2]
            # ## get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
                # b_z = np.random.uniform(low=-1, high=1, size=[batch_size, z_dim]).astype(np.float32)

            ## update E
            errE, _ = sess.run([loss, e_optim], feed_dict={
                            t_real_caption : b_real_caption,
                            t_z : b_z})
                            # t_real_image : b_real_images,})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, e_loss: %8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errE))

        if (epoch + 1) % 10 == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))
            # print(sample_image.shape, t_real_image)
            img_gen = sess.run(net_g2.outputs, feed_dict={
                                        t_real_caption : sample_sentence,
                                        t_real_image : sample_image,})
            img_gen = threading_data(img_gen, imresize, size=[64, 64], interp='bilinear')
            save_images(img_gen, [ni, ni], 'samples/step_pretrain_encoder/train_{:02d}_g(e(x))).png'.format(epoch))

        if (epoch != 0) and (epoch % 5) == 0:
            tl.files.save_npz(net_encoder.all_params, name=net_encoder_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")


def main_transaltion():
    generator_txt2img = model.generator_txt2img_resnet

    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')

    net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False)
    net_z = z_encoder(t_real_image, is_train=False, reuse=False)
    net_g, _ = generator_txt2img(net_z.outputs, net_rnn.outputs, is_train=False, reuse=False)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)

    net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_e_name = os.path.join(save_dir, 'net_encoder.npz')

    load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
    load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
    load_and_assign_npz(sess=sess, name=net_e_name, model=net_z)

    ## random images
    # idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size, seed=100)  # train set
    # images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    # sample_sentence = captions_ids_train[idexs]
    idexs = get_random_int(min=0, max=n_captions_test-1, number=batch_size) # test set
    n_captions_per_image=12
    images = [images_test[id] for id in np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
    for i in [0,8,16,24,32,40,48,56]:
        images[i] = images_test[1834]     # DONE easy 226
        images[i+1] = images_test[620]    # stand on big staff
        images[i+2] = images_test[653]    # 653
        images[i+3] = images_test[77]   # DONE flying 16 20 2166big 2303ok 2306ok 2311good 2313soso 2317soso  2311(want to change)
        images[i+4] = images_test[2167]   # brunch 275 559 2101

        images[i+5] = images_test[235]
        images[i+6] = images_test[1455]  # 717 402
        images[i+7] = images_test[159]  # fat 300  125  159  612
        # # train set
        # images[i] = images_train[620]
        # images[i+1] = images_train[653]
        # images[i+2] = images_train[300]
        # images[i+3] = images_train[350]
        # images[i+4] = images_train[550]
        # images[i+5] = images_train[700]
        # images[i+6] = images_train[717]
        # images[i+7] = images_train[275]
    # sample_sentence = captions_ids_test[idexs]
    images = threading_data(images, prepro_img, mode='translation')
    save_images(images, [ni, ni], 'samples/translation/_reed_method_ori.png')
    n = int(batch_size / ni)

    # all done
    sample_sentence = ["발끝에서 기름이 베어 나와 물 위를 미끄러지듯 걸을 수 있다."] * n + \
                      ["연못이나 호수의 미생물을 먹고 있다."] * n + \
                      ["위험을 감지하면 머리끝에서 물엿같이 달콤한 액체가 나온다."] * n + \
                      ["등에는 커다란 등껍질을 가지고 있다."] * n + \
                      ["소나기가 온 뒤에 웅덩이에 모여들고 있다."] * n + \
                      ["물의 표면을 미끌어 지듯이 가서 머리로부터 달콤한 향기의 꿀을 나오게 한다."] * n + \
                      ["보통은 연못에서 살고 있지만 소나기가 온 뒤에는 마을 안의 물웅덩이에 모습을 드러낸다."] * n + \
                      ["머리 앞쪽에서 물엿과 비슷한 달콤한 냄새의 액체를 낸다. "] * n

    for i, sentence in enumerate(sample_sentence):
        print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] #+ [vocab.end_id]    # add END_ID
        # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
        # print(sample_sentence[i])
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    for i in range(1):
        img_trans = sess.run(net_g.outputs, feed_dict={
                                    t_real_caption : sample_sentence,
                                    t_real_image : images,
                                    })

        save_images(img_trans, [ni, ni], 'samples/translation/_reed_method_tran%d.png' % i)
        print("completed %s" % i)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                       help='train, train_encoder, translation')

    args = parser.parse_args()
    # args.mode='translation'
    if args.mode == "train":
        main_train()

    ## you would not use this part, unless you want to try style transfer on GAN-CLS paper
    elif args.mode == "train_encoder":
        main_train_encoder()

    elif args.mode == "translation":
        main_transaltion()
















