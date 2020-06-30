import lmdb
import six
import random
import time
import numpy as np
from PIL import Image, ImageFile
import cv2
import math

import imgaug

from data_provider.generator_enqueuer import GeneratorEnqueuer
from data_provider.data_utils import get_vocabulary, rotate_img, get_gauss_distrib, find_min_rectangle, aff_gaussian, sum_norm, get_distrib_params, construct_gauss_distirb, estim_gauss_params, roi_sum, roi_max, adding_guass
from utils.visualization import line_visualize

def generator(lmdb_dir, input_height, input_width, batch_size, max_len, voc_type, keep_ratio=True, with_aug=True):
    env = lmdb.open(lmdb_dir, max_readers=32, readonly=True)
    txn = env.begin()

    if txn.get(b"num-samples") is None: # SynthText800K is still generating
        num_samples = 3000000
    else:
        num_samples = int(txn.get(b"num-samples").decode())
    print("There are {} images in {}".format(num_samples, lmdb_dir))
    index = np.arange(0, num_samples) # TODO check index is reliable

    voc, char2id, id2char = get_vocabulary(voc_type)
    is_lowercase = (voc_type == 'LOWERCASE')

    batch_images = []
    batch_images_width = []
    batch_labels = []
    batch_gausses = []
    batch_params = []
    batch_gauss_tags = []
    batch_gauss_masks = []
    batch_lengths = []
    batch_masks = []
    batch_labels_str = []
    batch_char_size = []
    batch_char_bbs = []

    while True:
        np.random.shuffle(index)
        for i in index:
            i += 1
            try:
                image_key = b'image-%09d' % i
                label_key = b'label-%09d' % i
                char_key = b'char-%09d' % i

                imgbuf = txn.get(image_key)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)

                img_pil = Image.open(buf).convert('RGB')
                img = np.array(img_pil)
                word = txn.get(label_key).decode()
                wo_char = (txn.get(char_key) is None)
                if wo_char:
                    charBBs = np.zeros(dtype=np.float32, shape=[len(word), 4, 2])
                else:
                    charBBs = np.array([math.ceil(float(c)) for c in txn.get(char_key).decode().split()]).reshape([-1, 4, 2]).astype(np.float32)

                num_char_bb = charBBs.shape[0]
                if is_lowercase:
                    word = word.lower()
                H, W, C = img.shape

                charBBs[:, :, 0] = np.clip(charBBs[:, :, 0], 0, W)
                charBBs[:, :, 1] = np.clip(charBBs[:, :, 1], 0, H)


                # Rotate the vertical images
                if H > 1.1 * W:
                    img = np.rot90(img)
                    H, W = W, H
                    charBBs = charBBs[:, :, ::-1]
                    charBBs[:, :, 1] = H - charBBs[:, :, 1]

                # Resize the images
                img_resize = np.zeros((input_height, input_width, C), dtype=np.uint8)

                # Data augmentation
                if with_aug:
                    blur_aug = imgaug.augmenters.blur.GaussianBlur(sigma=(0,1.0))
                    contrast_aug = imgaug.augmenters.contrast.LinearContrast((0.75,1.0))
                    affine_aug = imgaug.augmenters.geometric.PiecewiseAffine(scale=(0.01, 0.02), mode='constant', cval=0)

                    # Gaussian Blur
                    ratn = random.randint(0, 1)
                    # img = cv2.GaussianBlur(img, (3, 3), 2)
                    if ratn == 0:
                        img = blur_aug.augment_image(img)
                    # Gaussian noise
                    # img = adding_guass(img)

                    # Contrast
                    ratn = random.randint(0, 1)
                    if ratn == 0:
                        img = contrast_aug.augment_image(img)

                    # Affine
                    ratn = random.randint(0, 1)
                    if ratn == 0:
                        img = affine_aug.augment_image(img)

                    # Rotation
                    ratn = random.randint(0, 1)
                    if ratn == 0:
                        # rand_reg = random.random() * 30 - 15
                        rand_reg = random.randint(-15, 15)
                        img, charBBs, (W, H) = rotate_img(img, rand_reg, BBs=charBBs)

                if keep_ratio:
                    new_width = int((1.0 * H / input_height) * input_width)
                    new_width = new_width if new_width < input_width else input_width
                    new_width = new_width if new_width >= input_height else input_height
                    new_height = input_height
                    img = cv2.resize(img, (new_width, new_height))
                    img_resize[:new_height, :new_width, :] = img.copy()
                else:
                    new_width = input_width
                    new_height = input_height
                    img_resize = cv2.resize(img, (input_width, input_height))

                ratio_w = float(new_width) / float(W)
                ratio_h = float(new_height) / float(H)

                charBBs[:, :, 0] = charBBs[:, :, 0] * ratio_w
                charBBs[:, :, 1] = charBBs[:, :, 1] * ratio_h

                # visualization for debugging rotate augmentation
                # img_debug = img_resize.copy()
                # for bb in charBBs:
                #     img_debug = cv2.polylines(img_debug, [bb.astype(np.int32).reshape((-1, 1, 2))], True,
                #                               color=(255, 255, 0), thickness=1)
                # cv2.imwrite("./char_bb_vis/{}.jpg".format(i), img_debug)

                # feature_map_w = 40.
                # feature_map_h = 6.
                # feature_map_w = input_width
                # feature_map_h = input_height
                # ratio_w_f = feature_map_w / input_width
                # ratio_h_f = feature_map_h / input_height
                # charBBs[:, :, 0] = charBBs[:, :, 0] * ratio_w_f
                # charBBs[:, :, 1] = charBBs[:, :, 1] * ratio_h_f

                label = np.full((max_len), char2id['PAD'], dtype=np.int)
                label_mask = np.full((max_len), 0, dtype=np.int)
                label_list = []
                for char in word:
                    if char in char2id:
                        label_list.append(char2id[char])
                    else:
                        label_list.append(char2id['UNK'])

                if len(label_list) > (max_len - 1):
                    label_list = label_list[:(max_len - 1)]
                    num_char_bb = max_len - 1
                label_list = label_list + [char2id['EOS']]
                label[:len(label_list)] = np.array(label_list)

                if label.shape[0] <= 0:
                    continue

                label_len = len(label_list)
                label_mask[:label_len] = 1
                if (label_len-1) != num_char_bb:
                    print("Unmatched between char bb and label length in index {}".format(i))
                    print("Information: label: {} label_length: {} num_char: {}".format(word, label_len-1, num_char_bb))
                    continue

                # Get gaussian distribution labels
                # gauss_labels = np.zeros(dtype=np.float32, shape=[max_len, int(feature_map_h), int(feature_map_w)]) # T * H * W
                gauss_labels = np.zeros(dtype=np.float32, shape=[max_len, input_height, input_width]) # T * H * W
                # gauss_mask = np.zeros(dtype=np.float32, shape=[max_len, int(feature_map_h), int(feature_map_w)]) # T * H * W
                gauss_mask = np.zeros(dtype=np.float32, shape=[max_len, input_height, input_width]) # T * H * W
                # distrib_params = []
                distrib_params = np.zeros(dtype=np.float32, shape=[max_len, 4])
                distrib_params[:, 2:] = 1.
                char_size = np.ones(dtype=np.float32, shape=[max_len, 2])
                gauass_tags = [0] * max_len
                charBBs = charBBs[:num_char_bb]
                if wo_char == False:
                    for i, BB in enumerate(charBBs): # 4 * 2
                        # try:
                        # Here we use min bounding rectangles
                        min_rec, delta_x, delta_y = find_min_rectangle(BB) # 4 * 2
                        if delta_x < 2 or delta_y < 2:
                            param = get_distrib_params(BB)
                            distrib_params[i] = param
                            continue
                        gauss_distrib = get_gauss_distrib((delta_y, delta_x)) # delta_y * delta_x
                        # param = get_distrib_params(BB)
                        param = estim_gauss_params(gauss_distrib, delta_x, delta_y)
                        # param[0] = param[0] / feature_map_w
                        # param[1] = param[1] / feature_map_h
                        # param[2] = param[2] / (0.25 * feature_map_w * feature_map_w)
                        # param[3] = param[3] / (0.25 * feature_map_h * feature_map_h)
                        # gauss_distrib = construct_gauss_distirb(param, delta_x, delta_y)
                        char_size[i][0] = delta_x
                        char_size[i][1] = delta_y
                        distrib_params[i] = param
                        # res_gauss = aff_gaussian(gauss_distrib, min_rec, BB, delta_x, delta_y) # delta_y * delta_x
                        res_gauss = gauss_distrib
                        gauass_tags[i] = 1.
                        if np.max(res_gauss) > 0.:
                            start_x, start_y = int(min_rec[0][0]), int(min_rec[0][1])
                            end_x, end_y = start_x + delta_x, start_y + delta_y
                            end_x = end_x if end_x <= input_width else input_width
                            end_y = end_y if end_y <= input_height else input_height
                            gauss_labels[i, start_y:end_y, start_x:end_x] = res_gauss

                            ex_start_x = math.floor(start_x - 0.3 * delta_x)
                            ex_end_x = math.ceil(end_x + 0.3 * delta_x)
                            ex_start_y = math.floor(start_y - 0.3 * delta_y)
                            ex_end_y = math.ceil(end_y + 0.3 * delta_y)

                            ex_start_x = ex_start_x if ex_start_x >=0 else 0
                            ex_start_y = ex_start_y if ex_start_y >=0 else 0
                            ex_end_x = ex_end_x if ex_end_x <= input_width else input_width
                            ex_end_y = ex_end_y if ex_end_y <= input_height else input_height

                            gauss_mask[i, int(ex_start_y):int(ex_end_y), int(ex_start_x):int(ex_end_x)] = 1.
                            # gauss_mask[i, int(start_y):int(end_y), int(start_x):int(end_x)] = 1.

                        # except Exception as e:
                        #     print(e)
                    gauss_labels = sum_norm(gauss_labels.reshape([gauss_labels.shape[0], -1])).reshape([-1, input_height, input_width])

                # Reduce to feature map size
                gauss_labels = roi_sum(gauss_labels, target_h=6, target_w=40)
                gauss_mask = roi_max(gauss_mask, target_h=6, target_w=40)
                    # distrib_params = np.array(distrib_params)

                batch_images.append(img_resize)
                batch_images_width.append(new_width)
                batch_labels.append(label)
                batch_gausses.append(gauss_labels)
                batch_params.append(distrib_params)
                batch_gauss_tags.append(gauass_tags)
                batch_gauss_masks.append(gauss_mask)
                batch_masks.append(label_mask)
                batch_lengths.append(label_len)
                batch_labels_str.append(word)
                batch_char_size.append(char_size)
                batch_char_bbs.append(charBBs)

                assert len(batch_images) == len(batch_labels) == len(batch_lengths) == len(batch_gausses) == len(batch_char_size)

                if len(batch_images) == batch_size:
                    yield np.array(batch_images), \
                          np.array(batch_labels), \
                          np.array(batch_gausses), \
                          np.array(batch_masks), \
                          np.array(batch_lengths), \
                          batch_labels_str, \
                          np.array(batch_images_width), \
                          np.array(batch_gauss_tags), \
                          np.array(batch_params), \
                          np.array(batch_char_size), \
                          np.array(batch_gauss_masks)
                    batch_images = []
                    batch_images_width = []
                    batch_labels = []
                    batch_gausses = []
                    batch_params = []
                    batch_gauss_tags = []
                    batch_gauss_masks = []
                    batch_masks = []
                    batch_lengths = []
                    batch_labels_str = []
                    batch_char_size = []

            except Exception as e:
                print(e)
                print("Error in %d" % i)
                continue

def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=4, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

if __name__ == "__main__":
    import os
    data_gen = get_batch(num_workers=1, lmdb_dir="/data3/qz/Data/SynthTextCropChar_800K_LMDB", input_height=48, input_width=160, batch_size=4, max_len=30, voc_type='ALLCASES_SYMBOLS', keep_ratio=True, with_aug=True)

    for i in range(100):
        data = next(data_gen)
        # print("batch images shape: ", data[0].shape)
        # print("batch labels: ", data[1])
        # print("batch gauss: ", data[2].shape)
        # print("batch masks: ", data[3])
        # print("batch lengths: ", data[4])
        # print("batch labels string: ", data[5])

        demo_img = data[0][0] # H * W * 3
        demo_label = data[5][0] # str
        demo_gauss = data[2][0] # T * H * W
        demo_gauss_mask = data[10][0] # T * H * W
        demo_gauss_tag = data[7][0] # T
        demo_gauss_sum = np.sum(demo_gauss.reshape([30, -1]), axis=1)
        print("gauss_sum: ", demo_gauss_sum)
        print("demo_gauss_mask: ", demo_gauss_mask)
        H, W, _ = demo_img.shape
        demo_gauss = np.expand_dims(demo_gauss, axis=0)
        for j in range(demo_gauss_mask.shape[0]):
            mask_map = demo_gauss_mask[j]
            mask_map = cv2.resize(mask_map, (W, H))
            _mask_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
            _mask_map[:, :, -1] = (mask_map * 255).astype(np.uint8)
            show_attention = cv2.addWeighted(demo_img, 0.5, _mask_map, 2, 0)

            # cv2.imwrite(os.path.join("gauss_vis", "{}_{}.jpg".format(i,j)),show_attention)
        line_visualize(demo_img, demo_gauss, demo_label, "gauss_vis", "{}.jpg".format(i))