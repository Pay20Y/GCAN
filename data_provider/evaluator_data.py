import lmdb
import six
import random
import time
import numpy as np
from PIL import Image, ImageFile
import cv2

from data_provider.generator_enqueuer import GeneratorEnqueuer
from data_provider.data_utils import get_vocabulary, rotate_img

class Evaluator(object):
    def __init__(self, lmdb_data_dir, batch_size, height, width, max_len, keep_ratio, voc_type='LOWERCASE'):
        self.env = lmdb.open(lmdb_data_dir)
        self.txn = self.env.begin()
        self.num_samples = int(self.txn.get(b"num-samples").decode())
        print("There are {} images in {}".format(self.num_samples, lmdb_data_dir))
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.max_len = max_len
        self.keep_ratio = keep_ratio
        self.voc_type = voc_type
        self.index = 1

    def reset(self):
        self.index = 1

    def get_batch(self):
        voc, char2id, id2char = get_vocabulary(self.voc_type)
        is_lowercase = (self.voc_type == 'LOWERCASE')

        batch_images = []
        batch_images_width = []
        batch_labels = []
        batch_lengths = []
        batch_masks = []
        batch_labels_str = []
        if (self.index + self.batch_size - 1) > self.num_samples:
            self.reset()
            return None

        # for i in range(self.index, self.index + self.batch_size - 1):

        while len(batch_images) < self.batch_size:
            try:
                image_key = b'image-%09d' % self.index
                label_key = b'label-%09d' % self.index
                self.index += 1

                imgbuf = self.txn.get(image_key)
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)

                img_pil = Image.open(buf).convert('RGB')
                img = np.array(img_pil)
                word = self.txn.get(label_key).decode()
                if is_lowercase:
                    word = word.lower()
                H, W, C = img.shape

                # Rotate the vertical images
                if H > 1.1 * W:
                    img = np.rot90(img)
                    H, W = W, H

                # Resize the images
                img_resize = np.zeros((self.height, self.width, C), dtype=np.uint8)

                if self.keep_ratio:
                    new_width = int((1.0 * H / self.height) * self.width)
                    new_width = new_width if new_width < self.width else self.width
                    new_width = new_width if new_width >= self.height else self.height
                    new_height = self.height
                    img = cv2.resize(img, (new_width, new_height))
                    img_resize[:new_height, :new_width, :] = img.copy()
                else:
                    new_width = self.width
                    img_resize = cv2.resize(img, (self.width, self.height))

                label = np.full((self.max_len), char2id['PAD'], dtype=np.int)
                label_mask = np.full((self.max_len), 0, dtype=np.int)
                label_list = []
                for char in word:
                    if char in char2id:
                        label_list.append(char2id[char])
                    else:
                        label_list.append(char2id['UNK'])

                if len(label_list) > (self.max_len - 1):
                    label_list = label_list[:(self.max_len - 1)]
                label_list = label_list + [char2id['EOS']]
                label[:len(label_list)] = np.array(label_list)

                if label.shape[0] <= 0:
                    continue

                label_len = len(label_list)
                label_mask[:label_len] = 1

                batch_images.append(img_resize)
                batch_images_width.append(new_width)
                batch_labels.append(label)
                batch_masks.append(label_mask)
                batch_lengths.append(label_len)
                batch_labels_str.append(word)

                assert len(batch_images) == len(batch_labels) == len(batch_lengths)

            except Exception as e:
                print(e)
                print("Error in %d" % self.index)
                continue

        return np.array(batch_images), np.array(batch_labels), np.array(batch_masks), np.array(batch_lengths), batch_labels_str, np.array(batch_images_width)