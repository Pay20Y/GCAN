import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import tqdm
import lmdb
import six
from PIL import Image

from data_provider.data_utils import get_vocabulary

from utils.transcription_utils import idx2label, calc_metrics, calc_metrics_lexicon
from sar_model import SARModel
from utils.visualization import mask_visualize, line_visualize, heatmap_visualize

from config import get_args

def get_images(images_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(images_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def get_data(args):
    if args.test_data_gt != '' and os.path.exists(args.test_data_gt):
        if args.test_data_gt.split('.')[1] == 'json':
            # data_loader = ArTLoader(args.test_data_dir)
            data_loader = SynTextLoader(args.test_data_dir)
        elif args.test_data_gt.split('.')[1] == 'txt':
            data_loader = Syn90KLoader(args.test_data_dir)
        else:
            raise Exception("Unsupported file type")
        images_path, labels = data_loader.parse_gt(args.test_data_gt)
        return images_path, labels
    else:
        images_path = get_images(args.test_data_dir)
        labels = ['' for i in range(len(images_path))]
        return images_path, labels

def get_data_lexicon(args):
    images_path, labels, lexicons = [], [], []
    with open(args.test_data_gt, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split()
            images_path.append(os.path.join(args.test_data_dir, line[0]))
            labels.append(line[1])
            lexicons.append([w for w in line[3].split(',')])

    return images_path, labels, lexicons

def resize_pad_img(image, height, width):
    H, W, C = image.shape
    # Rotate the vertical images
    if H > 4 * W:
        image = np.rot90(image)
        H, W = W, H

    new_width = int((1.0 * height / H) * W)
    new_width = new_width if new_width < width else width
    new_height = height
    img_resize = np.zeros((height, width, C), dtype=np.uint8)
    image = cv2.resize(image, (new_width, new_height))
    img_resize[:, :new_width, :] = image

    return img_resize, new_width

def data_preprocess(image, word, char2id, args):
    """
    H, W, C = image.shape
    # Rotate the vertical images
    if H > 4 * W:
        image = np.rot90(image)
        H, W = W, H

    new_width = int((1.0 * args.height / H) * W)
    new_width = new_width if new_width < args.width else args.width
    new_height = args.height
    img_resize = np.zeros((args.height, args.width, C), dtype=np.uint8)
    image = cv2.resize(image, (new_width, new_height))
    img_resize[:, :new_width, :] = image
    """
    img_resize, new_width = resize_pad_img(image, args.height, args.width)
    label = np.full((args.max_len), char2id['PAD'], dtype=np.int)
    label_list = []
    for char in word:
        if char in char2id:
            label_list.append(char2id[char])
        else:
            label_list.append(char2id['UNK'])
    # label_list = label_list + [char2id['EOS']]
    # assert len(label_list) <= max_len
    if len(label_list) > (args.max_len - 1):
        label_list = label_list[:(args.max_len - 1)]
    label_list = label_list + [char2id['EOS']]
    label[:len(label_list)] = np.array(label_list)

    return img_resize, label, new_width

def main_test(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    input_images = tf.placeholder(dtype=tf.float32, shape=[1, args.height, args.width, 3], name="input_images")
    input_images_width = tf.placeholder(dtype=tf.float32, shape=[1], name="input_images_width")
    input_labels = tf.placeholder(dtype=tf.int32, shape=[1, args.max_len], name="input_labels")
    sar_model = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=False)

    # encoder_state, feature_map, mask_map = sar_model.inference(input_images, input_images_width, 1, reuse=True)
    # model_infer, attention_weights, pred = sar_model.decode(encoder_state, feature_map, input_labels, mask_map, reuse=True, decode_type=args.decode_type)
    model_infer, attention_weights, pred, _ = sar_model(input_images, input_labels, input_images_width, batch_size=1, reuse=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    # saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
        model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        print("Checkpoints step: {}".format(global_step.eval(session=sess)))
        images_path, labels = get_data(args)
        predicts = []
        # for img_path, label in zip(images_path, labels):
        for i in tqdm.tqdm(range(len(images_path))):
            img_path = images_path[i]
            label = labels[i]
            try:
                img = cv2.imread(img_path)
            except Exception as e:
                print("{} error: {}".format(img_path, e))
                continue

            img, la, width = data_preprocess(img, label, char2id, args)

            pred_value, attention_weights_value = sess.run([pred, attention_weights], feed_dict={input_images: [img],
                                                                                                 input_labels: [la],
                                                                                                 input_images_width: [width]})
            pred_value_str = idx2label(pred_value, id2char, char2id)[0]
            # print("predict: {} label: {}".format(pred_value_str, label))
            predicts.append(pred_value_str)
            if args.vis_dir != None and args.vis_dir != "":
                os.makedirs(args.vis_dir, exist_ok=True)
                os.makedirs(os.path.join(args.vis_dir, "errors"), exist_ok=True)
                _ = line_visualize(img, attention_weights_value, pred_value_str, args.vis_dir, img_path)
                if pred_value_str.lower() != label.lower():
                    _ = line_visualize(img, attention_weights_value, pred_value_str, os.path.join(args.vis_dir, "errors"), img_path)
    # acc_rate = calc_metrics(predicts, labels)
    acc_rate = calc_metrics_length(predicts, labels)
    if isinstance(acc_rate, dict):
        for k, v in acc_rate.items():
            print("length: {} accuracy: {}".format(k, v))
    else:
        print("Done, Accuracy: {}".format(acc_rate))


def main_test_with_lexicon(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    input_images = tf.placeholder(dtype=tf.float32, shape=[1, args.height, args.width, 3], name="input_images")
    input_images_width = tf.placeholder(dtype=tf.float32, shape=[1], name="input_images_width")
    input_labels = tf.placeholder(dtype=tf.int32, shape=[1, args.max_len], name="input_labels")
    sar_model = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=False)

    # encoder_state, feature_map, mask_map = sar_model.inference(input_images, input_images_width, 1, reuse=True)
    # model_infer, attention_weights, pred = sar_model.decode(encoder_state, feature_map, input_labels, mask_map, reuse=True, decode_type=args.decode_type)
    model_infer, attention_weights, pred, _ = sar_model(input_images, input_labels, input_images_width, batch_size=1, reuse=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    # saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
        model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        print("Checkpoints step: {}".format(global_step.eval(session=sess)))
        images_path, labels, lexicons = get_data_lexicon(args)
        predicts = []
        # for img_path, label in zip(images_path, labels):
        for i in tqdm.tqdm(range(len(images_path))):
            img_path = images_path[i]
            label = labels[i]
            try:
                img = cv2.imread(img_path)
            except Exception as e:
                print("{} error: {}".format(img_path, e))
                continue

            img, la, width = data_preprocess(img, label, char2id, args)

            pred_value, attention_weights_value = sess.run([pred, attention_weights], feed_dict={input_images: [img],
                                                                                                 input_labels: [la],
                                                                                                 input_images_width: [width]})
            pred_value_str = idx2label(pred_value, id2char, char2id)[0]
            # print("predict: {} label: {}".format(pred_value_str, label))
            predicts.append(pred_value_str)
            if args.vis_dir != None and args.vis_dir != "":
                os.makedirs(args.vis_dir, exist_ok=True)
                os.makedirs(os.path.join(args.vis_dir, "errors"), exist_ok=True)
                _ = line_visualize(img, attention_weights_value, pred_value_str, args.vis_dir, img_path)
                if pred_value_str.lower() != label.lower():
                    _ = line_visualize(img, attention_weights_value, pred_value_str, os.path.join(args.vis_dir, "errors"), img_path)
    acc_rate = calc_metrics_lexicon(predicts, labels, lexicons)
    print("Done, Accuracy: {}".format(acc_rate))

def main_test_lmdb(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    input_images = tf.placeholder(dtype=tf.float32, shape=[1, args.height, args.width, 3], name="input_images")
    input_images_width = tf.placeholder(dtype=tf.float32, shape=[1], name="input_images_width")
    input_labels = tf.placeholder(dtype=tf.int32, shape=[1, args.max_len], name="input_labels")
    sar_model = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=False)

    # encoder_state, feature_map, mask_map = sar_model.inference(input_images, input_images_width, 1, reuse=True)
    # model_infer, attention_weights, pred = sar_model.decode(encoder_state, feature_map, input_labels, mask_map, reuse=True, decode_type=args.decode_type)
    model_infer, attention_weights, pred, _ = sar_model(input_images, input_labels, input_images_width, batch_size=1, reuse=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    # saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
        model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        print("Checkpoints step: {}".format(global_step.eval(session=sess)))

        env = lmdb.open(args.test_data_dir, readonly=True)
        txn = env.begin()
        num_samples = int(txn.get(b"num-samples").decode())
        predicts = []
        labels = []
        # for img_path, label in zip(images_path, labels):
        for i in tqdm.tqdm(range(1, num_samples+1)):
            image_key = b'image-%09d' % i
            label_key = b'label-%09d' % i

            imgbuf = txn.get(image_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            img_pil = Image.open(buf).convert('RGB')
            img = np.array(img_pil)
            label = txn.get(label_key).decode()
            labels.append(label)
            img, la, width = data_preprocess(img, label, char2id, args)

            pred_value, attention_weights_value = sess.run([pred, attention_weights], feed_dict={input_images: [img],
                                                                                                 input_labels: [la],
                                                                                                 input_images_width: [
                                                                                                     width]})
            pred_value_str = idx2label(pred_value, id2char, char2id)[0]
            # print("predict: {} label: {}".format(pred_value_str, label))
            predicts.append(pred_value_str)
            img_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if args.vis_dir != None and args.vis_dir != "":
                os.makedirs(args.vis_dir, exist_ok=True)
                os.makedirs(os.path.join(args.vis_dir, "errors"), exist_ok=True)
                # _ = line_visualize(img, attention_weights_value, pred_value_str, args.vis_dir, "{}.jpg".format(i))
                _ = heatmap_visualize(img_vis, attention_weights_value, pred_value_str, args.vis_dir, "{}.jpg".format(i))
                # _ = mask_visualize(img, attention_weights_value, pred_value_str, args.vis_dir, img_path)
                if pred_value_str.lower() != label.lower():
                    _ = heatmap_visualize(img_vis, attention_weights_value, pred_value_str, os.path.join(args.vis_dir, "errors"), "{}.jpg".format(i))
        acc_rate = calc_metrics(predicts, labels)
        print("Done, Accuracy: {}".format(acc_rate))

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_test_lmdb(args)
    # main_test_with_lexicon(args)
    # main_test_two_stages(args)