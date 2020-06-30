import sys
import os
import time
import tensorflow as tf
import numpy as np

from sar_model import SARModel
from data_provider import lmdb_data_generator, lmdb_char_data_generator
from data_provider import evaluator_data
from data_provider.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
from config import get_args


def get_data(image_dir, voc_type, max_len, height, width, batch_size, workers, keep_ratio, with_aug):
    data_list = []
    if isinstance(image_dir, list) and len(image_dir) > 1:
        # assert len(image_dir) == len(gt_path), "datasets and gt are not corresponding"
        assert batch_size % len(image_dir) == 0, "batch size should divide dataset num"
        per_batch_size = batch_size // len(image_dir)
        for i in image_dir:
            data_list.append(lmdb_char_data_generator.get_batch(workers, lmdb_dir=i, input_height=height, input_width=width, batch_size=per_batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug))
    else:
        if isinstance(image_dir, list):
            data = lmdb_char_data_generator.get_batch(workers, lmdb_dir=image_dir[0], input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)

        else:
            data = lmdb_char_data_generator.get_batch(workers, lmdb_dir=image_dir, input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)
        data_list.append(data)

    return data_list

def get_batch_data(data_list, batch_size):
    batch_images = []
    batch_labels = []
    batch_gauss_labels = []
    batch_gauss_tags = []
    batch_gauss_params = []
    batch_labels_mask = []
    batch_labels_str = []
    batch_widths = []
    batch_char_sizes = []
    batch_gauss_mask = []
    for data in data_list:
        _data = next(data)
        batch_images.append(_data[0])
        batch_labels.append(_data[1])
        batch_gauss_labels.append(_data[2])
        batch_labels_mask.append(_data[3])
        batch_labels_str.extend(_data[5])
        batch_widths.append(_data[6])
        batch_gauss_tags.append(_data[7])
        batch_gauss_params.append(_data[8])
        batch_char_sizes.append(_data[9])
        batch_gauss_mask.append(_data[10])

    batch_images = np.concatenate(batch_images, axis=0)
    batch_labels = np.concatenate(batch_labels, axis=0)
    batch_gauss_labels = np.concatenate(batch_gauss_labels,axis=0)
    batch_labels_mask = np.concatenate(batch_labels_mask, axis=0)
    batch_widths = np.concatenate(batch_widths, axis=0)
    batch_gauss_tags = np.concatenate(batch_gauss_tags, axis=0)
    batch_gauss_params = np.concatenate(batch_gauss_params, axis=0)
    batch_char_sizes = np.concatenate(batch_char_sizes, axis=0)
    batch_gauss_mask = np.concatenate(batch_gauss_mask, axis=0)

    assert len(batch_images) == batch_size, "concat data is not equal to batch size"

    return batch_images, batch_labels, batch_gauss_labels, batch_labels_mask, batch_labels_str, batch_widths, batch_gauss_tags, batch_gauss_params, batch_char_sizes, batch_gauss_mask

def main_train(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)
    tf.set_random_seed(1)
    # Build graph
    input_train_images = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.height, args.width, 3], name="input_train_images")
    input_train_images_width = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size], name="input_train_width")
    input_train_labels = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels")
    input_train_gauss_labels = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.max_len, 6, 40], name="input_train_gauss_labels") # better way wanted!!!
    input_train_gauss_tags = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.max_len], name="input_train_gauss_tags")
    input_train_gauss_params = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.max_len, 4], name="input_train_gauss_params")
    input_train_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels_mask")
    input_train_char_sizes = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.max_len, 2], name="input_train_char_sizes")
    input_train_gauss_mask = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.max_len, 6, 40], name="input_train_gauss_mask")


    input_val_images = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size, args.height, args.width, 3],name="input_val_images")
    input_val_images_width = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size], name="input_val_width")
    input_val_labels = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels")
    # input_val_gauss_labels = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size, args.max_len, args.height, args.width], name="input_val_gauss_labels")
    input_val_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels_mask")

    sar_model = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=True,
                         att_loss_type=args.att_loss_type,
                         att_loss_weight=args.att_loss_weight)
    sar_model_val = SARModel(num_classes=len(voc),
                         encoder_dim=args.encoder_sdim,
                         encoder_layer=args.encoder_layers,
                         decoder_dim=args.decoder_sdim,
                         decoder_layer=args.decoder_layers,
                         decoder_embed_dim=args.decoder_edim,
                         seq_len=args.max_len,
                         is_training=False)
    train_model_infer, train_attention_weights, train_pred, train_attention_params = sar_model(input_train_images, input_train_labels,
                                                                                               input_train_images_width,
                                                                                               batch_size=args.train_batch_size, reuse=False)
    if args.att_loss_type == "kldiv":
        train_loss, train_recog_loss, train_att_loss = sar_model.loss(train_model_infer, train_attention_weights, input_train_labels, input_train_gauss_labels, input_train_labels_mask, input_train_gauss_tags)
    elif args.att_loss_type == "l1" or args.att_loss_type == "l2":
        # train_loss, train_recog_loss, train_att_loss = sar_model.loss(train_model_infer, train_attention_params, input_train_labels, input_train_gauss_params, input_train_labels_mask, input_train_gauss_tags, input_train_char_sizes)
        train_loss, train_recog_loss, train_att_loss = sar_model.loss(train_model_infer, train_attention_weights, input_train_labels, input_train_gauss_labels, input_train_labels_mask, input_train_gauss_tags)
    elif args.att_loss_type == 'ce':
        train_loss, train_recog_loss, train_att_loss = sar_model.loss(train_model_infer, train_attention_weights, input_train_labels, input_train_gauss_labels, input_train_labels_mask, input_train_gauss_tags, input_train_gauss_mask)
    elif args.att_loss_type == 'gausskldiv':
        train_loss, train_recog_loss, train_att_loss = sar_model.loss(train_model_infer, train_attention_params, input_train_labels, input_train_gauss_params, input_train_labels_mask, input_train_gauss_tags)

    else:
        print("Unimplemented loss type {}".format(args.att_loss_dtype))
        exit(-1)
    val_model_infer, val_attention_weights, val_pred, _ = sar_model_val(input_val_images,
                                                                        input_val_labels,
                                                                        input_val_images_width,
                                                                        batch_size=args.val_batch_size, reuse=True)

    train_data_list = get_data(args.train_data_dir,
                         args.voc_type,
                         args.max_len,
                         args.height,
                         args.width,
                         args.train_batch_size,
                         args.workers,
                         args.keep_ratio,
                         with_aug=args.aug)

    val_data_gen = evaluator_data.Evaluator(lmdb_data_dir=args.test_data_dir,
                                            batch_size=args.val_batch_size,
                                            height=args.height,
                                            width=args.width,
                                            max_len=args.max_len,
                                            keep_ratio=args.keep_ratio,
                                            voc_type=args.voc_type)
    val_data_gen.reset()

    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)

    learning_rate = tf.train.piecewise_constant(global_step, args.decay_bound, args.lr_stage)
    batch_norm_updates_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    # Save summary
    os.makedirs(args.checkpoints, exist_ok=True)
    tf.summary.scalar(name='train_loss', tensor=train_loss)
    tf.summary.scalar(name='train_recog_loss', tensor=train_recog_loss)
    tf.summary.scalar(name='train_att_loss', tensor=train_att_loss)
    # tf.summary.scalar(name='val_att_loss', tensor=val_att_loss)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)

    merge_summary_op = tf.summary.merge_all()

    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'sar_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(args.checkpoints, model_name)
    best_model_save_path = os.path.join(args.checkpoints, 'best_model', model_name)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, batch_norm_updates_op]):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(train_loss)
        if args.grad_clip > 0:
            print("With Gradients clipped!")
            for idx, (grad, var) in enumerate(grads):
                grads[idx] = (tf.clip_by_norm(grad, args.grad_clip), var)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    summary_writer = tf.summary.FileWriter(args.checkpoints)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log_file = open(os.path.join(args.checkpoints, args.checkpoints + ".log"), "w")
    with tf.Session(config=config) as sess:
        summary_writer.add_graph(sess.graph)
        start_iter = 0
        if args.resume == True and args.pretrained != '':
            print('Restore model from {:s}'.format(args.pretrained))
            ckpt_state = tf.train.get_checkpoint_state(args.pretrained)
            model_path = os.path.join(args.pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess=sess, save_path=model_path)
            start_iter = sess.run(tf.train.get_global_step())
        elif args.resume == False and args.pretrained != '':
            print('Restore pretrained model from {:s}'.format(args.pretrained))
            ckpt_state = tf.train.get_checkpoint_state(args.pretrained)
            model_path = os.path.join(args.pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess=sess, save_path=model_path)
            sess.run(tf.assign(global_step, 0))
        else:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)

        # Evaluate the model first
        val_pred_value_all = []
        val_labels = []
        for eval_iter in range(val_data_gen.num_samples // args.val_batch_size):

            val_data = val_data_gen.get_batch()
            if val_data is None:
                break
            print("Evaluation: [{} / {}]".format(eval_iter, (val_data_gen.num_samples // args.val_batch_size)))
            val_pred_value = sess.run(val_pred, feed_dict={input_val_images: val_data[0],
                                                           input_val_labels: val_data[1],
                                                           input_val_images_width: val_data[5],
                                                           input_val_labels_mask: val_data[2]})
            val_pred_value_all.extend(val_pred_value)
            val_labels.extend(val_data[4])

        val_data_gen.reset()
        val_metrics_result = calc_metrics(idx2label(np.array(val_pred_value_all)), val_labels, metrics_type="accuracy")
        print("Evaluation Before training: Test accuracy {:3f}".format(val_metrics_result))
        val_best_acc = val_metrics_result


        while start_iter < args.iters:
            start_iter += 1
            train_data = get_batch_data(train_data_list, args.train_batch_size)
            _, train_loss_value, train_recog_loss_value, train_att_loss_value, train_pred_value = sess.run([train_op, train_loss, train_recog_loss, train_att_loss, train_pred], feed_dict={input_train_images: train_data[0],
                                                                                                                                                                                            input_train_labels: train_data[1],
                                                                                                                                                                                            input_train_gauss_labels: train_data[2],
                                                                                                                                                                                            input_train_gauss_params: train_data[7],
                                                                                                                                                                                            input_train_labels_mask: train_data[3],
                                                                                                                                                                                            input_train_images_width: train_data[5],
                                                                                                                                                                                            input_train_gauss_tags: train_data[6],
                                                                                                                                                                                            input_train_char_sizes: train_data[8],
                                                                                                                                                                                            input_train_gauss_mask: train_data[9]})

            if start_iter % args.log_iter == 0:
                print("Iter {} train loss= {:3f} (recog loss= {:3f} att loss= {:3f})".format(start_iter, train_loss_value, train_recog_loss_value, train_att_loss_value))
                log_file.write("Iter {} train loss= {:3f} (recog loss= {:3f} att loss= {:3f})".format(start_iter, train_loss_value, train_recog_loss_value, train_att_loss_value))
            if start_iter % args.summary_iter == 0:
                merge_summary_value = sess.run(merge_summary_op, feed_dict={input_train_images: train_data[0],
                                                                            input_train_labels: train_data[1],
                                                                            input_train_gauss_labels: train_data[2],
                                                                            input_train_gauss_params: train_data[7],
                                                                            input_train_labels_mask: train_data[3],
                                                                            input_train_images_width: train_data[5],
                                                                            input_train_gauss_tags: train_data[6],
                                                                            input_train_char_sizes: train_data[8],
                                                                            input_train_gauss_mask: train_data[9]})

                summary_writer.add_summary(summary=merge_summary_value, global_step=start_iter)
                if start_iter % args.eval_iter == 0:
                    val_pred_value_all = []
                    val_labels = []
                    for eval_iter in range(val_data_gen.num_samples // args.val_batch_size):
                        val_data = val_data_gen.get_batch()
                        if val_data is None:
                            break
                        print("Evaluation: [{} / {}]".format(eval_iter, (val_data_gen.num_samples // args.val_batch_size)))
                        val_pred_value  = sess.run(val_pred, feed_dict={input_val_images: val_data[0],
                                                                       input_val_labels: val_data[1],
                                                                       input_val_labels_mask: val_data[2],
                                                                       input_val_images_width: val_data[5]})
                        val_pred_value_all.extend(val_pred_value)
                        val_labels.extend(val_data[4])

                    val_data_gen.reset()
                    train_metrics_result = calc_metrics(idx2label(train_pred_value), train_data[4], metrics_type="accuracy")
                    val_metrics_result = calc_metrics(idx2label(np.array(val_pred_value_all)), val_labels, metrics_type="accuracy")
                    print("Evaluation Iter {} train accuracy: {:3f} test accuracy {:3f}".format(start_iter, train_metrics_result, val_metrics_result))
                    log_file.write("Evaluation Iter {} train accuracy: {:3f} test accuracy {:3f}\n".format(start_iter, train_metrics_result, val_metrics_result))

                    if val_metrics_result >= val_best_acc:
                        print("Better results! Save checkpoitns to {}".format(best_model_save_path))
                        val_best_acc = val_metrics_result
                        best_saver.save(sess, best_model_save_path, global_step=global_step)

            if start_iter % args.save_iter == 0:
                print("Iter {} save to checkpoint".format(start_iter))
                saver.save(sess, model_save_path, global_step=global_step)
    log_file.close()
if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_train(args)