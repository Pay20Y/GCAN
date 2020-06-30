import tensorflow as tf
from module.Backbone import Backbone, Backbone_v2
from module.Encoder import Encoder
from module.Decoder import Decoder
from module.attention_loss import params_regress_loss, KLDivLoss, KLDivLossContirb, CrossEntropyLoss, two_d_gaussian_kl_div_loss, attention_regress_loss
from tensorflow.contrib import slim
class SARModel(object):
    def __init__(self,
                 num_classes,
                 encoder_dim=512,
                 encoder_layer=2,
                 decoder_dim=512,
                 decoder_layer=2,
                 decoder_embed_dim=512,
                 seq_len=40,
                 beam_width=5,
                 is_training=True,
                 att_loss_type="kldiv",
                 att_loss_weight=1.):
        self.num_classes = num_classes
        self.encoder_dim = encoder_dim
        self.encoder_layer = encoder_layer
        self.decoder_dim = decoder_dim
        self.decoder_layer = decoder_layer
        self.decoder_embed_dim = decoder_embed_dim
        self.seq_len = seq_len
        self.beam_width = beam_width
        self.is_training = is_training
        self.att_loss_type = att_loss_type
        self.att_loss_weight = att_loss_weight
        # if att_loss_type == 'kldiv':
        #     print("With KLDivLoss")
        #     self.att_loss_op = KLDivLoss
        # elif att_loss_type == 'l1':
        #     print("With Smooth L1 Loss")
        #     self.att_loss_op = params_regress_loss
        # else:
        #     print("Unimplemented loss type {}".format(att_loss_type))

        self.backbone = Backbone(is_training=self.is_training)
        # self.backbone = Backbone_v2(is_training=self.is_training)
        self.encoder = Encoder(hidden_nums=self.encoder_dim, layer_nums=self.encoder_layer, is_training=self.is_training)
        self.decoder = Decoder(output_classes=self.num_classes,
                               hidden_nums=self.decoder_dim,
                               layer_nums=self.decoder_layer,
                               embedding_dim=self.decoder_embed_dim ,
                               seq_len=self.seq_len,
                               lstm_keep_prob=1.0,
                               att_keep_prob=0.5,
                               is_training=self.is_training)

    def __call__(self, input_images, input_labels, input_widths, batch_size, reuse=False, decode_type='greed'):
        with tf.variable_scope(name_or_scope="sar", reuse=reuse):
            encoder_state,  feature_map, mask_map = self.inference(input_images, input_widths, batch_size)
            decoder_logits, attention_weights, pred, attention_params = self.decode(encoder_state, feature_map, input_labels, mask_map, decode_type=decode_type)

            return decoder_logits, attention_weights, pred, attention_params

    def inference(self, input_images, input_widths, batch_size):
        # with tf.variable_scope(name_or_scope='sar', reuse=reuse):
        img_W = tf.cast(tf.shape(input_images)[2], tf.float32)
        feature_map = self.backbone(input_images)
        fea_W = tf.cast(tf.shape(feature_map)[2], tf.float32)
        input_widths = tf.cast(tf.math.floor(input_widths * (fea_W / img_W)), tf.int32)
        encoder_state = self.encoder(feature_map, input_widths)

        with tf.name_scope(name="fea_post_process"):
            # construct mask map
            input_widths_list = tf.split(input_widths, batch_size)
            mask_map = []
            for i, width in enumerate(input_widths_list):
                mask_slice = tf.pad(tf.zeros(dtype=tf.float32, shape=width), [[0, tf.shape(feature_map)[2]-width[0]]], constant_values=1)
                mask_slice = tf.tile(tf.expand_dims(mask_slice, axis=0), [tf.shape(feature_map)[1], 1])
                mask_map.append(mask_slice)
            # mask_map = tf.expand_dims(tf.zeros_like(feature_map[:, :, :, 0]), axis=-1)  # N * H * W * 1
            mask_map = tf.stack(mask_map, axis=0)
            mask_map = tf.expand_dims(mask_map, axis=3) # N * H * W * 1
            reverse_mask_map = 1 - mask_map
            feature_map = feature_map * reverse_mask_map

        return encoder_state, feature_map, mask_map

    def loss(self, pred, attention, input_labels, input_attentoin_labels, input_lengths_mask, input_gauss_tags, input_gauss_mask=None, input_char_sizes=None):
        with tf.name_scope(name="MaskCrossEntropyLoss"):
            N, T, _ = pred.shape.as_list()
            N = N if N is not None else tf.shape(pred)[0]
            input_labels = tf.one_hot(input_labels, self.num_classes, 1, 0)  # N * L * C
            input_labels = tf.stop_gradient(input_labels)  # since softmax_cross_entropy_with_logits_v2 will bp to labels

            recog_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels, logits=pred, dim=-1)
            mask_recog_loss = recog_loss * tf.cast(input_lengths_mask, tf.float32)
            mask_recog_loss = tf.reduce_sum(mask_recog_loss) / tf.cast(N, tf.float32)

            input_attentoin_labels = tf.stop_gradient(input_attentoin_labels)
            if self.att_loss_type == 'kldiv':
                print("With KLDivLoss")
                attention = tf.reshape(attention, [N, T, -1])
                input_attentoin_labels = tf.reshape(input_attentoin_labels, [N, T, -1])
                attention_loss = KLDivLoss(attention, input_attentoin_labels)
                # attention_loss = KLDivLossContirb(attention, input_attentoin_labels)
            elif self.att_loss_type == 'l1' or self.att_loss_type == 'l2':
                print("With Smooth L1 Loss")
                attention = tf.reshape(attention, [N, T, -1])
                input_attentoin_labels = tf.reshape(input_attentoin_labels, [N, T, -1])
                # attention_loss = params_regress_loss(attention, input_attentoin_labels, img_size=None, char_size=input_char_sizes, type=self.att_loss_type)
                attention_loss = attention_regress_loss(attention, input_attentoin_labels, type=self.att_loss_type)
            elif self.att_loss_type == 'ce':
                print("With cross-entropy loss")
                # mask_attention = tf.multiply(attention, input_gauss_mask)
                mask_attention = tf.reshape(attention, [N, T, -1])
                input_attentoin_labels = tf.reshape(input_attentoin_labels, [N, T, -1])
                attention_loss = CrossEntropyLoss(mask_attention, input_attentoin_labels)
            elif self.att_loss_type == 'gausskldiv':
                print("With gaussian KLDivLoss")
                attention_loss = two_d_gaussian_kl_div_loss(attention, input_attentoin_labels)
            # mask_attention_loss = attention_loss * tf.cast(input_lengths_mask, tf.float32)
            mask_attention_loss = attention_loss * tf.cast(input_gauss_tags, tf.float32)

            mask_attention_loss = tf.reduce_sum(mask_attention_loss) / tf.cast(N, tf.float32)
            loss = self.att_loss_weight * mask_attention_loss + mask_recog_loss

            return loss, mask_recog_loss, mask_attention_loss

    def loss_kl(self, pred, alphas, input_labels, input_gauss_labels, input_lengths_mask, input_gauss_tags):
        """
        cross-entropy loss
        :param pred: Decoder outputs N * L * C
        :param alphas: N * T * H * W
        :param input_labels: N * L
        :param input_gauss_labels: N * T * H * W
        :param input_lengths_mask: N * L (0 & 1 like indicating the real length of the label)
        :return:
        """
        def kl_double_loss(x, y):
            with tf.name_scope("kl_loss"):
                x = tf.distributions.Categorical(probs=x)
                y = tf.distributions.Categorical(probs=y)
                return 0.5 * (tf.distributions.kl_divergence(x, y) + tf.distributions.kl_divergence(y, x))

        with tf.name_scope(name="MaskCrossEntropyLoss"):
            N = pred.shape.as_list()[0]
            N = N if N is not None else tf.shape(pred)[0]
            input_labels = tf.one_hot(input_labels, self.num_classes, 1, 0) # N * L * C
            input_labels = tf.stop_gradient(input_labels) # since softmax_cross_entropy_with_logits_v2 will bp to labels
            N, T, H, W = input_gauss_labels.shape.as_list()
            input_gauss_labels = tf.reshape(input_gauss_labels, [N, T, -1])
            input_gauss_labels = input_gauss_labels + 1e-5
            input_gauss_labels = tf.stop_gradient(input_gauss_labels)
            alphas = tf.reshape(alphas, [N, T, -1])
            alphas = alphas + 1e-5

            kl_loss = kl_double_loss(alphas, input_gauss_labels) # N * T
            mask_kl_loss = kl_loss * input_gauss_tags
            mask_kl_loss = tf.reduce_sum(mask_kl_loss) / tf.cast(N, tf.float32)

            recog_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels, logits=pred, dim=-1)
            mask_recog_loss = recog_loss * tf.cast(input_lengths_mask, tf.float32)
            mask_recog_loss = tf.reduce_sum(mask_recog_loss) / tf.cast(N, tf.float32)

            loss = mask_kl_loss + mask_recog_loss

            return loss, mask_recog_loss, mask_kl_loss

    def decode(self, encoder_state, feature_map, input_labels, mask_map, decode_type='greed'):
        assert decode_type.lower() in ['greed', 'beam', 'lexicon'], "Not support decode type"
        # with tf.variable_scope(name_or_scope='sar', reuse=reuse):
        if decode_type.lower() == "greed":
            decoder_outputs, attention_weights, attention_params = self.decoder(encoder_state, feature_map, input_labels, mask_map)
            pred = tf.argmax(decoder_outputs, axis=2)
            return decoder_outputs, attention_weights, pred, attention_params
        elif decode_type == "beam" and not self.is_training:
            pred, attention_weights = self.decoder.beam_search(encoder_state, feature_map, mask_map, self.beam_width)
            return None , attention_weights, pred, None
        elif decode_type == "lexicon":
            return None

def test():
    input_images = tf.placeholder(dtype=tf.float32, shape=[32, 48, 160, 3])
    input_labels = tf.placeholder(dtype=tf.int32, shape=[32, 40])
    input_lengths = tf.placeholder(dtype=tf.int32, shape=[32, 40])
    input_feature_map = tf.placeholder(dtype=tf.float32, shape=[32, 12, 20, 512])
    input_widths = tf.placeholder(dtype=tf.float32, shape=[32])

    sar_model = SARModel(95)
    encoder_state, feature_map, mask_map = sar_model.inference(input_images, input_widths, batch_size=32)
    logits, att_weights, pred = sar_model.decode(encoder_state, feature_map, input_labels, mask_map)
    loss = sar_model.loss(logits, input_labels, input_lengths)

    optimizer = tf.train.AdamOptimizer(learning_rate=1.0).minimize(loss)
    import numpy as np
    _input_images = np.random.rand(32, 48, 160, 3)
    _input_labels = np.random.randint(0,95,size=[32,40])
    _input_lenghts = np.random.randint(0,2,size=[32,40])
    _input_feature_map = np.random.rand(32, 12, 20, 512)
    _input_images_width = np.random.randint(10, 30, 32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            _, loss_value = sess.run([optimizer, loss], feed_dict={input_images: _input_images,
                                                                   input_labels: _input_labels,
                                                                   input_lengths: _input_lenghts,
                                                                   input_feature_map: _input_feature_map,
                                                                   input_widths: _input_images_width})
            print(loss_value)

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    test()