import tensorflow as tf

def smooth_L1(inputs, targets):
    inside = tf.subtract(inputs, targets)
    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside), 1.0), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside, inside), 0.5)
    smooth_l1_option2 = tf.subtract(tf.abs(inside), 0.5)

    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    return smooth_l1_result

def L2_loss(inputs, targets, reducation=True):
    with tf.name_scope("l2_regress_loss"):
        inside = tf.pow(tf.subtract(inputs, targets), 2) # N * T * 4
        if reducation:
            l2_loss = 0.5 * tf.reduce_sum(inside, axis=-1) # N * T
        else:
            l2_loss = 0.5 * inside
        return l2_loss

def params_regress_loss(pred_params, gt_params, img_size=None, char_size=None, type='l1'):
    """
    :param pred_params: N * T * 4
    :param gt_params: N * T * 4
    :param mask: N * T
    :param char_size: N * T * 2
    :param img_size: [H, W]
    :return:
    """
    with tf.name_scope("smooth_l1_regress_loss"):
        # mask = tf.cast(mask, tf.float32)
        N, T, C = pred_params.shape.as_list()

        if img_size is not None:
            img_size_ = tf.concat([img_size, 0.25 * tf.pow(img_size, 2)], axis=0) # 4: [W, H, (0.5*W)^2, (0.5*H)^2]
            img_size_ = tf.cast(img_size_, tf.float32)
            img_size_ = tf.expand_dims(tf.expand_dims(img_size_, axis=0), axis=0)
            img_size_ = tf.tile(img_size_, [N, T, 1]) # N * T * 4: [W, H, (0.5*W)^2, (0.5*H)^2]
            pred_params = pred_params / img_size_
            gt_params = gt_params / img_size_

        if char_size is not None:
            # Normalize
            char_size_ = tf.concat([0.5 * char_size, 0.25 * tf.pow(char_size, 2)], axis=2) # N * T * 4: [0.5*W, 0.5*H, (0.5*W)^2, (0.5*H)^2]
            pred_params = pred_params / char_size_
            gt_params = gt_params / char_size_

        if type == 'l1':
            l1_loss = smooth_L1(tf.reshape(pred_params, [-1, C]), tf.reshape(gt_params, [-1, C]))
            l1_loss = tf.reduce_sum(l1_loss, axis=1)
            loss = tf.reshape(l1_loss, shape=[N, T]) # N * T
        elif type == 'l2':
            loss = L2_loss(pred_params, gt_params) # N * T
        return loss

def attention_regress_loss(x, y, type="l1"):
    """
    regress attention value with gt
    :param x: N * T * (H * W)
    :param y: N * T * (H * W)
    :return:
    """
    N, T, C = x.shape.as_list()
    x = tf.reshape(x, [-1, C])
    y = tf.reshape(y, [-1, C])

    if type == 'l1':
        l1_loss = smooth_L1(x, y)
        l1_loss = tf.reduce_sum(l1_loss, axis=1)
        loss = tf.reshape(l1_loss, shape=[N, T])  # N * T
    else:
        print("Not implement!")
        loss = tf.zeros(dtype=tf.float32, shape=[N, T])

    return loss


def CrossEntropyLoss(x , y, axis=-1, reduction=True):
    with tf.name_scope("cross_entropy_loss"):
        # x = x + 1e-5
        # ce = -1. * tf.multiply(y, tf.log(x))
        # if reduction == True:
        #     ce = tf.reduce_sum(ce, axis=axis)
        # Here we use official implementation please use logits before softmax
        print("Official implementation of cross-entropy be careful of softmax")
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=x, dim=-1)

        return ce

def KLDivLoss(x, y, axis=-1, reduction=True):
    with tf.name_scope("kl_div_loss"):
        D = x.shape.as_list()[axis]
        x = tf.add(x, 1e-5) # avoid nan or inf
        y = tf.add(y, 1e-5)
        kl = tf.multiply(y, tf.log(y) - tf.log(x))
        # kl = tf.where(tf.is_inf(kl), tf.zeros_like(kl), kl)
        # kl = tf.where(tf.is_nan(kl), tf.zeros_like(kl), kl)

        if reduction == True:
            kl = tf.reduce_sum(kl, axis=axis)

        return kl

def KLDivLossContirb(x, y):
    def kl_double_loss(x, y):
        with tf.name_scope("kl_loss"):
            x = x + 1e-5
            y = y + 1e-5
            x = tf.distributions.Categorical(probs=x)
            y = tf.distributions.Categorical(probs=y)
            return 0.5 * (tf.distributions.kl_divergence(x, y) + tf.distributions.kl_divergence(y, x))
    return kl_double_loss(x, y)

def gaussian_kl(p, q, reduction=True):
    """Computes KL divergence between two isotropic Gaussian distributions.

    To ensure numerical stability, this op uses mu, log(sigma^2) to represent
    the distribution. If q is not provided, it's assumed to be unit Gaussian.

    Args:
    q: A tuple (mu, log(sigma^2)) representing a multi-variatie Gaussian. target
    q: N * T * 2 target
    p: A tuple (mu, log(sigma^2)) representing a multi-variatie Gaussian. source
    p: N * T * 2 source
    Returns:
    A tensor representing KL(q, p).
    """
    # mu1, log_sigma1_sq = q
    # mu2, log_sigma2_sq = p
    mu2, sigma_2 = tf.unstack(p, axis=2)
    mu1, sigma_1 = tf.unstack(q, axis=2)
    # log_sigma2_sq = tf.log(sigma_2) # N * T
    # log_sigma1_sq = tf.log(sigma_1) # N * T

    if reduction:
        return tf.reduce_sum(0.5 * (tf.log(sigma_2) - tf.log(sigma_1) + sigma_1 / sigma_2 + tf.square(mu1 - mu2) / sigma_2 - 1), axis=-1) # N
    else:
        return 0.5 * (tf.log(sigma_2) - tf.log(sigma_1) + sigma_1 / sigma_2 + tf.square(mu1 - mu2) / sigma_2 - 1) # N * T


def two_d_gaussian_kl_div_loss(pred_params, gt_params):
    with tf.name_scope("gauss_kl_div_loss"):
        p_mu_x, p_mu_y, p_sigma_x, p_sigma_y = tf.unstack(pred_params, axis=2)
        q_mu_x, q_mu_y, q_sigma_x, q_sigma_y = tf.unstack(gt_params, axis=2)
        p_x = tf.stack([p_mu_x, p_sigma_x], axis=2)
        q_x = tf.stack([q_mu_x, q_sigma_x], axis=2)
        p_y = tf.stack([p_mu_y, p_sigma_y], axis=2)
        q_y = tf.stack([q_mu_y, q_sigma_y], axis=2)

        x_guass_kl = gaussian_kl(p_x, q_x, False)
        y_guass_kl = gaussian_kl(p_y, q_y, False)

        return 0.5 * (x_guass_kl + y_guass_kl)

if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    # demo = np.random.randn(4, 30, 240)
    # demo = tf.constant(dtype=tf.float32, value=demo)
    # label = np.random.randn(4, 30, 240)
    # label = tf.constant(dtype=tf.float32, value=label)
    #
    # loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=demo, dim=-1)
    # loss2 = CrossEntropyLoss(tf.nn.softmax(demo, axis=-1), label)
    #
    # with tf.Session() as sess:
    #     print("Official:", loss1.eval(session=sess))
    #     print("Mine:", loss2.eval(session=sess))

    # demo = np.random.randn(4, 25, 4)
    # demo = tf.constant(dtype=tf.float32, value=demo)
    # label = np.random.randn(4, 25, 4)
    # label = tf.constant(dtype=tf.float32, value=label)
    demo = tf.concat([tf.zeros(dtype=tf.float32, shape=[4, 25, 2]), tf.ones(dtype=tf.float32, shape=[4, 25, 2])], axis=-1)
    label = tf.concat([tf.zeros(dtype=tf.float32, shape=[4, 25, 2]), tf.zeros(dtype=tf.float32, shape=[4, 25, 2])], axis=-1)
    loss = two_d_gaussian_kl_div_loss(demo, label)

    with tf.Session() as sess:
        print(loss.eval())
        print("pause")