import string
import numpy as np
import cv2
import math
import data_provider.TPS as tps
import itertools

def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):

    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

def find_min_rectangle(points):
    assert len(points.shape) == 2
    # if len(points) == 4:
    #     x1, y1, x2, y2 = points
    #     x3, y3, x4, y4 = x2, y1, x1, y2
    # elif len(points) == 8:
    #     x1, y1, x2, y2, x3, y3, x4, y4 = points
    # else:
    #     raise("please input 2 points or 4 points, check it")
    # lt_x = min(x1, x2, x3, x4)
    # lt_y = min(y1, y2, y3, y4)
    # rd_x = max(x1, x2, x3, x4)
    # rd_y = max(y1, y2, y3, y4)
    lt_x, rd_x = np.min(points[:, 0]), np.max(points[:, 0])
    lt_y, rd_y = np.min(points[:, 1]), np.max(points[:, 1])

    # return np.float32([[lt_x, lt_y], [rd_x, lt_y], [rd_x, rd_y], [lt_x, rd_y]]), int(math.ceil(rd_x - lt_x)), int(math.ceil(rd_y - lt_y))
    return np.float32([[lt_x, lt_y], [rd_x, lt_y], [rd_x, rd_y], [lt_x, rd_y]]), int(rd_x - lt_x), int(rd_y - lt_y)

def get_distrib_params(polygons):
    min_x, max_x = float(np.min(polygons[:, 0])), float(np.max(polygons[:, 0]))
    min_y, max_y = float(np.min(polygons[:, 1])), float(np.max(polygons[:, 1]))

    mu_x = (max_x + min_x) / 2.
    mu_y = (max_y + min_y) / 2.

    sigma_x = (max_x - min_x) * (max_x - min_x) / 4.
    sigma_y = (max_y - min_y) * (max_y - min_y) / 4.

    return np.stack([mu_x, mu_y, sigma_x, sigma_y], axis=0)

def get_gauss_distrib(kernel_size):
    # shrink
    sigma_x = math.ceil(0.4 * kernel_size[1])
    sigma_y = math.ceil(0.4 * kernel_size[0])

    # ky = cv2.getGaussianKernel(kernel_size[0], int(kernel_size[0] / 2))
    # kx = cv2.getGaussianKernel(kernel_size[1], int(kernel_size[1] / 2))
    ky = cv2.getGaussianKernel(kernel_size[0], int(sigma_y / 2.))
    kx = cv2.getGaussianKernel(kernel_size[1], int(sigma_x / 2.))
    return np.multiply(ky, np.transpose(kx))

def estim_gauss_params(gauss, delta_x, delta_y):
    coord_x = np.arange(delta_x).astype(np.float32) # W
    coord_y = np.arange(delta_y).astype(np.float32) # H

    gauss_x = np.sum(gauss, axis=0) # W
    gauss_y = np.sum(gauss, axis=1) # H

    mu_x = np.matmul(np.expand_dims(gauss_x, axis=0), np.expand_dims(coord_x, axis=1))
    mu_y = np.matmul(np.expand_dims(gauss_y, axis=0), np.expand_dims(coord_y, axis=1))

    sigma_x = np.matmul(np.expand_dims(gauss_x, axis=0), np.expand_dims(np.power(coord_x, 2), axis=1)) - np.power(mu_x, 2)  # 1 * 1
    sigma_y = np.matmul(np.expand_dims(gauss_y, axis=0), np.expand_dims(np.power(coord_y, 2), axis=1)) - np.power(mu_y, 2)  # 1 * 1

    mu_x, mu_y, sigma_x, sigma_y = np.squeeze(mu_x), np.squeeze(mu_y), np.squeeze(sigma_x), np.squeeze(sigma_y)

    return np.stack([mu_x, mu_y, sigma_x, sigma_y], axis=0).astype(np.float32)

def construct_gauss_distirb(params, delta_x, delta_y):
    # Bugs?
    mu_x = params[0]
    mu_y = params[1]
    sigma_x = params[2]
    sigma_y = params[3]

    start_x = int(math.floor(mu_x - delta_x / 2))
    end_x = int(start_x + delta_x)
    start_y = int(math.floor(mu_y - delta_y / 2))
    end_y = int(start_y + delta_y)


    coord_x = np.arange(start_x, end_x).astype(np.float32)
    coord_y = np.arange(start_y, end_y).astype(np.float32)

    gauss_x = np.exp(-1. * np.power(coord_x - mu_x, 2) / (2. * sigma_x)) # delta_x
    gauss_y = np.exp(-1. * np.power(coord_y - mu_y, 2) / (2. * sigma_y)) # delta_y

    # coefficient_x = 1. / (np.sqrt(2. * math.pi * sigma_x))
    # coefficient_y = 1. / (np.sqrt(2. * math.pi * sigma_y))

    # gauss_x = coefficient_x * gauss_x
    # gauss_y = coefficient_y * gauss_y

    gauss = np.matmul(np.expand_dims(gauss_y, axis=1), np.expand_dims(gauss_x, axis=0))

    # try:
    #     gauss = gauss / np.sum(gauss)
    # except Exception as e:
    #     print(e)
    return gauss

def aff_gaussian(gaussian, box, pts, deta_x, deta_y):
    """
    :param gaussian:
    :param box: min-bounding rectangle 4 * 2
    :param pts: real bounding polygon 4 * 2
    :param deta_x:
    :param deta_y:
    :return:
    """
    de_x, de_y = box[0] # left-top point
    box = box - [de_x, de_y]
    pts = pts - [de_x, de_y]
    M = cv2.getPerspectiveTransform(box, pts)
    res = cv2.warpPerspective(gaussian, M, (deta_x, deta_y))
    return res

def sum_norm(inputs, axis=-1):
    inputs[inputs==0] = 1e-5
    norm = inputs / np.expand_dims(np.sum(inputs, axis=axis), axis=axis)
    return norm

def roi_sum(inputs, target_h=6, target_w=40):
    """
    roi with sum
    :param inputs: T * H * W
    :return:
    """

    T, H, W = inputs.shape
    assert H % target_h == 0 and W % target_w == 0, "roi sum only for dividing exactly"


    # For H
    group_h = np.split(inputs, target_h, axis=1) # [T * 8 * 40 ...]
    group_h = [np.sum(g, axis=1) for g in group_h] # [T * 40]

    inputs = np.stack(group_h, axis=1) # N * 6 * 160

    # For W
    group_w = np.split(inputs, target_w, axis=2)  # [T * 6 * 4 ...]
    group_w = [np.sum(g, axis=2) for g in group_w]  # [T * 6]

    outputs = np.stack(group_w, axis=2) # N * 6 * 40

    return outputs

def roi_max(inputs, target_h=6, target_w=40):
    """
    roi with sum
    :param inputs: T * H * W
    :return:
    """

    T, H, W = inputs.shape
    assert H % target_h == 0 and W % target_w == 0, "roi sum only for dividing exactly"


    # For H
    group_h = np.split(inputs, target_h, axis=1) # [T * 8 * 40 ...]
    group_h = [np.max(g, axis=1) for g in group_h] # [T * 40]

    inputs = np.stack(group_h, axis=1) # N * 6 * 160

    # For W
    group_w = np.split(inputs, target_w, axis=2)  # [T * 6 * 4 ...]
    group_w = [np.max(g, axis=2) for g in group_w]  # [T * 6]

    outputs = np.stack(group_w, axis=2) # N * 6 * 40

    return outputs

def rotate_img(img, angle, BBs=None, scale=1):
    H, W, _ = img.shape
    rangle = np.deg2rad(angle)  # angle in radians
    new_width = (abs(np.sin(rangle) * H) + abs(np.cos(rangle) * W)) * scale
    new_height = (abs(np.cos(rangle) * H) + abs(np.sin(rangle) * W)) * scale

    rot_mat = cv2.getRotationMatrix2D((new_width * 0.5, new_height * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_width - W) * 0.5, (new_height - H) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(new_width)), int(math.ceil(new_height))), flags=cv2.INTER_LANCZOS4)

    rot_bboxes = list()
    if BBs is not None:
        for bbox in BBs:
            new_box = []
            for point in bbox:
                r_point = np.dot(rot_mat, np.array([point[0], point[1], 1]))
                new_box.append(r_point)
            rot_bboxes.append(new_box)
            # point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
    rot_bboxes = np.array(rot_bboxes)
    return rot_img, rot_bboxes, (int(math.ceil(new_width)), int(math.ceil(new_height)))

def adding_guass(image, param=30, grayscale=255):
    w=image.shape[1]
    h=image.shape[0]
    for i in range(3):
        img = image[:,:,i]
        newimg=np.zeros((h,w),np.uint8)
        for x in range(0,h):
            for y in range(0,w-1,2):
                r1=np.random.random_sample()
                r2=np.random.random_sample()
                z1=param*np.cos(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))
                z2=param*np.sin(2*np.pi*r2)*np.sqrt((-2)*np.log(r1))

                fxy=int(img[x,y]+z1)
                fxy1=int(img[x,y+1]+z2)
                #f(x,y)
                if fxy<0:
                    fxy_val=0
                elif fxy>grayscale:
                    fxy_val=grayscale
                else:
                    fxy_val=fxy
                #f(x,y+1)
                if fxy1<0:
                    fxy1_val=0
                elif fxy1>grayscale:
                    fxy1_val=grayscale
                else:
                    fxy1_val=fxy1
                newimg[x,y]=fxy_val
                newimg[x,y+1]=fxy1_val

        image[:,:,i] = newimg
    return image

def WarpImage_TPS(source,target,img):
    tps = cv2.createThinPlateSplineShapeTransformer()

    print(source)
    print(target)
    source=source.reshape(-1,len(source),2)
    target=target.reshape(-1,len(target),2)

    matches=list()
    for i in range(0,len(source[0])):
        matches.append(cv2.DMatch(i,i,0))

    tps.estimateTransformation(target,source,matches)
    # tps.estimateTransformation(source, target,matches)
    new_img = tps.warpImage(img)
    return new_img

def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def tps_aug_bak(img, alpha, num_control_points=20):
    H, W, _ = img.shape
    # First pad the image?
    # img_pad = np.zeros(dtype=np.uint8, shape=[H*2, W, 3])
    # img_pad[:H, :, :] = img

    # Second construct control points
    def linear_func(x, a, b):
        return a * x + b
    def quadratic_func(x, a, b, c):
        return a * math.pow((x-b), 2) + c

    sample_dis = int(math.floor(W // num_control_points))
    sample_x = list(range(0, W, sample_dis)) + list(range(0, W, sample_dis))
    mid_point = sample_x[len(sample_x) // 4 + 1]
    source_sample_y = [0] * (len(sample_x)//2) + [H] * (len(sample_x)//2)
    source_control_points = np.stack([sample_x, source_sample_y], axis=-1).astype(np.float32)
    source_control_points[:, 0] = source_control_points[:, 0] / W
    # source_control_points[:, 1] = source_control_points[:, 1] / (2 * H)
    source_control_points[:, 1] = source_control_points[:, 1] / H
    # TODO new points sampler
    # Check is the alpha valid
    if alpha >= 0:
        alpha = alpha if alpha <= (float(H)/(mid_point*mid_point)) else round(float(H)/(mid_point * mid_point), 3)
        target_sample_y = [quadratic_func(x, alpha, mid_point, 0) for x in sample_x[:len(sample_x) // 2]] + \
                          [quadratic_func(x, alpha, mid_point, H / 2) for x in sample_x[len(sample_x) // 2:]]
        # target_sample_y = source_sample_y[:len(sample_x) // 2] + [linear_func(x, alpha, H / 2) for x in sample_x[len(sample_x) // 2:]]
    else:
        alpha = alpha if alpha >= (float(H) / (mid_point * mid_point)) else round(float(H) / (mid_point * mid_point), 3)
        target_sample_y = [quadratic_func(x, alpha, mid_point, H / 2) for x in sample_x[:len(sample_x) // 2]] + \
                          [quadratic_func(x, alpha, mid_point, H) for x in sample_x[len(sample_x) // 2:]]
        # target_sample_y = source_sample_y[:len(sample_x) // 2] + [linear_func(x, alpha, H / 2) for x in sample_x[len(sample_x) // 2:]]

    target_sample_y = np.clip(target_sample_y, 0, H)

    target_control_points = np.stack([sample_x, target_sample_y], axis=-1).astype(np.float32)
    target_control_points[:, 0] = target_control_points[:, 0] / W
    # target_control_points[:, 1] = target_control_points[:, 1] / (2 * H)
    target_control_points[:, 1] = target_control_points[:, 1] / H
    # tps_img = WarpImage_TPS(source_control_points, target_control_points, img)
    tps_img = warp_image_cv(img, source_control_points, target_control_points)

    return tps_img

def tps_aug(img, alpha, num_control_points=20, scale=0.4):
    H, W, _ = img.shape
    # First pad the image?
    img_pad = np.zeros(dtype=np.uint8, shape=[H*2, W, 3])
    img_pad[:H, :, :] = img

    # Second construct control points
    def linear_func(x, a, b):
        return a * x + b
    def quadratic_func(x, a, b, c):
        return a * math.pow((x-b), 2) + c
    def quadratic_der(x, a, b):
        return 2 * a * x - 2 * a * b

    # sample_dis = int(math.floor(W // num_control_points))
    # sample_dis = sample_dis if sample_dis > 0 else 1
    # sample_x = list(range(0, W, sample_dis)) + list(range(0, W, sample_dis))
    sample_x = np.concatenate([np.unique(np.round(np.linspace(0, W, num_control_points))), np.unique(np.round(np.linspace(0, W, num_control_points)))])
    mid_point = np.median(sample_x)
    source_sample_y = np.array([0] * (len(sample_x)//2) + [H] * (len(sample_x)//2)).astype(np.int32)
    source_control_points = np.stack([sample_x, source_sample_y], axis=-1).astype(np.float32)
    source_control_points[:, 0] = source_control_points[:, 0] / W
    source_control_points[:, 1] = source_control_points[:, 1] / (2 * H)

    # TODO new points sampler
    # Here we using vertical line in the tangential direction of longer edge to get the target points x-axis
    # Check is the alpha valid
    if alpha >= 0:
        # alpha = alpha if alpha <= (float(H)/(mid_point*mid_point)) else round(float(H)/(mid_point * mid_point), 3)
        # Longer edge sample points
        target_sample_x_0 = sample_x[:len(sample_x) // 2]
        target_sample_y_0 = [quadratic_func(x, alpha, mid_point, 0) for x in target_sample_x_0]

        # Choice 1: Shorter edge sample points
        # Using line and x-axis cross points
        # tang_dirt = quadratic_der(0, alpha, mid_point)
        # target_sample_x_start = -1 * tang_dirt * (H - quadratic_func(0, alpha, mid_point, 0))
        # target_sample_x_end = 2 * mid_point - target_sample_x_start
        # target_sample_x_1 = np.linspace(target_sample_x_start, target_sample_x_end, len(target_sample_x_0))

        # Choice 2: Or just scale shorter
        scale_length = scale * W
        target_sample_x_1 = np.linspace(scale_length + 0, W - scale_length, len(target_sample_x_0))
        target_sample_y_1 = [quadratic_func(x, alpha, mid_point, H) for x in target_sample_x_0]

        # Choice 3: adaptive scale factor
        # tang_dirt = -1. / quadratic_der(0, alpha, mid_point)
        # scale_length = H / tang_dirt
        # target_sample_x_1 = np.linspace(scale_length + 0, W - scale_length, len(target_sample_x_0))
        # target_sample_y_1 = [quadratic_func(x, alpha, mid_point, H) for x in target_sample_x_0]

        # Construct the sample points
        target_sample_x = np.concatenate([target_sample_x_0, target_sample_x_1])
        target_sample_y = np.concatenate([target_sample_y_0, target_sample_y_1])

    else:
        # Longer edge sample points
        target_sample_x_0 = sample_x[len(sample_x) // 2 :]
        target_sample_y_0 = [quadratic_func(x, alpha, mid_point, (2 * H)) for x in target_sample_x_0]

        # Choice 1: Shorter edge sample points
        # Shorter edge sample points
        # tang_dirt = quadratic_der(0, alpha, mid_point)
        # target_sample_x_start = tang_dirt * quadratic_func(0, alpha, mid_point, H)
        # target_sample_x_end = 2 * mid_point - target_sample_x_start
        # target_sample_x_1 = np.linspace(target_sample_x_start, target_sample_x_end, len(target_sample_x_0))

        # Choice 2: Or just scale shorter
        scale_length = scale * W
        target_sample_x_1 = np.linspace(scale_length + 0, W - scale_length, len(target_sample_x_0))
        target_sample_y_1 = [quadratic_func(x, alpha, mid_point, H) for x in target_sample_x_0]

        # Choice 3: adaptive scale factor
        # tang_dirt = -1. / quadratic_der(0, alpha, mid_point)
        # scale_length = H / tang_dirt
        # target_sample_x_1 = np.linspace(scale_length + 0, W - scale_length, len(target_sample_x_0))
        # target_sample_y_1 = [quadratic_func(x, alpha, mid_point, H) for x in target_sample_x_0]

        # Construct the sample points
        target_sample_x = np.concatenate([target_sample_x_1, target_sample_x_0])
        target_sample_y = np.concatenate([target_sample_y_1, target_sample_y_0])

        # alpha = alpha if alpha >= (float(H) / (mid_point * mid_point)) else round(float(H) / (mid_point * mid_point), 3)
        # target_sample_y = [quadratic_func(x, alpha, mid_point, H) for x in sample_x[:len(sample_x) // 2]] + \
        #                   [quadratic_func(x, alpha, mid_point, (2 * H)) for x in sample_x[len(sample_x) // 2:]]
        # target_sample_y = source_sample_y[:len(sample_x) // 2] + [linear_func(x, alpha, H / 2) for x in sample_x[len(sample_x) // 2:]]

    target_sample_y = np.clip(target_sample_y, 0, (2 * H))

    target_control_points = np.stack([sample_x, target_sample_y], axis=-1).astype(np.float32)
    target_control_points[:, 0] = target_control_points[:, 0] / W
    target_control_points[:, 1] = target_control_points[:, 1] / (2 * H)
    # target_control_points[:, 1] = target_control_points[:, 1] / H
    # tps_img = WarpImage_TPS(source_control_points, target_control_points, img)
    tps_img = warp_image_cv(img_pad, source_control_points, target_control_points)

    return tps_img

"""
class TPS_aug(object):
    def __init__(self, control_points):
        self.control_points = control_points

    def construct_control_points(self, alpha, img_height, img_width):
        def quadratic_func(x, a, c):
            return a * math.pow(x, 2) + c

        sample_x = list(range(-img_width // 2, 0, (img_width//self.control_points))) + list(range(0, img_width // 2, (img_width//self.control_points)))
        source_y = [img_height] * len(sample_x) + [0] * len(sample_x)

        # if is_convex:
        #     target_y = [quadratic_func(x, alpha, 0) for x in sample_x] + [quadratic_func(x, alpha, -img_height) for x in sample_x]
        # else:
        #     target_y = [quadratic_func(x, alpha, img_height) for x in sample_x] + [quadratic_func(x, alpha, 0) for x in sample_x]

        if alpha < 0:
            target_y = [quadratic_func(x, alpha, img_height) for x in sample_x] + [quadratic_func(x, alpha, 0) for x in sample_x]
        else:
            target_y = [quadratic_func(x, alpha, 0) for x in sample_x] + [quadratic_func(x, alpha, -img_height) for x in sample_x]

        sample_x = np.array(sample_x * 2)
        source_y = np.array(source_y)
        target_y = np.array(target_y).astype(np.int32)

        source_ctrl_points = np.concatenate([np.expand_dims(sample_x, axis=1), np.expand_dims(source_y, axis=1)], axis=1) # 2c * 2
        target_ctrl_points = np.concatenate([np.expand_dims(sample_x, axis=1), np.expand_dims(target_y, axis=1)], axis=-1) # 2c * 2

        return source_ctrl_points, target_ctrl_points

    def compute_partial_repr(self, input_points, control_points):
        N = input_points.shape[0]
        M = control_points.shape[0]
        pairwise_diff = input_points.reshape(N, 1, 2) - control_points.reshape(1, M, 2)
        # original implementation, very slow
        # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
        pairwise_diff_square = pairwise_diff * pairwise_diff
        pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
        repr_matrix = 0.5 * pairwise_dist * np.log(pairwise_dist)
        # fix numerical error for 0 * log(0), substitute all nan with 0
        mask = repr_matrix != repr_matrix
        repr_matrix[mask] = 0.
        return repr_matrix

    def bilinear_sampler(self, input, coordinates):
        H, W, _ = input.shape
        max_x = W - 1
        max_y = H - 1
        x0 = np.floor(coordinates[:, 0]).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(coordinates[:, 1]).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, -W, max_x)
        x1 = np.clip(x1, -W, max_x)
        y0 = np.clip(y0, 0, max_y)
        y1 = np.clip(y1, 0, max_y)

        Ia = input[x0, y0]
        Ib = input[x0, y1]
        Ic = input[x1, y0]
        Id = input[x1, y1]

        x0 = x0.astype(np.float32)
        x1 = x1.astype(np.float32)
        y0 = y0.astype(np.float32)
        y1 = y1.astype(np.float32)

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

    def __call__(self, input, alpha=0):
        H, W, _ = input.shape
        # source_control_points, target_control_points = self.construct_control_points(alpha, H, W)
        source_control_points = np.array([[-W//2,H], [0,H], [W//2,H], [-W//2,0], [0,0] ,[W //2,0]])
        target_control_points = np.array([[-W//2,H], [0,H], [W//2,H], [-W//2,0], [0,0] ,[W //2,0]])
        Y = np.concatenate([source_control_points, np.zeros([3, 2], dtype=np.int32)], axis=0)

        N = source_control_points.shape[0]

        reper_matrix_value = self.compute_partial_repr(source_control_points, target_control_points)

        forward_kernel = np.zeros([N + 3, N + 3], dtype=np.float32)
        forward_kernel[:N, :N] = reper_matrix_value
        forward_kernel[:N, -3] = 1
        forward_kernel[-3, :N] = 1
        forward_kernel[:N, -2:] = target_control_points
        forward_kernel[-2:, :N] = target_control_points.T
        forward_kernel = np.matrix(forward_kernel).I

        mapping_matrix = np.matmul(forward_kernel, Y)

        half_width = W // 2
        # if alpha < 0:
        #     target_coordinate = np.array(list(itertools.product(range(-half_width, W-half_width), range(-H, H)))).astype(np.int32) # coordinates of a image
        # else:
        #     target_coordinate = np.array(list(itertools.product(range(-half_width, W-half_width), range(0, 2*H)))).astype(np.int32) # coordinates of a image

        target_coordinate = np.array(list(itertools.product(range(-half_width, W-half_width), range(-H, H)))).astype(np.int32) # coordinates of a image

        target_reper_matrix_value = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = np.concatenate([target_reper_matrix_value, np.ones([target_coordinate.shape[0], 1], dtype=np.float32), target_coordinate.astype(np.float32)], axis=1)
        source_coordinate = np.matmul(target_coordinate_repr, mapping_matrix)
        source_coordinate[:, 1] = np.clip(source_coordinate[:, 1], 0, H-1)

        # tps_image = np.zeros(shape=[2*H, W], dtype=np.uint8)
        tps_image = self.bilinear_sampler(input, source_coordinate.getA())

        print("Pause")
"""



if __name__ == "__main__":
    # tps_module = TPS_aug(20)
    import os
    import random
    # img = cv2.imread("bedroom_91_73_3.jpg")
    # tps_aug(img, 0.037)
    # tps_module(img)
    for i, img_name in enumerate(os.listdir("/home/qz/data/SynthTextCrop_800K/17")):
        if i > 500:
            break
        img = cv2.imread(os.path.join("/home/qz/data/SynthTextCrop_800K/17", img_name))
        H, W, _ = img.shape
        valid_alpha_abs = float(H) / ((W//2 + 1) * (W//2 + 1))
        alpha = random.uniform(-1*valid_alpha_abs, valid_alpha_abs)
        tps_img = tps_aug(img, alpha)
        concat_img = np.zeros(dtype=np.uint8, shape=[3 * H + 10, W, 3])
        concat_img[:H, :, :] = img
        concat_img[(H+10):, :, :] = tps_img
        cv2.imwrite(os.path.join('tps_aug_samples', str(i)+'.jpg'), concat_img)