import cv2
import numpy as np
import os

def mask_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])

    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break
        att_map = cv2.resize(att_map, (W, H))
        _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
        _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)

        show_attention = cv2.addWeighted(img, 0.5, _att_map, 2, 0)
        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), show_attention)

    return True

def contour_visualize(img, alpha, pred, vis_dir, img_path, num_line=10, equal_bound=1e-4):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])

    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break

        att_map = cv2.resize(att_map, (W, H))
        max_weights = np.max(att_map)
        min_weights = np.max(att_map)

        weights_rank = np.arange(min_weights, max_weights, num_line, dtype=np.float32)

        pass

def line_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha_H, alpha_W = alpha.shape[2], alpha.shape[3]
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])
    for i, att_map in enumerate(alpha): # H * W * 1
        img_line = img.copy()
        if i >= len(pred):
            break
        # att_map = cv2.resize(att_map, (W, H)) # H * W * 1
        x_axis_att_map = np.sum(att_map, axis=0).reshape(alpha_W) # W
        y_axis_att_map = np.sum(att_map, axis=1).reshape(alpha_H) # H

        key_coord_x = np.arange(0, W, W//alpha_W)
        key_coord_y = np.arange(0, H, H//alpha_H)
        coord_x = np.arange(0, W)
        coord_y = np.arange(0, H)
        x_axis_att_value = np.interp(coord_x, key_coord_x, x_axis_att_map)  # 100
        x_axis_att_value = x_axis_att_value * H
        y_axis_att_value = np.interp(coord_y, key_coord_y, y_axis_att_map)  # 100
        y_axis_att_value = y_axis_att_value * H


        x_att_pts = np.stack([coord_x, x_axis_att_value]).transpose((1, 0)).astype(np.int32)
        x_att_pts[:, 1] = H - x_att_pts[:, 1]

        y_att_pts = np.stack([y_axis_att_value, coord_y]).transpose((1, 0)).astype(np.int32)

        img_line = cv2.polylines(img_line, [x_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
        img_line = cv2.polylines(img_line, [y_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)

        att_map = cv2.resize(att_map, (W, H))
        _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
        _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)

        mask_line_img = cv2.addWeighted(img_line, 0.5, _att_map, 2, 0)

        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), mask_line_img)

    return True


def line_sep_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    assert isinstance(alpha, tuple)
    H, W, _ = img.shape
    alpha_H, alpha_W = alpha[0].shape[2], alpha[0].shape[3]
    alpha_x = alpha[0].reshape([-1, alpha_W, 1])
    alpha_y = alpha[0].reshape([-1, alpha_H, 1])

    for i, (att_map_x, att_map_y)in enumerate(zip(alpha_x, alpha_y)): # H * W * 1
        img_line = img.copy()
        if i >= len(pred):
            break
        # att_map = cv2.resize(att_map, (W, H)) # H * W * 1
        x_axis_att_map = att_map_x.reshape(alpha_W) # W
        y_axis_att_map = att_map_y.reshape(alpha_H) # H

        key_coord_x = np.arange(0, W, W//alpha_W)
        key_coord_y = np.arange(0, H, H//alpha_H)
        coord_x = np.arange(0, W)
        coord_y = np.arange(0, H)
        x_axis_att_value = np.interp(coord_x, key_coord_x, x_axis_att_map)  # 100
        x_axis_att_value = x_axis_att_value * H
        y_axis_att_value = np.interp(coord_y, key_coord_y, y_axis_att_map)  # 100
        y_axis_att_value = y_axis_att_value * H


        x_att_pts = np.stack([coord_x, x_axis_att_value]).transpose((1, 0)).astype(np.int32)
        x_att_pts[:, 1] = H - x_att_pts[:, 1]

        y_att_pts = np.stack([y_axis_att_value, coord_y]).transpose((1, 0)).astype(np.int32)

        img_line = cv2.polylines(img_line, [x_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
        img_line = cv2.polylines(img_line, [y_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)

        # att_map = cv2.resize(att_map, (W, H))
        # _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])
        # _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)

        # mask_line_img = cv2.addWeighted(img_line, 0.5, _att_map, 2, 0)

        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), img_line)

    return True

def line_mask_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha_H, alpha_W = alpha.shape[2], alpha.shape[3]
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])
    np.save("softmax_att_alpha.npy", alpha)
    for i, att_map in enumerate(alpha):  # H * W * 1
        img_line = img.copy()
        if i >= len(pred):
            break
        # att_map = cv2.resize(att_map, (W, H)) # H * W * 1
        x_axis_att_map = np.sum(att_map, axis=0).reshape(alpha_W)  # W
        y_axis_att_map = np.sum(att_map, axis=1).reshape(alpha_H)  # H

        key_coord_x = np.arange(0, W, W // alpha_W)
        key_coord_y = np.arange(0, H, H // alpha_H)
        coord_x = np.arange(0, W)
        coord_y = np.arange(0, H)
        x_axis_att_value = np.interp(coord_x, key_coord_x, x_axis_att_map)  # 100
        x_axis_att_value = x_axis_att_value * H
        y_axis_att_value = np.interp(coord_y, key_coord_y, y_axis_att_map)  # 100
        y_axis_att_value = y_axis_att_value * H

        x_att_pts = np.stack([coord_x, x_axis_att_value]).transpose((1, 0)).astype(np.int32)
        x_att_pts[:, 1] = H - x_att_pts[:, 1]

        y_att_pts = np.stack([y_axis_att_value, coord_y]).transpose((1, 0)).astype(np.int32)

        img_line = cv2.polylines(img_line, [x_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)
        img_line = cv2.polylines(img_line, [y_att_pts.reshape((-1, 1, 2))], False, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])),
                    img_line)

    return True

def heatmap_visualize_bak(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])
    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break
        att_map = cv2.resize(att_map, (W, H))
        _att_map = np.zeros(dtype=np.uint8, shape=[H, W, 3])

        convas = img.copy()
        convas = cv2.rectangle(convas, (0, 0), (W, H), (255, 0, 0), -1)

        _att_map[:, :, -1] = (att_map * 255).astype(np.uint8)
        _att_map[:, :, 1] = ((1-att_map) * 255).astype(np.uint8)

        # show_attention = cv2.addWeighted(img, 0.5, _att_map, 2, 0)
        show_attention = img.copy()
        show_attention = cv2.addWeighted(convas, 0.5, show_attention, 0.5, 0)
        show_attention = cv2.addWeighted(_att_map, 0.5, show_attention, 0.5, 0)
        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), show_attention)

    return True

def heatmap_visualize(img, alpha, pred, vis_dir, img_path):
    assert len(img.shape) == 3
    H, W, _ = img.shape
    alpha = alpha.reshape([-1, alpha.shape[2], alpha.shape[3], 1])
    for i, att_map in enumerate(alpha):
        if i >= len(pred):
            break
        att_map = cv2.resize(att_map, (W, H))
        att_max = att_map.max()
        att_map /= att_max
        att_map *= 255
        att_map = att_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET)

        show_attention = img.copy()
        show_attention = cv2.addWeighted(heatmap, 0.5, show_attention, 0.5, 0)
        cv2.imwrite(os.path.join(vis_dir, "{}_{}_{}.jpg".format(os.path.basename(img_path).split('.')[0], i, pred[i])), show_attention)

    return True
