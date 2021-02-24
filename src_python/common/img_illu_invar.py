import cv2
import time
import numpy as np
from utils import vap_imshow_set, vap_impop_up, vap_imwrite, vap_imshow, vap_imsave
from scipy import fftpack, signal
from sklearn.cluster import MeanShift, estimate_bandwidth

""" 1-D SHADOW-FREE IMAGES: TEST 1(BASED ON LOGRG, LOGBG)
def compute_log_chromaticity(img):
    const = 1e-10
    img = img.astype(np.float)
    log_rg = np.log((img[:, :, 2] / (img[:, :, 1] + const)) + const)
    log_bg = np.log((img[:, :, 0] / (img[:, :, 1] + const)) + const)

    return log_rg, log_bg

def compute_gray_img(img, log_rg, log_bg, alpha):
    h, w, c = img.shape
    gray_img = np.zeros((h, w))

    a1 = np.tan(np.pi * alpha / 180)

    if a1 == 0:
        gray_img[:, :] = log_rg[:, :]
    else:
        a2 = np.full((h, w), -1/a1)
        b2 = log_bg - a2*log_rg

        x = -b2/(a2-a1)
        y = a2*x + b2

        gray_img[:, :] = np.sqrt(x[:, :] ** 2 + y[:, :] ** 2)
        # gray_img[:, :] = x[:, :]
    return gray_img
"""


def write_img_with_intensity(img, title, img_path, output_name):
    vap_imshow_set(1, 1, 1, img, title)
    vap_imsave(img_path, output_name)


def write_img_with_intensity_3(img1, title1, img2, title2,
                               img3, title3, img_path, output_name):
    vap_imshow_set(2, 2, 1, img1, title1)
    vap_imshow_set(2, 2, 2, img2, title2)
    vap_imshow_set(2, 2, 3, img3, title3)
    vap_imsave(img_path, output_name)


def normalize(img):
    min_val = np.amin(img)
    max_val = np.amax(img)

    img = (img - min_val) / ((max_val - min_val) / 255)
    img = img.astype(np.uint8)
    return img


def compute_gradient(img, mode='constant', cval=0):
    # ==============================================
    # im_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3, scale=1/16)
    # im_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3, scale=1/16)

    im_x, im_y = np.gradient(img)

    return im_x, im_y

# **************************************************************************************************
#                                                                                                  *
#                                       1D IMAGE PROCESSING                                        *
#                                                                                                  *
# **************************************************************************************************


def _1d_compute_log_chromaticity(img):
    # ==============================================
    # === configuration parameter
    H, W, _ = img.shape
    const = 1

    # ==============================================
    # === replace pixel 0 => const
    img = np.where(img == 0, const, img).astype(np.float)

    # === compute log-colour
    geo_mean = np.cbrt(img[:, :, 0] * img[:, :, 1] * img[:, :, 2])

    log_b = np.log(img[:, :, 0] / geo_mean)
    log_g = np.log(img[:, :, 1] / geo_mean)
    log_r = np.log(img[:, :, 2] / geo_mean)

    log_b = np.expand_dims(log_b, -1)
    log_g = np.expand_dims(log_g, -1)
    log_r = np.expand_dims(log_r, -1)
    log_bgr = np.reshape(np.concatenate((log_b, log_g, log_r), -1), [H, W, 3, 1])

    # === orthogonal matrix
    #            = [[1/sqrt(2), -1/sqrt(2), 0         ]
    #               [1/sqrt(6),  1/sqrt(6), -2/sqrt(6)]]
    orth_mat = np.array([[1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                         [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)]])
    x = np.einsum('ij,abjk->abik', orth_mat, log_bgr)

    return x


def _1d_compute_gray_img(x, alpha):
    # === convert alpha angle degrees to radians
    alpha = np.pi * alpha / 180

    gray_img = x[:, :, 0, 0] * np.cos(alpha) + x[:, :, 1, 0] * np.sin(alpha)

    return gray_img


def derive_1D_shadow_free_image(img, img_shw_step=False):
    # ==============================================
    # === configuration parameter
    h, w, _ = img.shape

    entropys = []
    theta = 0

    # ==============================================
    # === compute log-chromaticity
    # log_rg, log_bg = compute_log_chromaticity(img)
    x = _1d_compute_log_chromaticity(img)

    for alpha in range(180):
        # === step 1: Obtain a grayscale image projecting log-chromaticity pixel values of alpha
        # gray_img = compute_gray_img(img, log_rg, log_bg, alpha)
        gray_img = _1d_compute_gray_img(x, alpha)

        # === step 2: reject the ouliers in gray image according to Chebyshev’s theorem
        # get parameter for compute confidence interval [mean-k*std, mean+k*std]
        # for 90% confidence interval 1 - 1/k**2 = 0.9 => k = 3.16
        gray_img = np.reshape(gray_img, (h * w))
        k = 3.16
        mean_val = np.mean(gray_img)
        std_val = np.std(gray_img)
        lower_bound = mean_val - k * std_val
        upper_bound = mean_val + k * std_val
        gray_img = np.where((gray_img >= lower_bound) & (gray_img <= upper_bound), gray_img, lower_bound - 1)
        # print("mean_val: ", mean_val)
        # print("std_val: ", std_val)
        # print("lower_bound: ", lower_bound)
        # print("upper_bound: ", upper_bound)

        # === step 3: get middle 90% range of nonoutlier pixels
        gray_img = np.sort(gray_img, -1)
        lower_idx = 0
        upper_idx = h * w
        for i in range(0, h * w):
            if (gray_img[i] != lower_bound - 1):
                break
            lower_idx = i
        dist = (int)((upper_idx - lower_idx) * 5 / 100)
        lower_idx += dist
        upper_idx -= dist
        gray_img = gray_img[lower_idx:upper_idx]
        # print("lower_idx: ", lower_idx)
        # print("upper_idx: ", upper_idx)

        # === step 4: plot image gray to histogram
        std_val = np.std(gray_img)
        n = gray_img.shape[0]
        bin_width = 3.5 * ((n)**(-1 / 3)) * std_val
        curr = gray_img[0]
        bins = []
        while curr < gray_img[n - 1] + bin_width:
            bins.append(curr)
            curr = curr + bin_width
        hist, bins = np.histogram(gray_img, bins)

        # === step 5: Compute the entropy
        total_hist = np.sum(hist)
        hist = np.where(hist == 0, 1, hist)
        entropy = np.where(hist > 0, (hist / total_hist) * np.log(hist / total_hist), 0)
        entropys.append(-np.sum(entropy))

    # === step 6: Get theta is min entropys
    theta = np.argmin(entropys)
    # out_img = compute_gray_img(img, log_rg, log_bg, theta)
    out_img = _1d_compute_gray_img(x, theta)
    out_img = normalize(out_img)

    return out_img, theta, x


# **************************************************************************************************
#                                                                                                  *
#                                       2D IMAGE PROCESSING                                        *
#                                                                                                  *
# **************************************************************************************************

def _2d_compute_lighting_add_back(img, x, x_theta):
    H, W, _ = img.shape
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === find min of 1% brightest pixels
    bright_arr = np.reshape(gray_img, (-1))
    bright_arr = np.sort(bright_arr)
    index = H * W - int(H * W * 0.01)
    thresh_value = bright_arr[index]

    extra_1 = np.where(gray_img > thresh_value, abs(x[:, :, 0, 0] - x_theta[:, :, 0, 0]), 0).reshape((-1))
    extra_2 = np.where(gray_img > thresh_value, abs(x[:, :, 1, 0] - x_theta[:, :, 1, 0]), 0).reshape((-1))
    return np.mean(extra_1[extra_1 > 0]), np.mean(extra_2[extra_2 > 0])


def derive_2D_shadow_free_image(img, theta, x, img_path, img_shw_step=False):
    # ==============================================
    # === configuration parameter
    H, W, _ = img.shape

    # ==============================================
    # === compute projector matrix 2x2
    theta = np.pi * theta / 180  # convert theta angle degrees to radians
    p_mat = np.array([[np.cos(theta)], [np.sin(theta)]])
    p_mat = np.dot(p_mat, np.transpose(p_mat))

    # === compute invariant log-chromaticity
    x_theta = np.einsum('ij,abjk->abik', p_mat, x)
    extra_1, extra_2 = _2d_compute_lighting_add_back(img, x, x_theta)
    x_theta[:, :, 0, 0] = x_theta[:, :, 0, 0] + extra_1
    x_theta[:, :, 1, 0] = x_theta[:, :, 1, 0] + extra_2

    # === orthogonal matrix
    #            = [[1/sqrt(2), -1/sqrt(2), 0         ]
    #               [1/sqrt(6),  1/sqrt(6), -2/sqrt(6)]]
    orth_matrix = np.array([[1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                            [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6)]])
    chro_est = np.exp(np.einsum('ij,abjk->abik', np.transpose(orth_matrix), x_theta)).reshape(H, W, 3)

    total_bright = np.sum(chro_est, -1)
    out_img = np.zeros([H, W, 3])
    out_img[:, :, 0] = chro_est[:, :, 0] / total_bright
    out_img[:, :, 1] = chro_est[:, :, 1] / total_bright
    out_img[:, :, 2] = chro_est[:, :, 2] / total_bright
    out_img = normalize(out_img)

    if img_shw_step:
        vap_imwrite(img_path + "/ill_2DShadowFreeImg.jpg", out_img)

    return x_theta


# **************************************************************************************************
#                                                                                                  *
#                                       3D IMAGE PROCESSING                                        *
#                                                                                                  *
# **************************************************************************************************

def _3d_shadow_detect_smooth_filter_by_gaussian(img):
    # ==============================================
    # === configuration parameter
    k_size = 5
    sigma_x = 1
    sigma_y = 1

    # ==============================================
    img = cv2.GaussianBlur(img, (k_size, k_size), sigmaX=sigma_x, sigmaY=sigma_y)

    return img


def _3d_shadow_detect_smooth_filter_by_mean_shift(img, img_path, img_shw_step=False):
    # ==============================================
    # === configuration parameter
    H, W = img.shape

    # ==============================================
    # === mean-shift
    flatImg = np.reshape(img, [-1, 1])
    print(H, W)
    print(flatImg.shape)
    print(img.shape)

    bandwidth = estimate_bandwidth(flatImg, quantile=0.2, n_samples=1000)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    ms.fit(flatImg)

    labels = ms.labels_

    print(labels.shape)
    print(labels)

    segmentedImg = np.reshape(labels, (H, W))

    segmentedImg = normalize(segmentedImg)
    vap_imshow_set(1, 1, 1, segmentedImg, "log Blue")
    vap_imsave(img_path, ['ill_3DSegmentedImg'])

    return img


def _3d_shadow_detect_smooth_filter(img, img_path, img_shw_step):
    # ==============================================
    # === configuration parameter
    opt = 0

    # ==============================================
    if (opt == 0):  # filter by gaussian
        img = _3d_shadow_detect_smooth_filter_by_gaussian(img)
    elif (opt == 1):
        img = _3d_shadow_detect_smooth_filter_by_mean_shift(img, img_path, img_shw_step)

    return img


def _3d_shadow_detect_compute_edge_by_canny_cv(img, h_ratio, l_ratio):
    # ==============================================
    # get threshold by otsu
    ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_threshold = ret * h_ratio
    low_threshold = ret * l_ratio
    print(low_threshold, high_threshold)

    # canny
    img = cv2.Canny(img, low_threshold, high_threshold)
    return img


def _3d_shadow_detect_compute_edge_by_canny_self(img):

    return img


def _3d_shadow_detect_smooth_filter_by_susan(img):
    return img

def _3d_shadow_detect_compute_edge(img, h_ratio, l_ratio):
    # ==============================================
    # === configuration parameter
    opt = 0

    # ==============================================
    if (opt == 0):  # compute edge by cv.canny
        img = _3d_shadow_detect_compute_edge_by_canny_cv(img, h_ratio, l_ratio)
    elif (opt == 1): # compute edge by canny self
        img = _3d_shadow_detect_compute_edge_by_canny_self(img)
    elif (opt == 2): # compute edge by susan
        img = _3d_shadow_detect_smooth_filter_by_susan(img)

    return img


def _3d_shadow_detect(origin_img, x_theta, img_path, img_shw_step=False):
    # ==============================================
    # === configuration parameter
    rows, cols, _ = origin_img.shape

    # ==============================================
    origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

    # === smooth filter
    origin_img = _3d_shadow_detect_smooth_filter(origin_img, img_path, img_shw_step)
    x_theta[:, :, 0, 0] = _3d_shadow_detect_smooth_filter(x_theta[:, :, 0, 0], img_path, img_shw_step)
    x_theta[:, :, 1, 0] = _3d_shadow_detect_smooth_filter(x_theta[:, :, 1, 0], img_path, img_shw_step)
    x_theta = normalize(x_theta)

    # === canny edge
    edge_origin_img = _3d_shadow_detect_compute_edge(origin_img, 1, 0.5)
    edge_x_theta_1_img = _3d_shadow_detect_compute_edge(x_theta[:, :, 0, 0], 0.5, 0.25)
    edge_x_theta_2_img = _3d_shadow_detect_compute_edge(x_theta[:, :, 1, 0], 0.5, 0.25)

    edge_x_theta = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (edge_x_theta_1_img[i, j] > edge_x_theta_2_img[i, j]):
                edge_x_theta[i, j] = edge_x_theta_1_img[i, j]
            else:
                edge_x_theta[i, j] = edge_x_theta_2_img[i, j]
    ''' get edge image
    '''
    edge_img = np.zeros((rows, cols), np.uint8)
    # x=[-1, -1, -1, 0, 0, 0, 1, 1, 1]
    # y=[-1, 0, 1, -1, 0, 1, -1, 0, 1]
    k_size = 7
    x = np.mgrid[0:k_size, 0:k_size][0].reshape(k_size * k_size) - (int)(k_size / 2)
    y = np.mgrid[0:k_size, 0:k_size][1].reshape(k_size * k_size) - (int)(k_size / 2)
    num_point = k_size * k_size
    for i in range(rows):
        for j in range(cols):
            if (edge_origin_img[i, j] == 255):
                count = 0
                for t in range(num_point):
                    ii = i + x[t]
                    jj = j + y[t]
                    if (ii > 0 and ii < rows and jj > 0 and jj < cols
                            and edge_x_theta[ii, jj] == 0):
                        count += 1
                if (count == k_size * k_size):
                    edge_img[i, j] = 255

    # edge_img = cv2.imread("/home/thanghv7/Downloads/edge_map_binary.jpg", 0)
    kernel = np.ones((3, 3), np.uint8)
    edge_img = cv2.dilate(edge_img, kernel)
    edge_img = cv2.dilate(edge_img, kernel)

    if img_shw_step:
        write_img_with_intensity(edge_origin_img, "edge origin image", img_path, ['ill_3DEdgeOriginImage'])
        write_img_with_intensity(edge_x_theta, "edge 2d image", img_path, ['ill_3DEdge2DImage'])
        vap_imwrite(img_path + "/ill_3DEdgeResultImage.jpg", edge_img)

    return edge_img


def _3d_threshold_gradient(grad_img):
    threshold_value = 1

    grad_img = np.where(grad_img < threshold_value, 0, grad_img)

    return grad_img


def _3d_threshold_shadow(grad_img, edge_img):
    grad_img = np.where(edge_img == 255, 0, grad_img)

    return grad_img


def _3d_compute_z(grad_x, grad_y, edge_img, a_x, a_y,
                  color, img_path, img_shw_step=False):
    # ==============================================
    # === configuration parameters
    rows, cols = grad_x.shape
    list_log = []
    t = 0

    # ==============================================
    # === step 1: threshold shadow
    ts_grad_x = _3d_threshold_shadow(grad_x, edge_img)
    ts_grad_y = _3d_threshold_shadow(grad_y, edge_img)

    if img_shw_step:
        title = "ill_3DLogGradientTS" + color
        write_img_with_intensity(np.sqrt(ts_grad_x**2 + ts_grad_y**2), "3DLogGradientThresholdShadow", img_path, [title])

    while True:
        # === step 2: update pixel

        ts_grad_x_new = np.zeros((rows, cols), np.float)
        ts_grad_y_new = np.zeros((rows, cols), np.float)

        ii = [-1, 0, 1, 0]
        jj = [0, -1, 0, 1]
        for i in range(rows):
            for j in range(cols):
                if (edge_img[i, j] == 255):
                    sum_grad_x = 0
                    sum_grad_y = 0
                    for k in range(4):
                        x = i + ii[k]
                        y = j + jj[k]
                        if (x >= 0 and x < rows and y >= 0 and y < cols):
                            sum_grad_x += ts_grad_x[x, y]
                            sum_grad_y += ts_grad_y[x, y]
                    ts_grad_x_new[i, j] = sum_grad_x
                    ts_grad_y_new[i, j] = sum_grad_y
                else:
                    ts_grad_x_new[i, j] = ts_grad_x[i, j]
                    ts_grad_y_new[i, j] = ts_grad_y[i, j]
        ts_grad_x_new = ts_grad_x
        ts_grad_y_new = ts_grad_y

        # === step 3: get Z
        # f_x = np.fft.fftshift(np.fft.fft2(ts_grad_x_new))
        # f_y = np.fft.fftshift(np.fft.fft2(ts_grad_y_new))
        f_x = np.fft.fft2(ts_grad_x_new)
        f_y = np.fft.fft2(ts_grad_y_new)

        Z = np.zeros((rows, cols), np.complex)
        for i in range(rows):
            for j in range(cols):
                if (i == 0 and j == 0):
                    Z[i, j] = 0j
                else:
                    ts = np.conjugate(a_x[i]) * f_x[i, j] + np.conjugate(a_y[j]) * f_y[i, j]
                    ms = np.abs(a_x[i]) ** 2 + np.abs(a_y[j]) ** 2
                    Z[i, j] = ts / ms

        # === step 4: check condition
        # ts_grad_x_new = np.fft.ifft2(np.fft.ifftshift(a_x[:, None] * Z)).real
        # ts_grad_y_new = np.fft.ifft2(np.fft.ifftshift(a_y[None, :] * Z)).real
        ts_grad_x_new = np.fft.ifft2(a_x[:, None] * Z).real
        ts_grad_y_new = np.fft.ifft2(a_y[None, :] * Z).real

        residual = np.sum(np.abs(ts_grad_x_new-ts_grad_x)) + np.sum(np.abs(ts_grad_y_new-ts_grad_y))
        residual /= (rows * cols)
        print('residual', residual)

        ts_grad_x = ts_grad_x_new
        ts_grad_y = ts_grad_y_new

        # log_complex = np.fft.ifft2(np.fft.ifftshift(Z))
        log_complex = np.fft.ifft2(Z)
        log_real = np.zeros((rows, cols), np.float)
        for i in range(rows):
            for j in range(cols):
                log_real[i, j] = log_complex[i, j].real
        list_log.append(log_real)

        t += 1
        if (t == 3):
            break

    return list_log


def _3d_fix_unknown_factor(img, ori_img):
    H, W = img.shape

    # === find 0.005% brightest pixels
    # img
    img_arr = np.reshape(img, (-1))
    img_arr = np.sort(img_arr)
    img_mean = 0
    for i in range(int(H * W * 0.01)):
        img_mean += img_arr[H*W - i - 1]
    img_mean /= int(H * W * 0.01)

    # ori_img
    ori_img_arr = np.reshape(ori_img, (-1))
    ori_img_arr = np.sort(ori_img_arr)
    ori_img_mean = 0
    for i in range(int(H * W * 0.01)):
        ori_img_mean += ori_img_arr[H*W - i - 1]
    ori_img_mean /= int(H * W * 0.01)

    ratio = ori_img_mean / img_mean
    img *= ratio
    return img


def _3d_recover_by_dft(origin_img, grad_bx, grad_by, grad_gx, grad_gy,
                       grad_rx, grad_ry, edge_img, img_path, img_shw_step):
    # ==============================================
    # === configuration parameters
    rows, cols, _ = origin_img.shape

    # ==============================================
    # === get coefficient a = e^(2piiu/N) - 1
    a_x = np.zeros(rows, np.complex)
    for i in range(0, rows):
        a_x[i] = np.cos(2 * np.pi * i / rows) + 1j * np.sin(2 * np.pi * i / rows) - 1

    a_y = np.zeros(cols, np.complex)
    for i in range(0, cols):
        a_y[i] = np.cos(2 * np.pi * i / cols) + 1j * np.sin(2 * np.pi * i / cols) - 1

    # === get z
    zb = _3d_compute_z(grad_bx, grad_by, edge_img, a_x, a_y, "Blue", img_path, img_shw_step)
    zg = _3d_compute_z(grad_gx, grad_gy, edge_img, a_x, a_y, "Green", img_path, img_shw_step)
    zr = _3d_compute_z(grad_rx, grad_ry, edge_img, a_x, a_y, "Red", img_path, img_shw_step)

    for i in range(len(zb)):

        if img_shw_step:
            write_img_with_intensity_3(zb[i], "blue", zg[i], "green", zr[i], "red", img_path, ['ill_3DLogResult0', str(i)])

        zbi = np.exp(zb[i])
        zgi = np.exp(zg[i])
        zri = np.exp(zr[i])

        zbi = _3d_fix_unknown_factor(zbi, origin_img[:, :, 0])
        zgi = _3d_fix_unknown_factor(zgi, origin_img[:, :, 1])
        zri = _3d_fix_unknown_factor(zri, origin_img[:, :, 2])

        output_img = np.concatenate((np.expand_dims(zbi, -1), np.expand_dims(zgi, -1), np.expand_dims(zri, -1)), -1)

        if img_shw_step:
            vap_imwrite(img_path + "/ill_3DShadowFreeImg_" + str(i) + ".jpg", output_img)

    return output_img


def derive_3D_shadow_free_image(origin_img, sha_2d_img, img_path, img_shw_step=False):
    # ==============================================
    # === configuration parameter

    # ==============================================
    # === step 1: get edge shadow img
    edge_img = _3d_shadow_detect(origin_img, sha_2d_img, img_path, img_shw_step)

    # === step 2: convert to log space
    # replace pixel 0 => const
    origin_img = np.where(origin_img == 0, 1, origin_img).astype(np.float)

    log_b = np.log(origin_img[:, :, 0])
    log_g = np.log(origin_img[:, :, 1])
    log_r = np.log(origin_img[:, :, 2])

    if img_shw_step:
        write_img_with_intensity_3(log_b, "blue", log_g, "green", log_r, "red", img_path, ['ill_3DLogOrigin0'])

    # === step 3: get gradient
    grad_bx, grad_by = compute_gradient(log_b)
    grad_gx, grad_gy = compute_gradient(log_g)
    grad_rx, grad_ry = compute_gradient(log_r)

    if img_shw_step:
        write_img_with_intensity(np.sqrt(grad_bx**2 + grad_by**2), "3DLogGradientBlue", img_path, ['ill_3DLogGradientBlue'])
        write_img_with_intensity(np.sqrt(grad_gx**2 + grad_gy**2), "3DLogGradientGreen", img_path, ['ill_3DLogGradientGreen'])
        write_img_with_intensity(np.sqrt(grad_rx**2 + grad_ry**2), "3DLogGradientRed", img_path, ['ill_3DLogGradientRed'])

    # === step 4: threshold gradient
    # grad_bx = _3d_threshold_gradient(grad_bx)
    # grad_by = _3d_threshold_gradient(grad_by)

    # grad_gx = _3d_threshold_gradient(grad_gx)
    # grad_gy = _3d_threshold_gradient(grad_gy)

    # grad_rx = _3d_threshold_gradient(grad_rx)
    # grad_ry = _3d_threshold_gradient(grad_ry)

    # === step 5: dft
    output_img = _3d_recover_by_dft(origin_img, grad_bx, grad_by, grad_gx, grad_gy,
                                    grad_rx, grad_ry, edge_img, img_path, img_shw_step)

    return output_img


def img_illuminant_invariance(img, img_path, img_shw_step=False):
    vap_imwrite(img_path + "/ill_0DOriginImg.jpg", img)

    # === derive 1D shadow free image
    start = time.time()
    in_img = np.array(img, copy=True)
    print("**********************************************************")
    print("===== derive 1D shadown free image")
    sha_1d_img, theta, x = derive_1D_shadow_free_image(in_img, img_shw_step)
    end = time.time()

    print("+ Angle: " + str(theta))
    if img_shw_step:
        vap_imwrite(img_path + "/ill_1DShadowFreeImg.jpg", sha_1d_img)
    print("+ Time of Comsumtion:", end - start, end="\n\n")

    # === derive 2D shadow free image
    start = time.time()
    print("**********************************************************")
    print("===== derive 2D shadown free image")
    x_theta = derive_2D_shadow_free_image(img, theta, x, img_path, img_shw_step)
    end = time.time()
    print("+ Time of Comsumtion:", end - start, end="\n\n")

    # === derive 3D shadow free image
    start = time.time()
    print("**********************************************************")
    print("===== derive 3D shadown free image")
    sha_3d_img = derive_3D_shadow_free_image(img, x_theta, img_path, img_shw_step)
    end = time.time()

    if img_shw_step:
        vap_imwrite(img_path + "/ill_3DShadowFreeImg.jpg", sha_3d_img)
    print("+ Time of Comsumtion:", end - start, end="\n\n")

    return sha_1d_img, theta
