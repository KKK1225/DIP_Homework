import cv2
import numpy as np
import gradio as gr
import math

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def norm(vec):
    sum = 0.0
    sum = math.sqrt(vec[0]**2+vec[1]**2)
    return sum

def vert(arr):
    result = np.zeros((1,2))
    result[0][0] = -arr[0][1]
    result[0][1] = arr[0][0]
    return result

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    n=len(source_pts)
    image_new = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    # image_new = np.copy(image)
    """
    |           --> xlable
    \/   |-----------------------|
     y   |                       |  image.shape[0]
         |-----------------------|
             image.shape[1]
    """
    # print(image.shape, image[199,99])
    for i in range(image.shape[1]):
        for j in range(image.shape[0]):
            v = np.array([i,j])
            if not any((source_pts == v).all(axis=1)):
                w = np.array([], dtype=float)
                for m in range(n):
                    w = np.append(w, 1/norm(v - source_pts[m])**(2*alpha))
                px = np.zeros((2,))
                qx = np.zeros((2,))
                for m in range(n):
                    px += w[m] * source_pts[m]
                    qx += w[m] * target_pts[m]
                px /= sum(w)
                qx /= sum(w)
                p_hat =[]
                q_hat = []
                for m in range(n):
                    p_hat.append((source_pts[m] - px).reshape(1,2))
                    q_hat.append((target_pts[m] - qx).reshape(1,2))
                
                # source_pts[m] targe_pts[m] v px qx: (2,); p_hat[m] q_hat[m]: (1,2)
                v_px = (v - px).reshape(1,2)        # v - px: (1,2)

                # 2.1
                A = np.zeros((2,2))
                B = np.zeros((2,2))
                for m in range(n):
                    # print(p_hat[m].shape, p_hat[m][0])
                    A += w[m] * np.dot(vert(p_hat[m]).T, p_hat[m])
                    B += w[m] * np.dot(vert(p_hat[m]).T, q_hat[m])
                v_px = (v - px).reshape(1,2)
                target_v1 = np.dot(v_px, np.dot(np.linalg.pinv(A), B)).reshape(2,) + qx
                # print(target_v[0],target_v[1], i, j)
                if target_v1[0]<image.shape[1] and target_v1[1]<image.shape[0] and target_v1[0]>=0 and target_v1[1]>=0:
                    image_new[int(target_v1[1]), int(target_v1[0])] = image[j,i]


                # 2.3
                fr_v = np.zeros((1,2))
                for m in range(n):
                    A1 = np.vstack((p_hat[m], -vert(p_hat[m])))
                    A2 = np.vstack((v_px, -vert(v_px)))
                    fr_v += w[m] * np.dot(q_hat[m], np.dot(A1, A2.T))
                fr_v = fr_v.reshape(2,)
                target_v = norm(v-px) / norm(fr_v) * fr_v + qx
                if target_v[0]<image.shape[1] and target_v[1]<image.shape[0] and target_v[0]>=0 and target_v[1]>=0:
                    image_new[int(target_v[1]), int(target_v[0])] = image[j,i]

    """ 
    Return
    ------
        A deformed image.
    """
    
    for m in range(n):
        image_new[target_pts[m][1], target_pts[m][0]] = image[source_pts[m][1], source_pts[m][0]]

    warped_image = np.array(image_new)
    ### FILL: 基于MLS or RBF 实现 image warping

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
