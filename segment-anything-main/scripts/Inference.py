import cv2  # type: ignore
import time
import timeit
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
#用于保存图片
global flag1
import subprocess

# 调用predict.py


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([0, 1, 0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    np.set_printoptions(threshold=np.inf)
    # print(mask)
    area = np.sum(mask)
    # print("面积:"+str(area))
    with open("predictarea.txt","a+")as f:
        f.write(str(area)+"\n")
    # ax.imshow(mask_image)
# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels == 1]
#     neg_points = coords[labels == 0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
#                linewidth=1.25)
# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

'''
这部分是展示原图
'''
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# plt.axis('on')
# plt.show()
'''
这部分是展示原图
'''

def multi_box_prompt(list):
    input_boxes = torch.tensor([
        list
    ], device=predictor.device)
    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    # 将 PyTorch 张量转换为 numpy 数组
    tensor = masks.numpy()
    # 将 True 和 False 值分别转换为 255 和 0
    array = np.where(tensor == True, 255, 0)
    # 可视化 numpy 数组
    image_name=f"{flag1}.jpg"
    plt.imsave(f'images/{image_name}', array[0, 0], cmap='gray')
    plt.imshow(array[0, 0], cmap='gray')
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # for mask in masks:
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=False)
    # # for box in input_boxes:
    # #     show_box(box.cpu().numpy(), plt.gca())
    # plt.axis('off')
    # plt.savefig("result.png")
    # plt.show()
def conmulateTime():
    start_time = time.time()
    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间为：", run_time, "秒")

if __name__=="__main__":
    currentDir=os.path.abspath("..")
    sam_checkpoint=os.path.join(currentDir,"checkpoint","sam_vit_l_0b3195.pth")
    lastIndex=-1
    nowIndex=0
    isRun=0
    final_list = []
    with open(r'D:\CODES2\KH-SAM\ultralytics\yolo\v8\detect\fourinex.txt', 'r') as f:
        for line in f:
            file_path = r"D:\CODES2\KH-SAM\ultralytics\yolo\v8\detect\Railsurfaceimages\{}.jpg"
            flag1=int(line.split(',')[0])
            nowIndex=flag1
            if nowIndex!=lastIndex and lastIndex!=-1:
                isRun=1
            print("flag1的值为："+str(flag1))
            file_path = file_path.format(line.split(',')[0])
            # image = Image.open(file_path)
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # sys.path.append("..")
            model_type = "vit_l"
            device = "cpu"  # or  "cuda"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=device)
            predictor = SamPredictor(sam)
            list = []
            line = line.replace("\n", "")
            for x in line.split(",")[1:]:
                list.append(int(x))
                print(x)
            final_list.append(list)
            if isRun==1:
                start_time = time.time()
                multi_box_prompt(list)
                end_time = time.time()
                run_time = end_time - start_time
                with open("predicttime.txt", "a+") as f:
                    f.write(str(run_time) + "\n")
                isRun=0
                final_list.clear()
            lastIndex=nowIndex
            # count += 1



