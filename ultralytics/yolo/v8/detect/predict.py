# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
import torch
import re
import time
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops

global flag
class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        """Convert an image to PyTorch tensor and normalize pixel values."""
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            with open(r'./fourinex.txt','a+') as f:
                # 遍历每个预测框，将其坐标信息写入txt文件中
                for box in pred:
                    x1, y1, x2, y2, conf, cls = box.tolist()
                    # print(cls)
                    x1=int(x1)
                    y1=int(y1)
                    x2=int(x2)
                    y2=int(y2)
                    f.write(f"{flag},{x1},{y1},{x2},{y2}\n")
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results


def predict(source,cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    # model = cfg.model or 'yolov8n.pt'
    model='best.pt'
    # source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
    #     else 'https://ultralytics.com/images/bus.jpg'
    # source='rail_7.jpg'
    start_time = time.time()
    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
    end_time = time.time()
    with open("timerecodes.txt","a+")as f:
        f.write(str(end_time-start_time)+'\n')

if __name__ == '__main__':
    folder_path = r'./Railsurfaceimages'  # 图片文件夹路径
    file_names = os.listdir(folder_path)
    for file_name in file_names:
        print(file_name)
        result=re.findall(r"\d+",file_name)
        flag=int(result[0])
        file_name=folder_path+'\\'+file_name
        predict(source=file_name)

    # with open(r'D:\CODES\ultralytics-main\ultralytics-main\ultralytics\yolo\v8\detect\fourinex.txt','a+') as f:
    #     f.write(f"{flag},")
    # predict(source="rail_39.jpg")
