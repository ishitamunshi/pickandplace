from robodk import robolink  # RoboDK API
from robodk import robomath
import cv2
import numpy as np
import torch
from robodk import *  # RoboDK API
from robolink import *  # Robot toolbox
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import save_one_box  # Robot toolbox

global img
z = []


# RDK = RoboDK.RoboDK()


class DetectionPredictor(BasePredictor):

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        z=[]
        xscale = 1660 / 640
        yscale = 1250 / 480
        ix = 150
        iy = 550
        px = 379
        py = 251
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
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
            det_img = results[-1].plot()
            for res in results:
                boxes = res.boxes  # get the predicted boxes
                for box in boxes:
                    # print(box.xywh)
                    x = int(box.xywh[0][0])
                    y = int(box.xywh[0][1])
                    w = int(box.xywh[0][2])
                    h = int(box.xywh[0][3])
                    i = int(box.cls)
                    c = str(self.model.names[i])
                    xcoord = ix + ((x - px) * xscale)
                    ycoord = iy + ((py - y) * yscale) - 33
                    a = (c, xcoord, ycoord)
                    z.append(a)
                    cv2.putText(img=det_img, text=str(a), org=(x, y), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.25,
                                color=(0, 0, 0), thickness=1)

                    cv2.circle(det_img, (x, y), 20, (0, 0, 0), 4)

            # cv2.circle(det_img, (379,251), 20,(0,0,0),4)
            cv2.imshow('Processed Image', det_img)
            cv2.waitKey(10000)
            sorted_z = sorted(z, key=lambda x: x[2])
            print(sorted_z)
            l = sorted_z[0][1]
            b = sorted_z[0][2]
            print(str(l))
            q = int(l)
            s = int(b - 95)
            m = int(385)
            pose = Mat([[1.000000, 0.000000, 0.000000, q],
                        [-0.000000, -0.000000, 1.000000, s],
                        [0.000000, -1.000000, -0.000000, m],
                        [0.000000, 0.000000, 0.000000, 1.000000]])

            joint_angles = robot.SolveIK(pose)
            print(joint_angles)
            robot.MoveJ(joint_angles)

        return results


def predict(cfg=DEFAULT_CFG, use_python=False):
    global img
    model = "C:\\Users\\n\\Downloads\\weight2.pt"  # Link your weight file ie '.pt' file
    source = img
    # else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


# Forward and backwards compatible use of the RoboDK API:
# Remove these 2 lines to follow python programming guidelines
# Link to RoboDK
# RDK = Robolink()
RDK = robolink.Robolink()
cam_item = RDK.Item('camera', robolink.ITEM_TYPE_CAMERA)
robot = RDK.Item('UR10e')
# Get the camera object by name
cam_item.setParam('Open', 1)
while cam_item.setParam('isOpen') == '1':

    # ----------------------------------
    # Method 1: Get the camera image, by socket
    img_socket = None
    bytes_img = RDK.Cam2D_Snapshot('', cam_item)
    if isinstance(bytes_img, bytes) and bytes_img != b'':
        nparr = np.frombuffer(bytes_img, np.uint8)
        img_socket = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_socket is None:
        break
    cv2.imshow("img", img_socket)
    img = img_socket
    predict()
    cv2.waitKey(10000)
    key = cv2.waitKey(1)
    if key == 27:
        break  # User pressed ESC, exit
    if cv2.getWindowProperty("img", cv2.WND_PROP_VISIBLE) < 1:
        break  # User killed the main window, exit
cv2.destroyAllWindows()
RDK.Cam2D_Close(cam_item)