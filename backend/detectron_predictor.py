import torch
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import ImageEnhance
import pandas as pd
from shapely import Polygon, unary_union, minimum_rotated_rectangle

import cv2
from skimage import exposure, filters, morphology
from PIL import Image
import re
import io
import base64

import numpy as np

class DetectronPredictor:
    def __init__(self):
        ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
        CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
        MAX_ITER = 3000
        EVAL_PERIOD = 200
        BASE_LR = 0.001
        NUM_CLASSES = 1

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
        cfg.MODEL.WEIGHTS = "../models/detectron.pth"
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
        cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.INPUT.MASK_FORMAT='bitmask'
        cfg.SOLVER.BASE_LR = BASE_LR
        cfg.SOLVER.MAX_ITER = MAX_ITER
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
        cfg.OUTPUT_DIR = "RODS/output_dir"

        cfg.MODEL.DEVICE='cpu'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        cfg.TEST.DETECTIONS_PER_IMAGE = 500
        self.predictor = DefaultPredictor(cfg)
        self.SIZE = (800, 800)
        self.MEDIAN = 100
        self.HEIGHT_THRESHOLD = 180


    def height_from_image(self, image):
        image = np.array(image).astype(np.float64)
        heights = exposure.rescale_intensity(image, out_range=(0, 1)).astype(np.float64)
        heights = 0.5 - 0.1 * np.log(1 / heights - 1.0)
        heights = np.nan_to_num(heights, nan=0, posinf=1, neginf=0)
        heights = np.clip(heights, 0, 1)
        heights = filters.gaussian(heights, 2)
        heights = exposure.rescale_intensity(heights, out_range=(0, 255)).astype(np.uint8)
        heights = cv2.GaussianBlur(heights, (3, 3), 2)
        heights = cv2.dilate(heights, morphology.disk(5))
        return heights

    def visualise(self, imdata, outputs):
        visualizer = Visualizer(
            imdata[:, :, ::-1],
            metadata={},
            instance_mode=ColorMode.SEGMENTATION,
        )

        out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]


    def predict_image_arr(self, imdata):
        outputs = self.predictor(imdata)
        return outputs
    
    def get_image(self, data, params):
        data = data.convert("RGB")
        data = ImageEnhance.Brightness(data).enhance(params["brightness"])
        data = ImageEnhance.Contrast(data).enhance(params["contrast"])
        data = np.array(data)
        return data
    
    def get_heights(self, outputs, factor):
        instances = outputs['instances']
        masks = instances.pred_masks.numpy().astype(np.uint8)
        scores_ = instances.scores.numpy()

        lengths = []
        widths = []
        areas = []
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        scores = []

        for mask, score in zip(masks, scores_):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            full_poly = []

            for contour in contours:
                contour = np.squeeze(contour)
                if contour.shape[0] > 4:
                    full_poly.append(Polygon(contour))

            poly = unary_union(full_poly)
            xmin, ymin, xmax, ymax = poly.bounds
            w = xmax - xmin
            h = ymax - ymin

            w, h = min(w, h), max(w, h)

            widths.append(w)
            lengths.append(h)
            areas.append(poly.area)
            scores.append(score)
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

        return widths, lengths, areas, score, xmins, xmaxs, ymins, ymaxs

    def get_angles(self, height_image, outputs):
        instances = outputs["instances"]
        masks = instances.pred_masks.numpy().astype(np.uint8)

        angles = []
        height_image = cv2.cvtColor(height_image, cv2.COLOR_RGB2GRAY)

        for mask in masks:
            image = mask * height_image
            angles.append(90 if np.max(image) > self.HEIGHT_THRESHOLD else 0)
        
        return angles


    def image_to_url(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(stream.getvalue()).decode()}"

    def make_table(self, widths, heights, areas, scores, xmins, xmaxs, ymins, ymaxs):
        table_data = {
            "Rod Lengths": heights,
            "Rod Widths": widths,
            "Areas": areas,
            "Scores": scores,
            "xmin": xmins,
            "xmax": xmaxs,
            "ymin": ymins,
            "ymax": ymaxs
        }

        stream = io.StringIO()
        table = pd.DataFrame(table_data)
        table.to_csv(stream)
        return stream.getvalue()


    def predict(self, data, factor, params):
        imgarr = self.get_image(data, params)
        print(type(imgarr), imgarr.shape)
        outputs = self.predict_image_arr(imgarr)
        heightimg = self.height_from_image(imgarr)

        visimg = self.visualise(imgarr, outputs)
        widths, lengths, areas, scores, xmins, xmaxs, ymins, ymaxs = self.get_heights(outputs, factor)
        table = self.make_table(widths, lengths, areas, scores, xmins, xmaxs, ymins, ymaxs)

        print("Total lengths:", len(lengths))

        return {
            "predicted_images": [self.image_to_url(visimg)],
            "depth_map": self.image_to_url(heightimg),
            "heights": lengths,
            "widths": widths,
            "areas": areas,
            "scores": scores,
            "avg_length": np.mean(lengths),
            "avg_width": np.mean(widths),
            "detections": len(outputs["instances"].pred_masks),
            "angles": self.get_angles(heightimg, outputs),
            "table": table
        }

    def process_txt(self, mag, ut):
        factor = 1
        match ut:
            case "nm":
                factor = 1
            case "um":
                factor = 1e3
            case "mm":
                factor = 1e6
            case "cm":
                factor = 1e7
        mag *= factor
        return 1
            
    def work(self, txt_contents, img_bytes, params):
        stream = io.BytesIO()
        stream.write(img_bytes)
        imgarr = Image.open(stream)

        factor = self.process_txt(params["scale"], params["unit"])
        predictions = self.predict(imgarr, factor, params)

        predictions["original"] = self.image_to_url(imgarr)
        return predictions
