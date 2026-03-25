# IMPORTS
import math
import pyrealsense2 as rs
from ultralytics import YOLO
import numpy as np

YOLO_MODEL_PATH = "models\\yolo8n_version3.pt"  # Path to your trained YOLO model weights

class Yolo_model:
    def __init__(self, conf_score=0.6):
        """Initializes the YOLO model with a confidence threshold.

            Parameters: conf_score (float): The minimum confidence score (between 0 and 1) required
                        to consider a detection valid. Default is 0.6."""
        self.conf_threshold = conf_score

    def load_model(self, model=YOLO_MODEL_PATH):
        """loads YOLO model
        
        Parameters: model (str): (yolo model e.g. 'my_yolo_model.pt')"""
        self.model = YOLO(model)

    def load_image(self, image, save_as="yolo_output.jpg"):
        """loads image input
        
        Parameters: image (str): (color image .jpg or .png),
                    
                    save_as (str): (name of the output image e.g. 'output.jpg')"""
        self.image_input = image
        self.save_output_as = save_as

    def save_result(self, save):
        self._results[0].save(filename=save)

    def request_labels(self):
        """Detects objects in the loaded image and returns their labels and center points.

        Returns: dict: key is the detection number and the value is a dictionary with 
        the object label and the (x, y) coordinates of the bounding box centroid.
            
        Example:
        {  0: { "label": "chemist", "centroid": (320, 240) },
           1: { "label": "chemspeed", "centroid": (150, 300) }  }"""

        self.label_info = {}
        label_names = []
        self._results = self.model(self.image_input)
        self._results[0].save(filename=self.save_output_as)

        for result in self._results:
            for nr in range(0, len(result.boxes)):
                result_cls = int(result.boxes[nr].cls)
                result_conf = float(result.boxes[nr].conf)
                result_xyxy = result.boxes[nr].xyxy

                if result_conf > self.conf_threshold:

                    {1:{"label":"Chemist", "centroid":(100, 300)}}
                    
                    result_name = self.model.names[result_cls]
                    label_names.append(result_name)
                
                    x, y = self.get_centroid(xyxy=result_xyxy)

                    self.label_info[nr] = {"label":result_name, "centroid":(x,y)}

        return self.label_info
    
    def get_centroid(self, xyxy):
        """Returns centroid of label in image"""

        x1, y1, x2, y2 = map(int, xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return (cx, cy)
    


class Live_Depth_Dist:

    def __init__(self, depth):
        """Initializes the class with live depth frame data.

        Parameters: depth: A RealSense depth frame object obtained from a live camera stream.
        Used to extract depth values and camera intrinsics for 3D calculations.
        
        Only use with depth data from Live streaming camera."""

        self.depth = depth
        self.intrinsics = self.depth.profile.as_video_stream_profile().get_intrinsics()

    def _get_depth(self, point):
        """Returns depth of a point"""
        x = point[0]
        y = point[1]

        return self.depth.get_distance(int(x), int(y))

    def request_distance(self, point_1, point_2):
        """Calculates the 3D distance (in meters) between two points using depth data.

        Parameters: point_1 (tuple): The (x, y) pixel coordinates of the first point.
        
                    point_2 (tuple): The (x, y) pixel coordinates of the second point.

        Returns: float: The Euclidean distance in meters between the two points.
        """
        self.point = point_1
        depth_1 = self._get_depth(point=point_1)

        self.point = point_2
        depth_2 = self._get_depth(point=point_2)

        point1 = rs.rs2_deproject_pixel_to_point(self.intrinsics, point_1, depth_1)
        point2 = rs.rs2_deproject_pixel_to_point(self.intrinsics, point_2, depth_2)

        # Same distance formula
        euclidean_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))

        return euclidean_distance


class File_Depth_Dist:
    INTRINSICS_DICT= {
    "width": 640,
    "height": 480,
    "fx": 383.4836730957031,
    "fy": 383.4836730957031,
    "ppx": 317.34674072265625,
    "ppy": 239.41297912597656,
    "model": "distortion.brown_conrady",
    "coeffs": [0.0,0.0,0.0,0.0,0.0]
    }
    
    def __init__(self, depth_image, depth_scale):
        """Use with recorded depth data

        Parameters: depth_image (npy file),
        
                    depth_scale (txt file)"""
        self.depth_image = np.load(depth_image)
        self._load_depth_scale(scale=depth_scale)
        self._load_intrinsics()
    
    def _load_depth_scale(self, scale):
        with open(scale, "r") as f:
            depth_scale = float(f.read().strip())
        self.depth_scale = depth_scale

    def _load_intrinsics(self):
        self.intrinsics = rs.intrinsics()
        self.intrinsics.width = self.INTRINSICS_DICT['width']
        self.intrinsics.height = self.INTRINSICS_DICT['height']
        self.intrinsics.fx = self.INTRINSICS_DICT['fx']
        self.intrinsics.fy = self.INTRINSICS_DICT['fy']
        self.intrinsics.ppx = self.INTRINSICS_DICT['ppx']
        self.intrinsics.ppy = self.INTRINSICS_DICT['ppy']
        self.intrinsics.model = getattr(rs.distortion, self.INTRINSICS_DICT['model'].split('.')[-1])  # e.g. 'distortion.brown_conrady'
        self.intrinsics.coeffs = self.INTRINSICS_DICT['coeffs']

    def _get_depth(self, point):
        x, y = int(point[0]), int(point[1])
        raw_depth = self.depth_image[y, x]
        return raw_depth * self.depth_scale  # convert to meters

    def request_distance(self, point_1, point_2):
        """Calculates the 3D distance (in meters) between two points using depth data.

        Parameters: point_1 (tuple): The (x, y) pixel coordinates of the first point.
        
                    point_2 (tuple): The (x, y) pixel coordinates of the second point.

        Returns: float: The Euclidean distance in meters between the two points.
        """
        depth_1 = self._get_depth(point=point_1)
        depth_2 = self._get_depth(point=point_2)

        point1 = rs.rs2_deproject_pixel_to_point(self.intrinsics, point_1, depth_1)
        point2 = rs.rs2_deproject_pixel_to_point(self.intrinsics, point_2, depth_2)

        euclidean_distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(point1, point2)]))
        return euclidean_distance
