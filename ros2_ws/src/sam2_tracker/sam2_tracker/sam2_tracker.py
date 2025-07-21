#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PolygonInstance, PolygonInstanceStamped, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
#from sam2.build_sam import build_sam2_camera_predictor
from efficient_track_anything.build_efficienttam import build_efficienttam_camera_predictor

class SAMTracker(Node):
    def __init__(self):
        super().__init__('sam_tracker')
        
        self.bridge = CvBridge()
        self.declare_parameter('output_quality', 75)
        
        input_topic = '/camera/image/compressed'
        output_topic = '/masks/compressed'
        self.output_quality = self.get_parameter('output_quality').value
        
        self.subscription = self.create_subscription(
            CompressedImage,
            input_topic,
            self.image_callback,
            10)
        
        self.publisher = self.create_publisher(
            CompressedImage,
            output_topic,
            10)
        
        self.poly_pub = self.create_publisher(
            PolygonInstanceStamped,
            'polygon',
            10)
        
        #sam2_checkpoint = "/home/user/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
        #model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        tam_checkpoint = "/home/user/REALTIME_SAM2/checkpoints/efficienttam_ti_512x512.pt"
        model_cfg = "./configs/efficienttam/efficienttam_ti_512x512.yaml"

        #self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.predictor = build_efficienttam_camera_predictor(model_cfg, tam_checkpoint, device=torch.device("cuda"))
        
        self.if_init = False

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)            
            processed_image, polygons = self.process_image(cv_image)            
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(
                processed_image,
                dst_format='jpg')  # or 'png' if you prefer lossless            
            compressed_msg.header = msg.header
            self.publisher.publish(compressed_msg)
            for poly in polygons:
                polygon_msg = PolygonInstanceStamped()
                polygon_msg.header = msg.header
                polygon_msg.polygon = poly
                self.poly_pub.publish(polygon_msg)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def process_image(self, image):
        #frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = image
        width, height = frame.shape[:2][::-1]
        polygons = []
        if not self.if_init:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.first_mask = self.predictor.load_first_frame(frame, 3)
            self.if_init = True
            #ann_frame_idx = 0
            cls = 1
            points = np.array([[595, 65], [480,150], [472,120], [425,170], [395,188]], dtype=np.float32)
            labels = np.array([1,0,0,1,1], dtype=np.int32)
            first_hit = np.array([True, False, False, False, False], dtype=np.bool_)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.predictor.add_new_points_during_track(cls, points, labels, first_hit=first_hit[0], frame=frame)
            #_, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            #    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels,
            #)
            cls = 2
            points = np.array([[350, 275]], dtype=np.float32)
            labels = np.array([1], np.int32)
            first_hit = np.array([False], dtype=np.bool_)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.predictor.add_new_points_during_track(cls, points, labels, first_hit=first_hit[0], frame=frame)
            #_, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            #    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels, clear_old_points=False
            #)
            cls = 3
            points = np.array([[298, 210]], dtype=np.float32)
            labels = np.array([1], np.int32)
            first_hit = np.array([False], dtype=np.bool_)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                self.predictor.add_new_points_during_track(cls, points, labels, first_hit=first_hit[0], frame=frame)
            #_, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            #    frame_idx=ann_frame_idx, obj_id=ann_obj_id, points=points, labels=labels, clear_old_points=False
            #)
            processed_image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, out_mask_logits = self.predictor.track(frame)
            processed_image = self.apply_mask_to_frame(frame, out_mask_logits)
            #all_mask = np.zeros((height, width, 3), dtype=np.uint8)
            #all_mask[..., 1] = 255
            #for i in range(0, len(out_obj_ids)):
            #    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
            #        np.uint8
            #    ) * 255
            #    contours, _ = cv2.findContours(out_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #    if len(contours) > 0:
            #        poly = PolygonInstance()
            #        poly.id = out_obj_ids[i]
            #        all_points = np.vstack([contour.squeeze() for contour in contours])
            #        hull = cv2.convexHull(all_points)
            #        for point in hull.squeeze():
            #            p = Point32()
            #            p.x = float(point[0])
            #            p.y = float(point[1])
            #            p.z = 0.0
            #            poly.polygon.points.append(p)
            #        polygons.append(poly)
    
            #    hue = (i + 3) / (len(out_obj_ids) + 3) * 255
            #    all_mask[out_mask[..., 0] == 255, 0] = hue
            #    all_mask[out_mask[..., 0] == 255, 2] = 255
            #processed_image = cv2.cvtColor(all_mask, cv2.COLOR_HSV2RGB)
        return processed_image, polygons

    def apply_mask_to_frame(self, frame, mask_logits, colors=None):
        if isinstance(mask_logits, torch.Tensor):
            mask_logits = mask_logits.cpu().numpy()

        if colors is None:
            colors = [
                (0, 255, 0),
                (0, 0, 255),
                (255, 0, 0),
                (0, 255, 255)
            ]
        
        mask_colored = np.zeros_like(frame, dtype=np.uint8)

        for class_idx in range(mask_logits.shape[0]):
            mask = (mask_logits[class_idx] > 0.0).astype(np.uint8) * 255

            for i in range(3):
                mask_colored[:, :, i] = np.clip(
                    mask_colored[:, :, i] + (mask * (colors[class_idx][i] / 255)).astype(np.uint8), 
                    0, 
                    255
                )

        return cv2.addWeighted(frame, 0.6, mask_colored, 0.4, 0)
        
def main(args=None):
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    #if torch.cuda.get_device_properties(0).major >= 8:
    #    torch.backends.cuda.matmul.allow_tf32 = True
    #    torch.backends.cudnn.allow_tf32 = True
    
    rclpy.init(args=args)
    sam_tracker = SAMTracker()
    rclpy.spin(sam_tracker)
    sam_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

