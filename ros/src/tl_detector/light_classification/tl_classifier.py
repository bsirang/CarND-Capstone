from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2

class TLClassifier(object):
    """
    A class for classifying traffic lights.

    This class uses the Mobilenet COCO SSD pre-trained model for object detection
    and classification. This model contains 90 different object classifications,
    one of which are traffic lights.

    The model is used to detect the traffic lights and their bounding boxes.
    After which a heuristic color analysis is used to identify the traffic light
    color. This approach was chosen for its simplicity.

    """
    def __init__(self):
        self.graph_file = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        self.graph = self.load_graph(self.graph_file)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')

        self.sess = tf.Session(graph=self.graph)

        # Classification index for traffic lights from SSD Mobilenet V1 COCO
        self.traffic_light_class = 10

    @staticmethod
    def color_detection(image):
        color_strength_threshold = 10
        R = image[:,:,0].flatten()
        G = image[:,:,1].flatten()
        B = image[:,:,2].flatten()
        R_avg = np.sum(R) / len(R)
        G_avg = np.sum(G) / len(G)
        B_avg = np.sum(B) / len(B)
        #print("{} {} {}".format(R_avg, G_avg, B_avg))
        if R_avg - G_avg > color_strength_threshold:
            return TrafficLight.RED
        elif G_avg - R_avg > color_strength_threshold:
            return TrafficLight.GREEN
        else:
            return TrafficLight.YELLOW

    @staticmethod
    def crop_image(image, box_coord):
        top, left, bot, right = box_coord[...]
        bot = int(bot)
        top = int(top)
        left = int(left)
        right = int(right)
        return image[top:bot, left:right]

    @staticmethod
    def filter_boxes(min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    @staticmethod
    def filter_class(filter_class, boxes, scores, classes):
        n = len(classes)
        idxs = []
        for i in range(n):
            if classes[i] >= filter_class:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    @staticmethod
    def to_image_coords(boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    @staticmethod
    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                        feed_dict={self.image_tensor: image.reshape((1,) + image.shape)})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than `confidence_cutoff`
        confidence_cutoff = 0.2
        boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        # Filter boxes that are classified as traffic lights
        boxes, scores, classes = self.filter_class(self.traffic_light_class, boxes, scores, classes)

        if len(scores) == 0:
            return TrafficLight.UNKNOWN

        best_score_ind = np.argmax(scores)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        height = image.shape[0]
        width = image.shape[1]
        box_coords = self.to_image_coords(boxes, height, width)

        image_cropped = self.crop_image(image, box_coords[best_score_ind])
        color = self.color_detection(image_cropped)
        #print("detected color {}".format(color))
        return color
