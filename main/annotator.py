#!/bin/python
import argparse
import glob
import json
import os
import re

import cv2
import numpy as np
from tqdm import tqdm

from lxml import etree
import xml.etree.cElementTree as ET


DELAY = 20 # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


# parser = argparse.ArgumentParser(description='Open-source image labeling tool')
# parser.add_argument('-i', '--input_dir', default='input', type=str, help='Path to input directory')
# parser.add_argument('-o', '--output_dir', default='output', type=str, help='Path to output directory')
# parser.add_argument('-t', '--thickness', default='1', type=int, help='Bounding box and cross line thickness')
# parser.add_argument('--draw-from-PASCAL-files', action='store_true', help='Draw bounding boxes from the PASCAL files') # default YOLO
'''
tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']
    Recomended tracker_type:
        DASIAMRPN -> best
        KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
        CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
        MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
'''
# parser.add_argument('--tracker', default='KCF', type=str, help="tracker_type being used: ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']")
# parser.add_argument('-n', '--n_frames', default='200', type=int, help='number of frames to track object for')
# args = parser.parse_args()

class ClassManager():
    def __init__(self, class_list):
        self.class_index = 0
        self.class_list = class_list
        self.last_class_index = len(class_list) - 1

        # Make the class colors the same each session
        # The colors are in BGR order because we're using OpenCV
        class_rgb = [
            (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0),
            (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128)]
        self.class_rgb = np.array(class_rgb)
        # If there are still more classes, add new colors randomly
        num_colors_missing = len(self.class_list) - len(class_rgb)
        if num_colors_missing > 0:
            more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
            self.class_rgb = np.vstack([self.class_rgb, more_colors])

    def init_class_index(self):
        def set_class_index_callback(x):
            return self.set_class_index(x)
        cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0, self.last_class_index, set_class_index_callback)

    def set_class_index(self, x):
        self.class_index = x
        text = 'Selected class {}/{} -> {}'.format(str(self.class_index), str(self.last_class_index), self.class_list[self.class_index])
        display_text(text, 3000)


class ImageManager():
    def __init__(self):
        self.image_path_list = []
        self.video_name_dict = {}
        self.img_objects = []
        self.last_img_index = 0
        self.img_index = 0
        self.img = None

    def init_img_index(self):
        def set_img_index_callback(x):
            return self.set_img_index(x)

        cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0, self.last_img_index, set_img_index_callback)

    def set_img_index(self, x):
        self.img_index = x
        img_path = self.image_path_list[self.img_index]
        self.img = cv2.imread(img_path)
        text = 'Showing image {}/{}, path: {}'.format(str(self.img_index), str(self.last_img_index), img_path)
        display_text(text, 1000)


# INPUT_DIR  = args.input_dir
# OUTPUT_DIR = args.output_dir
# N_FRAMES   = args.n_frames
# TRACKER_TYPE = args.tracker

# if TRACKER_TYPE == "DASIAMRPN":
from dasiamrpn import dasiamrpn

WINDOW_NAME    = 'OpenLabeling'
TRACKBAR_IMG   = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'PASCAL_VOC' : '.xml', 'YOLO_darknet' : '.txt'}

# DRAW_FROM_PASCAL = args.draw_from_PASCAL_files

LINE_THICKNESS = 3


# Check if a point belongs to a rectangle
def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
    return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom



# Class to deal with bbox resizing
class dragBBox:
    def __init__(self, class_manager, image_manager):
        self.class_manager = class_manager
        self.image_manager = image_manager
        self.anchor_being_dragged = None
        self.selected_object = None

    '''
        LT -- MT -- RT
        |            |
        LM          RM
        |            |
        LB -- MB -- RB
    '''

    # Size of resizing anchors (depends on LINE_THICKNESS)
    sRA = LINE_THICKNESS * 2

    '''
    \brief This method is used to check if a current mouse position is inside one of the resizing anchors of a bbox
    '''

    def check_point_inside_resizing_anchors(self, eX, eY, obj):
        _class_name, x_left, y_top, x_right, y_bottom = obj
        # first check if inside the bbox region (to avoid making 8 comparisons per object)
        if pointInRect(eX, eY,
                        x_left - self.sRA,
                        y_top - self.sRA,
                        x_right + self.sRA,
                        y_bottom + self.sRA):

            anchor_dict = get_anchors_rectangles(x_left, y_top, x_right, y_bottom, self)
            for anchor_key in anchor_dict:
                rX_left, rY_top, rX_right, rY_bottom = anchor_dict[anchor_key]
                if pointInRect(eX, eY, rX_left, rY_top, rX_right, rY_bottom):
                    self.anchor_being_dragged = anchor_key
                    break

    '''
    \brief This method is used to select an object if one presses a resizing anchor
    '''
    def handler_left_mouse_down(self, eX, eY, obj):
        self.check_point_inside_resizing_anchors(eX, eY, obj)
        if self.anchor_being_dragged is not None:
            self.selected_object = obj

    def handler_mouse_move(self, eX, eY, input_dir, output_dir):
        if self.selected_object is not None:
            class_name, x_left, y_top, x_right, y_bottom = self.selected_object

            # Do not allow the bbox to flip upside down (given a margin)
            margin = 3 * self.sRA
            change_was_made = False

            if self.anchor_being_dragged[0] == "L":
                # left anchors (LT, LM, LB)
                if eX < x_right - margin:
                    x_left = eX
                    change_was_made = True
            elif self.anchor_being_dragged[0] == "R":
                # right anchors (RT, RM, RB)
                if eX > x_left + margin:
                    x_right = eX
                    change_was_made = True

            if self.anchor_being_dragged[1] == "T":
                # top anchors (LT, RT, MT)
                if eY < y_bottom - margin:
                    y_top = eY
                    change_was_made = True
            elif self.anchor_being_dragged[1] == "B":
                # bottom anchors (LB, RB, MB)
                if eY > y_top + margin:
                    y_bottom = eY
                    change_was_made = True

            if change_was_made:
                action = "resize_bbox:{}:{}:{}:{}".format(x_left, y_top, x_right, y_bottom)
                edit_bbox(self.selected_object, action, input_dir, output_dir, self.class_manager, self.image_manager)
                # update the selected bbox
                self.selected_object = [class_name, x_left, y_top, x_right, y_bottom]

    '''
    \brief This method will reset this class
     '''
    def handler_left_mouse_up(self, eX, eY):
        # print(
        #     'selected_object not None: ' + self.selected_object is not None + 'anchor_being_dragged: ' + self.anchor_being_dragged
        # )
        if self.selected_object is not None:
            self.selected_object = None
            self.anchor_being_dragged = None

def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)


def yolo_format(class_index, point_1, point_2, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x_center = float((point_1[0] + point_2[0]) / (2.0 * width) )
    y_center = float((point_1[1] + point_2[1]) / (2.0 * height))
    x_width = float(abs(point_2[0] - point_1[0])) / width
    y_height = float(abs(point_2[1] - point_1[1])) / height
    items = map(str, [class_index, x_center, y_center, x_width, y_height])
    return ' '.join(items)


def voc_format(class_name, point_1, point_2):
    # Order: class_name xmin ymin xmax ymax
    xmin, ymin = min(point_1[0], point_2[0]), min(point_1[1], point_2[1])
    xmax, ymax = max(point_1[0], point_2[0]), max(point_1[1], point_2[1])
    items = map(str, [class_name, xmin, ymin, xmax, ymax])
    return items

def findIndex(obj_to_find, image_manager):
    #return [(ind, img_objects[ind].index(obj_to_find)) for ind in xrange(len(img_objects)) if item in img_objects[ind]]
    ind = -1

    ind_ = 0
    for listElem in image_manager.img_objects:
        if listElem == obj_to_find:
            ind = ind_
            return ind
        ind_ = ind_+1

    return ind

def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def append_bb(ann_path, line, extension):
    if '.txt' in extension:
        with open(ann_path, 'a') as myfile:
            myfile.write(line + '\n') # append line
    elif '.xml' in extension:
        class_name, xmin, ymin, xmax, ymax = line

        tree = ET.parse(ann_path)
        annotation = tree.getroot()

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = xmin
        ET.SubElement(bbox, 'ymin').text = ymin
        ET.SubElement(bbox, 'xmax').text = xmax
        ET.SubElement(bbox, 'ymax').text = ymax

        xml_str = ET.tostring(annotation)
        write_xml(xml_str, ann_path)


def yolo_to_voc(x_center, y_center, x_width, y_height, width, height):
    x_center *= float(width)
    y_center *= float(height)
    x_width *= float(width)
    y_height *= float(height)
    x_width /= 2.0
    y_height /= 2.0
    xmin = int(round(x_center - x_width))
    ymin = int(round(y_center - y_height))
    xmax = int(round(x_center + x_width))
    ymax = int(round(y_center + y_height))
    return xmin, ymin, xmax, ymax


def get_xml_object_data(obj, class_manager):
    class_name = obj.find('name').text
    class_index = class_manager.class_list.index(class_name)
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return [class_name, class_index, xmin, ymin, xmax, ymax]


def get_txt_object_data(obj, img_width, img_height, class_manager):
    classId, centerX, centerY, bbox_width, bbox_height = obj.split()
    bbox_width = float(bbox_width)
    bbox_height  = float(bbox_height)
    centerX = float(centerX)
    centerY = float(centerY)

    class_index = int(classId)
    class_name = class_manager.class_list[class_index]
    xmin = int(img_width * centerX - img_width * bbox_width/2.0)
    xmax = int(img_width * centerX + img_width * bbox_width/2.0)
    ymin = int(img_height * centerY - img_height * bbox_height/2.0)
    ymax = int(img_height * centerY + img_height * bbox_height/2.0)
    return [class_name, class_index, xmin, ymin, xmax, ymax]


def get_anchors_rectangles(xmin, ymin, xmax, ymax, drag_box):
    anchor_list = {}

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2

    L_ = [xmin - drag_box.sRA, xmin + drag_box.sRA]
    M_ = [mid_x - drag_box.sRA, mid_x + drag_box.sRA]
    R_ = [xmax - drag_box.sRA, xmax + drag_box.sRA]
    _T = [ymin - drag_box.sRA, ymin + drag_box.sRA]
    _M = [mid_y - drag_box.sRA, mid_y + drag_box.sRA]
    _B = [ymax - drag_box.sRA, ymax + drag_box.sRA]

    anchor_list['LT'] = [L_[0], _T[0], L_[1], _T[1]]
    anchor_list['MT'] = [M_[0], _T[0], M_[1], _T[1]]
    anchor_list['RT'] = [R_[0], _T[0], R_[1], _T[1]]
    anchor_list['LM'] = [L_[0], _M[0], L_[1], _M[1]]
    anchor_list['RM'] = [R_[0], _M[0], R_[1], _M[1]]
    anchor_list['LB'] = [L_[0], _B[0], L_[1], _B[1]]
    anchor_list['MB'] = [M_[0], _B[0], M_[1], _B[1]]
    anchor_list['RB'] = [R_[0], _B[0], R_[1], _B[1]]

    return anchor_list


def draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color, listener):
    anchor_dict = get_anchors_rectangles(xmin, ymin, xmax, ymax, listener.drag_box)
    for anchor_key in anchor_dict:
        x1, y1, x2, y2 = anchor_dict[anchor_key]
        cv2.rectangle(tmp_img, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)
    return tmp_img

def draw_bboxes_from_file(tmp_img, annotation_paths, width, height, draw_from_pascal, listener, class_manager, image_manager):
    ann_path = None
    if draw_from_pascal:
        # Drawing bounding boxes from the PASCAL files
        ann_path = next(path for path in annotation_paths if 'PASCAL_VOC' in path)
    else:
        # Drawing bounding boxes from the YOLO files
        ann_path = next(path for path in annotation_paths if 'YOLO_darknet' in path)
    if os.path.isfile(ann_path):
        if draw_from_pascal:
            tree = ET.parse(ann_path)
            annotation = tree.getroot()
            for idx, obj in enumerate(annotation.findall('object')):
                class_name, class_index, xmin, ymin, xmax, ymax = get_xml_object_data(obj, class_manager)
                #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                image_manager.img_objects.append([class_index, xmin, ymin, xmax, ymax])
                color = class_manager.class_rgb[class_index].tolist()
                # draw bbox
                cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                # draw resizing anchors if the object is selected
                if listener.is_bbox_selected:
                    if idx == listener.selected_bbox:
                        tmp_img = draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color. listener)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(tmp_img, class_name, (xmin, ymin - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)
        else:
            # Draw from YOLO
            with open(ann_path) as fp:
                for idx, line in enumerate(fp):
                    obj = line
                    class_name, class_index, xmin, ymin, xmax, ymax = get_txt_object_data(obj, width, height, class_manager)
                    #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                    image_manager.img_objects.append([class_index, xmin, ymin, xmax, ymax])
                    color = class_manager.class_rgb[class_index].tolist()
                    # draw bbox
                    cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                    # draw resizing anchors if the object is selected
                    if listener.is_bbox_selected:
                        if idx == listener.selected_bbox:
                            tmp_img = draw_bbox_anchors(tmp_img, xmin, ymin, xmax, ymax, color, listener)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(tmp_img, class_name, (xmin, ymin - 5), font, 0.6, color, LINE_THICKNESS, cv2.LINE_AA)
    return tmp_img


def get_bbox_area(x1, y1, x2, y2):
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    return width*height


def set_selected_bbox(set_class, listener):
    smallest_area = -1
    # if clicked inside multiple bboxes selects the smallest one
    for idx, obj in enumerate(listener.image_manager.img_objects):
        ind, x1, y1, x2, y2 = obj
        x1 = x1 - listener.drag_box.sRA
        y1 = y1 - listener.drag_box.sRA
        x2 = x2 + listener.drag_box.sRA
        y2 = y2 + listener.drag_box.sRA
        if pointInRect(listener.mouse_x, listener.mouse_y, x1, y1, x2, y2):
            listener.is_bbox_selected = True
            tmp_area = get_bbox_area(x1, y1, x2, y2)
            if tmp_area < smallest_area or smallest_area == -1:
                smallest_area = tmp_area
                listener.selected_bbox = idx
                if set_class:
                    # set class to the one of the selected bounding box
                    cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, ind)


def is_mouse_inside_delete_button(listener):
    for idx, obj in enumerate(listener.image_manager.img_objects):
        if idx == listener.selected_bbox:
            _ind, x1, y1, x2, y2 = obj
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            if pointInRect(listener.mouse_x, listener.mouse_y, x1_c, y1_c, x2_c, y2_c):
                return True
    return False


def edit_bbox(obj_to_edit, action, input_dir, output_dir, class_manager, image_manager):
    tracker_dir = os.path.join(output_dir, '.tracker')
    ''' action = `delete`
                 `change_class:new_class_index`
                 `resize_bbox:new_x_left:new_y_top:new_x_right:new_y_bottom`
    '''
    if 'change_class' in action:
        new_class_index = int(action.split(':')[1])
    elif 'resize_bbox' in action:
        new_x_left = max(0, int(action.split(':')[1]))
        new_y_top = max(0, int(action.split(':')[2]))
        new_x_right = min(1, int(action.split(':')[3]))
        new_y_bottom = min(1, int(action.split(':')[4]))

    # 1. initialize bboxes_to_edit_dict
    #    (we use a dict since a single label can be associated with multiple ones in videos)
    bboxes_to_edit_dict = {}
    current_img_path = image_manager.image_path_list[image_manager.img_index]
    bboxes_to_edit_dict[current_img_path] = obj_to_edit

    # 2. add elements to bboxes_to_edit_dict
    '''
        If the bbox is in the json file then it was used by the video Tracker, hence,
        we must also edit the next predicted bboxes associated to the same `anchor_id`.
    '''
    # if `current_img_path` is a frame from a video
    is_from_video, video_name = is_frame_from_video(current_img_path, input_dir, image_manager)
    if is_from_video:
        # get json file corresponding to that video
        json_file_path = '{}.json'.format(os.path.join(tracker_dir, video_name))
        file_exists, json_file_data = get_json_file_data(json_file_path)
        # if json file exists
        if file_exists:
            # match obj_to_edit with the corresponding json object
            frame_data_dict = json_file_data['frame_data_dict']
            json_object_list = get_json_file_object_list(current_img_path, frame_data_dict)
            obj_matched = get_json_object_dict(obj_to_edit, json_object_list)
            # if match found
            if obj_matched is not None:
                # get this object's anchor_id
                anchor_id = obj_matched['anchor_id']

                frame_path_list = get_next_frame_path_list(video_name, current_img_path, image_manager)
                frame_path_list.insert(0, current_img_path)

                if 'change_class' in action:
                    # add also the previous frames
                    prev_path_list = get_prev_frame_path_list(video_name, current_img_path, image_manager)
                    frame_path_list = prev_path_list + frame_path_list

                # update json file if contain the same anchor_id
                for frame_path in frame_path_list:
                    json_object_list = get_json_file_object_list(frame_path, frame_data_dict)
                    json_obj = get_json_file_object_by_id(json_object_list, anchor_id)
                    if json_obj is not None:
                        bboxes_to_edit_dict[frame_path] = [
                            json_obj['class_index'],
                            json_obj['bbox']['xmin'],
                            json_obj['bbox']['ymin'],
                            json_obj['bbox']['xmax'],
                            json_obj['bbox']['ymax']
                        ]
                        # edit json file
                        if 'delete' in action:
                            json_object_list.remove(json_obj)
                        elif 'change_class' in action:
                            json_obj['class_index'] = new_class_index
                        elif 'resize_bbox' in action:
                            json_obj['bbox']['xmin'] = new_x_left
                            json_obj['bbox']['ymin'] = new_y_top
                            json_obj['bbox']['xmax'] = new_x_right
                            json_obj['bbox']['ymax'] = new_y_bottom
                    else:
                        break

                # save the edited data
                with open(json_file_path, 'w') as outfile:
                    json.dump(json_file_data, outfile, sort_keys=True, indent=4)

    # 3. loop through bboxes_to_edit_dict and edit the corresponding annotation files
    for path in bboxes_to_edit_dict:
        obj_to_edit = bboxes_to_edit_dict[path]
        class_index, xmin, ymin, xmax, ymax = map(int, obj_to_edit)

        for ann_path in get_annotation_paths(path, annotation_formats, output_dir):
            if '.txt' in ann_path:
                # edit YOLO file
                with open(ann_path, 'r') as old_file:
                    lines = old_file.readlines()

                yolo_line = yolo_format(class_index, (xmin, ymin), (xmax, ymax), 1, 1) # TODO: height and width ought to be stored
                ind = findIndex(obj_to_edit, image_manager)
                i=0

                with open(ann_path, 'w') as new_file:
                    for line in lines:

                        if i != ind:
                           new_file.write(line)

                        elif 'change_class' in action:
                            new_yolo_line = yolo_format(new_class_index, (xmin, ymin), (xmax, ymax), 1, 1)
                            new_file.write(new_yolo_line + '\n')
                        elif 'resize_bbox' in action:
                            new_yolo_line = yolo_format(class_index, (new_x_left, new_y_top), (new_x_right, new_y_bottom), 1, 1)
                            new_file.write(new_yolo_line + '\n')

                        i=i+1

            elif '.xml' in ann_path:
                # edit PASCAL VOC file
                tree = ET.parse(ann_path)
                annotation = tree.getroot()
                for obj in annotation.findall('object'):
                    class_name_xml, class_index_xml, xmin_xml, ymin_xml, xmax_xml, ymax_xml = get_xml_object_data(obj, class_manager)
                    if ( class_index == class_index_xml and
                                     xmin == xmin_xml and
                                     ymin == ymin_xml and
                                     xmax == xmax_xml and
                                     ymax == ymax_xml ) :
                        if 'delete' in action:
                            annotation.remove(obj)
                        elif 'change_class' in action:
                            # edit object class name
                            object_class = obj.find('name')
                            object_class.text = class_manager.class_list[new_class_index]
                        elif 'resize_bbox' in action:
                            object_bbox = obj.find('bndbox')
                            object_bbox.find('xmin').text = str(new_x_left)
                            object_bbox.find('ymin').text = str(new_y_top)
                            object_bbox.find('xmax').text = str(new_x_right)
                            object_bbox.find('ymax').text = str(new_y_bottom)
                        break

                xml_str = ET.tostring(annotation)
                write_xml(xml_str, ann_path)


class MouseListener():
    def __init__(self, class_manager, drag_box, image_manager, input_dir, output_dir):
        self.class_manager = class_manager
        self.drag_box = drag_box
        self.image_manager = image_manager
        self.input_dir = input_dir
        self.output_dir = output_dir

        # selected bounding box
        self.prev_was_double_click = False
        self.is_bbox_selected = False
        self.selected_bbox = -1

        self.mouse_x = 0
        self.mouse_y = 0
        self.point_1 = (-1, -1)
        self.point_2 = (-1, -1)

        '''
            0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)
        '''
    def listen(self):
        def listener_func(event, x, y, flags, param):
            return self.mouse_listener(event, x, y, flags, param)

        cv2.setMouseCallback(WINDOW_NAME, listener_func)

    def mouse_listener(self, event, x, y, flags, param):
        # mouse callback function

        set_class = True
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.prev_was_double_click = True
            #print('Double click')
            self.point_1 = (-1, -1)
            # if clicked inside a bounding box we set that bbox
            set_selected_bbox(set_class, self)
        # By AlexeyGy: delete via right-click
        elif event == cv2.EVENT_RBUTTONDOWN:
            set_class = False
            set_selected_bbox(set_class, self)
            if self.is_bbox_selected:
                obj_to_edit = self.image_manager.img_objects[self.selected_bbox]
                edit_bbox(obj_to_edit, 'delete', self.input_dir, self.output_dir, self.class_manager, self.image_manager)
                self.is_bbox_selected = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.prev_was_double_click:
                #print('Finish double click')
                self.prev_was_double_click = False
            else:
                print(self.drag_box.anchor_being_dragged)

                # Check if mouse inside on of resizing anchors of the selected bbox
                if self.is_bbox_selected:
                    self.drag_box.handler_left_mouse_down(x, y, self.image_manager.img_objects[self.selected_bbox])

                if self.drag_box.anchor_being_dragged is None:
                    if self.point_1[0] == -1:
                        if self.is_bbox_selected:
                            if is_mouse_inside_delete_button(self):
                                set_selected_bbox(set_class, self)
                                obj_to_edit = self.image_manager.img_objects[self.selected_bbox]
                                edit_bbox(obj_to_edit, 'delete', self.input_dir, self.output_dir, self.class_manager, self.image_manager)
                            self.is_bbox_selected = False
                        else:
                            # first click (start drawing a bounding box or delete an item)
                            # print('First click')
                            self.point_1 = (x, y)
                    else:
                        # minimal size for bounding box to avoid errors
                        threshold = 5
                        if abs(x - self.point_1[0]) > threshold or abs(y - self.point_1[1]) > threshold:
                            # second click
                            # print('Second click')
                            self.point_2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            # print('cv2.EVENT_LBUTTONUP')
            if self.drag_box.anchor_being_dragged is not None:
                self.drag_box.handler_left_mouse_up(x, y)



def get_close_icon(x1, y1, x2, y2):
    percentage = 0.05
    height = -1
    while height < 15 and percentage < 1.0:
        height = int((y2 - y1) * percentage)
        percentage += 0.1
    return (x2 - height), y1, x2, (y1 + height)


def draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c):
    red = (0,0,255)
    cv2.rectangle(tmp_img, (x1_c + 1, y1_c - 1), (x2_c, y2_c), red, -1)
    white = (255, 255, 255)
    cv2.line(tmp_img, (x1_c, y1_c), (x2_c, y2_c), white, 2)
    cv2.line(tmp_img, (x1_c, y2_c), (x2_c, y1_c), white, 2)
    return tmp_img


def draw_info_bb_selected(tmp_img, listener):
    for idx, obj in enumerate(listener.image_manager.img_objects):
        ind, x1, y1, x2, y2 = obj
        if idx == listener.selected_bbox:
            x1_c, y1_c, x2_c, y2_c = get_close_icon(x1, y1, x2, y2)
            draw_close_icon(tmp_img, x1_c, y1_c, x2_c, y2_c)
    return tmp_img


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def convert_video_to_images(video_path, n_frames, desired_img_format):
    # create folder to store images (if video was not converted to images already)
    file_path, file_extension = os.path.splitext(video_path)
    # append extension to avoid collision of videos with same name
    # e.g.: `video.mp4`, `video.avi` -> `video_mp4/`, `video_avi/`
    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(' Converting video to individual frames...')
        cap = cv2.VideoCapture(video_path)
        os.makedirs(file_path)
        # read the video
        for i in tqdm(range(n_frames)):
            if not cap.isOpened():
                break
            # capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # save each frame (we use this format to avoid repetitions)
                frame_name =  '{}_{}{}'.format(video_name_ext, i, desired_img_format)
                frame_path = os.path.join(file_path, frame_name)
                cv2.imwrite(frame_path, frame)
        # release the video capture object
        cap.release()
    return file_path, video_name_ext


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_paths(img_path, annotation_formats, output_dir):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(output_dir, ann_dir)
        new_path = os.path.join(new_path, os.path.basename(os.path.normpath(img_path))) #img_path.replace(INPUT_DIR, new_path, 1)
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths


def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)


def save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height, class_manager):
    for ann_path in annotation_paths:
        if '.txt' in ann_path:
            line = yolo_format(class_index, point_1, point_2, width, height)
            append_bb(ann_path, line, '.txt')
        elif '.xml' in ann_path:
            line = voc_format(class_manager.class_list[class_index], point_1, point_2)
            append_bb(ann_path, line, '.xml')

def is_frame_from_video(img_path, input_dir, image_manager):
    for video_name in image_manager.video_name_dict:
        video_dir = os.path.join(input_dir, video_name)
        if os.path.dirname(img_path) == video_dir:
            # image belongs to a video
            return True, video_name
    return False, None


def get_json_file_data(json_file_path):
    if os.path.isfile(json_file_path):
        with open(json_file_path) as f:
            data = json.load(f)
            return True, data
    else:
        return False, {'n_anchor_ids':0, 'frame_data_dict':{}}


def get_prev_frame_path_list(video_name, img_path, image_manager):
    first_index = image_manager.video_name_dict[video_name]['first_index']
    last_index = image_manager.video_name_dict[video_name]['last_index']
    img_index = image_manager.image_path_list.index(img_path)
    return image_manager.image_path_list[first_index:img_index]


def get_next_frame_path_list(video_name, img_path, image_manager):
    first_index = image_manager.video_name_dict[video_name]['first_index']
    last_index = image_manager.video_name_dict[video_name]['last_index']
    img_index = image_manager.image_path_list.index(img_path)
    return image_manager.image_path_list[(img_index + 1):last_index]


def get_json_object_dict(obj, json_object_list):
    if len(json_object_list) > 0:
        class_index, xmin, ymin, xmax, ymax = map(int, obj)
        for d in json_object_list:
                    if ( d['class_index'] == class_index and
                         d['bbox']['xmin'] == xmin and
                         d['bbox']['ymin'] == ymin and
                         d['bbox']['xmax'] == xmax and
                         d['bbox']['ymax'] == ymax ) :
                        return d
    return None


def remove_already_tracked_objects(object_list, img_path, json_file_data):
    frame_data_dict = json_file_data['frame_data_dict']
    json_object_list = get_json_file_object_list(img_path, frame_data_dict)
    # copy the list since we will be deleting elements without restarting the loop
    temp_object_list = object_list[:]
    for obj in temp_object_list:
        obj_dict = get_json_object_dict(obj, json_object_list)
        if obj_dict is not None:
            object_list.remove(obj)
            json_object_list.remove(obj_dict)
    return object_list


def get_json_file_object_by_id(json_object_list, anchor_id):
    for obj_dict in json_object_list:
        if obj_dict['anchor_id'] == anchor_id:
            return obj_dict
    return None


def get_json_file_object_list(img_path, frame_data_dict):
    object_list = []
    if img_path in frame_data_dict:
        object_list = frame_data_dict[img_path]
    return object_list


def json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj):
    object_list = get_json_file_object_list(img_path, frame_data_dict)
    class_index, xmin, ymin, xmax, ymax = obj

    bbox = {
      'xmin': xmin,
      'ymin': ymin,
      'xmax': xmax,
      'ymax': ymax
    }

    temp_obj = {
      'anchor_id': anchor_id,
      'prediction_index': pred_counter,
      'class_index': class_index,
      'bbox': bbox
    }

    object_list.append(temp_obj)
    frame_data_dict[img_path] = object_list

    return frame_data_dict


class LabelTracker():
    ''' Special thanks to Rafael Caballero Gonzalez '''
    # extract the OpenCV version info, e.g.:
    # OpenCV 3.3.4 -> [major_ver].[minor_ver].[subminor_ver]
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # TODO: press ESC to stop the tracking process

    def __init__(self, tracker_type, init_frame, next_frame_path_list):
        tracker_types = ['CSRT', 'KCF','MOSSE', 'MIL', 'BOOSTING', 'MEDIANFLOW', 'TLD', 'GOTURN', 'DASIAMRPN']
        ''' Recomended tracker_type:
              KCF -> KCF is usually very good (minimum OpenCV 3.1.0)
              CSRT -> More accurate than KCF but slightly slower (minimum OpenCV 3.4.2)
              MOSSE -> Less accurate than KCF but very fast (minimum OpenCV 3.4.1)
        '''
        self.tracker_type = tracker_type
        # -- TODO: remove this if I assume OpenCV version > 3.4.0
        if tracker_type == tracker_types[0] or tracker_type == tracker_types[2]:
            if int(self.major_ver == 3) and int(self.minor_ver) < 4:
                self.tracker_type = tracker_types[1] # Use KCF instead of CSRT or MOSSE
        # --
        self.init_frame = init_frame
        self.next_frame_path_list = next_frame_path_list

        self.img_h, self.img_w = init_frame.shape[:2]


    def call_tracker_constructor(self, tracker_type):
        if tracker_type == 'DASIAMRPN':
            tracker = dasiamrpn()
        else:
            # -- TODO: remove this if I assume OpenCV version > 3.4.0
            if int(self.major_ver == 3) and int(self.minor_ver) < 3:
                #tracker = cv2.Tracker_create(tracker_type)
                pass
            # --
            else:
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError as error:
                    print(error)
                    print('\nMake sure that OpenCV contribute is installed: opencv-contrib-python\n')
                if tracker_type == 'CSRT':
                    tracker = cv2.TrackerCSRT_create()
                elif tracker_type == 'KCF':
                    tracker = cv2.TrackerKCF_create()
                elif tracker_type == 'MOSSE':
                    tracker = cv2.TrackerMOSSE_create()
                elif tracker_type == 'MIL':
                    tracker = cv2.TrackerMIL_create()
                elif tracker_type == 'BOOSTING':
                    tracker = cv2.TrackerBoosting_create()
                elif tracker_type == 'MEDIANFLOW':
                    tracker = cv2.TrackerMedianFlow_create()
                elif tracker_type == 'TLD':
                    tracker = cv2.TrackerTLD_create()
                elif tracker_type == 'GOTURN':
                    tracker = cv2.TrackerGOTURN_create()
        return tracker


    def start_tracker(self, json_file_data, json_file_path, img_path, obj, color, annotation_formats, output_dir, n_frames):
        tracker = self.call_tracker_constructor(self.tracker_type)
        anchor_id = json_file_data['n_anchor_ids']
        frame_data_dict = json_file_data['frame_data_dict']

        pred_counter = 0
        frame_data_dict = json_file_add_object(frame_data_dict, img_path, anchor_id, pred_counter, obj)
        # tracker bbox format: xmin, xmax, w, h
        xmin, ymin, xmax, ymax = obj[1:5]
        initial_bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        tracker.init(self.init_frame, initial_bbox)
        for frame_path in self.next_frame_path_list:
            next_image = cv2.imread(frame_path)
            # get the new bbox prediction of the object
            success, bbox = tracker.update(next_image.copy())
            if pred_counter >= n_frames:
                success = False
            if success:
                pred_counter += 1
                xmin, ymin, w, h = map(int, bbox)
                xmax = xmin + w
                ymax = ymin + h
                obj = [class_index, xmin, ymin, xmax, ymax]
                frame_data_dict = json_file_add_object(frame_data_dict, frame_path, anchor_id, pred_counter, obj)
                cv2.rectangle(next_image, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)
                # save prediction
                annotation_paths = get_annotation_paths(frame_path, annotation_formats, output_dir)
                save_bounding_box(annotation_paths, class_index, (xmin, ymin), (xmax, ymax), self.img_w, self.img_h, class_manager)
                # show prediction
                cv2.imshow(WINDOW_NAME, next_image)
                pressed_key = cv2.waitKey(DELAY)
            else:
                break

        json_file_data.update({'n_anchor_ids': (anchor_id + 1)})
        # save the updated data
        with open(json_file_path, 'w') as outfile:
            json.dump(json_file_data, outfile, sort_keys=True, indent=4)


def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)

def run(
        input_dir='input', output_dir='output', tracker_type='KCF', n_frames=200,
        draw_from_pascal=False, max_n_frames=None
    ):
    tracker_dir = os.path.join(output_dir, '.tracker')
    cwd = os.getcwd()

    # change to the directory of this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    image_manager = ImageManager()

    # load all images and videos (with multiple extensions) from a directory using OpenCV
    for f in sorted(os.listdir(input_dir), key = natural_sort_key):
        f_path = os.path.join(input_dir, f)
        if os.path.isdir(f_path):
            # skip directories
            continue
        # check if it is an image
        test_img = cv2.imread(f_path)
        if test_img is not None:
            image_manager.image_path_list.append(f_path)
        else:
            # test if it is a video
            test_video_cap = cv2.VideoCapture(f_path)
            n_frames = int(test_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_video_cap.release()
            if n_frames > 0:
                # it is a video
                desired_img_format = '.jpg'
                frames_to_process = n_frames
                if max_n_frames is not None:
                    frames_to_process = min(n_frames, int(max_n_frames))
                video_frames_path, video_name_ext = convert_video_to_images(f_path, frames_to_process, desired_img_format)
                # add video frames to image list
                frame_list = sorted(os.listdir(video_frames_path), key = natural_sort_key)
                ## store information about those frames
                first_index = len(image_manager.image_path_list)
                last_index = first_index + len(frame_list) # exclusive
                indexes_dict = {}
                indexes_dict['first_index'] = first_index
                indexes_dict['last_index'] = last_index
                image_manager.video_name_dict[video_name_ext] = indexes_dict
                image_manager.image_path_list.extend((os.path.join(video_frames_path, frame) for frame in frame_list))
    image_manager.last_img_index = len(image_manager.image_path_list) - 1

    # create output directories
    if len(image_manager.video_name_dict) > 0:
        if not os.path.exists(tracker_dir):
            os.makedirs(tracker_dir)
    for ann_dir in annotation_formats:
        new_dir = os.path.join(output_dir, ann_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for video_name_ext in image_manager.video_name_dict:
            new_video_dir = os.path.join(new_dir, video_name_ext)
            if not os.path.exists(new_video_dir):
                os.makedirs(new_video_dir)

    # create empty annotation files for each image, if it doesn't exist already
    for img_path in image_manager.image_path_list:
        # image info for the .xml file
        test_img = cv2.imread(img_path)
        abs_path = os.path.abspath(img_path)
        folder_name = os.path.dirname(img_path)
        image_name = os.path.basename(img_path)
        img_height, img_width, depth = (str(number) for number in test_img.shape)

        for ann_path in get_annotation_paths(img_path, annotation_formats, output_dir):
            if not os.path.isfile(ann_path):
                if '.txt' in ann_path:
                    open(ann_path, 'a').close()
                elif '.xml' in ann_path:
                    create_PASCAL_VOC_xml(ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)

    # load class list
    with open('class_list.txt') as f:
        class_list = list(nonblank_lines(f))
    
    class_manager = ClassManager(class_list)

    drag_box = dragBBox(class_manager, image_manager)
    
    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    
    listener = MouseListener(class_manager, drag_box, image_manager, input_dir, output_dir)
    listener.listen()

    # selected image
    image_manager.init_img_index()

    # selected class
    if class_manager.last_class_index != 0:
        class_manager.init_class_index()

    # initialize
    image_manager.set_img_index(0)
    edges_on = False

    display_text('Welcome!\n Press [h] for help.', 4000)

    # loop
    while True:
        color = class_manager.class_rgb[class_manager.class_index].tolist()
        # clone the img
        tmp_img = image_manager.img.copy()
        height, width = tmp_img.shape[:2]
        if edges_on == True:
            # draw edges
            tmp_img = draw_edges(tmp_img)
        # draw vertical and horizontal guide lines
        draw_line(tmp_img, listener.mouse_x, listener.mouse_y, height, width, color)
        # write selected class
        class_name = class_manager.class_list[class_manager.class_index]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        margin = 3
        text_width, text_height = cv2.getTextSize(class_name, font, font_scale, LINE_THICKNESS)[0]
        tmp_img = cv2.rectangle(tmp_img, (listener.mouse_x + LINE_THICKNESS, listener.mouse_y - LINE_THICKNESS), (listener.mouse_x + text_width + margin, listener.mouse_y - text_height - margin), complement_bgr(color), -1)
        tmp_img = cv2.putText(tmp_img, class_name, (listener.mouse_x + margin, listener.mouse_y - margin), font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
        # get annotation paths
        img_path = image_manager.image_path_list[image_manager.img_index]
        annotation_paths = get_annotation_paths(img_path, annotation_formats, output_dir)
        if drag_box.anchor_being_dragged is not None:
            drag_box.handler_mouse_move(listener.mouse_x, listener.mouse_y, input_dir, output_dir)
        # draw already done bounding boxes
        tmp_img = draw_bboxes_from_file(
            tmp_img, annotation_paths, width, height, draw_from_pascal, listener, class_manager, image_manager
        )
        # if bounding box is selected add extra info
        if listener.is_bbox_selected:
            tmp_img = draw_info_bb_selected(tmp_img, listener)
        # if first click
        if listener.point_1[0] != -1:
            # draw partial bbox
            cv2.rectangle(tmp_img, listener.point_1, (listener.mouse_x, listener.mouse_y), color, LINE_THICKNESS)
            # if second click
            if listener.point_2[0] != -1:
                # save the bounding box
                save_bounding_box(annotation_paths, class_manager.class_index, listener.point_1, listener.point_2, width, height, class_manager)
                # reset the points
                listener.point_1 = (-1, -1)
                listener.point_2 = (-1, -1)

        cv2.imshow(WINDOW_NAME, tmp_img)
        pressed_key = cv2.waitKey(DELAY)

        if drag_box.anchor_being_dragged is None:
            ''' Key Listeners START '''
            if pressed_key == ord('a') or pressed_key == ord('d'):
                # show previous image key listener
                if pressed_key == ord('a'):
                    image_manager.img_index = decrease_index(image_manager.img_index, image_manager.last_img_index)
                # show next image key listener
                elif pressed_key == ord('d'):
                    image_manager.img_index = increase_index(image_manager.img_index, image_manager.last_img_index)
                image_manager.set_img_index(image_manager.img_index)
                cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, image_manager.img_index)
            elif pressed_key == ord('s') or pressed_key == ord('w'):
                # change down current class key listener
                if pressed_key == ord('s'):
                    class_manager.class_index = decrease_index(class_manager.class_index, class_manager.last_class_index)
                # change up current class key listener
                elif pressed_key == ord('w'):
                    class_manager.class_index = increase_index(class_manager.class_index, class_manager.last_class_index)
                draw_line(tmp_img, listener.mouse_x, listener.mouse_y, height, width, color)
                class_manager.set_class_index(class_manager.class_index)
                cv2.setTrackbarPos(TRACKBAR_CLASS, WINDOW_NAME, class_manager.class_index)
                if listener.is_bbox_selected:
                    obj_to_edit = image_manager.img_objects[listener.selected_bbox]
                    edit_bbox(
                        obj_to_edit, 'change_class:{}'.format(class_manager.class_index), input_dir,
                        output_dir, class_manager, image_manager
                    )
            # help key listener
            elif pressed_key == ord('h'):
                text = ('[e] to show edges;\n'
                        '[q] to quit;\n'
                        '[a] or [d] to change Image;\n'
                        '[w] or [s] to change Class.\n'
                        )
                display_text(text, 5000)
            # show edges key listener
            elif pressed_key == ord('e'):
                if edges_on == True:
                    edges_on = False
                    display_text('Edges turned OFF!', 1000)
                else:
                    edges_on = True
                    display_text('Edges turned ON!', 1000)
            elif pressed_key == ord('p'):
                # check if the image is a frame from a video
                is_from_video, video_name = is_frame_from_video(img_path, input_dir, image_manager)
                if is_from_video:
                    # get list of objects associated to that frame
                    object_list = image_manager.img_objects[:]
                    # remove the objects in that frame that are already in the `.json` file
                    json_file_path = '{}.json'.format(os.path.join(tracker_dir, video_name))
                    file_exists, json_file_data = get_json_file_data(json_file_path)
                    if file_exists:
                        object_list = remove_already_tracked_objects(object_list, img_path, json_file_data)
                    if len(object_list) > 0:
                        # get list of frames following this image
                        next_frame_path_list = get_next_frame_path_list(video_name, img_path, image_manager)
                        # initial frame
                        init_frame = image_manager.img.copy()
                        label_tracker = LabelTracker(tracker_type, init_frame, next_frame_path_list)
                        for obj in object_list:
                            class_index = obj[0]
                            color = class_manager.class_rgb[class_index].tolist()
                            label_tracker.start_tracker(
                                json_file_data, json_file_path, img_path, obj, color, annotation_formats,
                                n_frames, class_manager
                            )
            # quit key listener
            elif pressed_key == ord('q'):
                break
            ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
    os.chdir(cwd)
