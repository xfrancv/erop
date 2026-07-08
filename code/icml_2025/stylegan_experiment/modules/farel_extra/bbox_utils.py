from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import cv2
import math
import numpy as np
import importlib.resources as resources

def bbox_area(bbox):
    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])

    bbox_height = np.hypot(*(A-D))
    bbox_width = np.hypot(*(A-B))
    area = bbox_height*bbox_width
    return area


def pick_bbox_largest(list_of_bboxes,
                      list_of_landmarks):
    """
    Selects the largest bounding box within the image.

    Note that bounding box area outside of the image is included.
    """

    max_area = -np.inf
    picked_bbox = None
    picked_landmarks = None
    for bbox, landmarks in zip(list_of_bboxes, list_of_landmarks):

        area = bbox_area(bbox)
        if area > max_area:
            max_area = area
            picked_bbox = bbox
            picked_landmarks = landmarks

    return picked_bbox, picked_landmarks, max_area


def pick_bbox_centered(list_of_bboxes,
                       list_of_landmarks,
                       image_size):
    """
    Selects the bounding box closest to the center of the image.    
    """

    image_center = (np.array(image_size)/2.0).astype(int)
    min_distance = np.inf
    picked_bbox = None
    picked_landmarks = None
    for bbox, landmarks in zip(list_of_bboxes, list_of_landmarks):
        center = np.array([np.mean(bbox[0::2]), np.mean(bbox[1::2])])
        distance = np.hypot(*(center-image_center))

        if distance < min_distance:
            min_distance = distance
            picked_bbox = bbox
            picked_landmarks = landmarks

    return picked_bbox, picked_landmarks, min_distance


def crop_bbox(img, 
              bbox, 
              out_size, 
              margin=(0, 0), 
              one_based_bbox=True):
    """
    Crop subimage around bounding box extended by a margin.

    Input:
     img input image [2D numpy array]
     bbox = [A_col,A_row,B_col,B_row,C_col,C_row,D_col,D_row] bounding box
     out_size (cols,rows) size of output image
     margin (horizontal, vertical) margin; portion of bonding box size
     one_based_bbox [bool] if True assumes bbox to be given on 1-base coordinates
    Output:
     dst: output image [numpy array]
     M: affine transformation used for the crop
    """

    A = np.float32([bbox[0], bbox[1]])
    B = np.float32([bbox[2], bbox[3]])
    C = np.float32([bbox[4], bbox[5]])
    D = np.float32([bbox[6], bbox[7]])

    if one_based_bbox:
        A = A - 1
        B = B - 1
        C = C - 1
        D = D - 1

    ext_A = A + (A-B)*margin[0] + (A-D)*margin[1]
    ext_B = B + (B-A)*margin[0] + (B-C)*margin[1]
    ext_C = C + (C-D)*margin[0] + (C-B)*margin[1]

    pts1 = np.float32([ext_A, ext_B, ext_C])
    pts2 = np.float32([[0, 0], [out_size[0]-1, 0],
                      [out_size[0]-1, out_size[1]-1]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (out_size[0], out_size[1]))

    return dst, M


def crop_extended_bbox(img, 
                       bbox, 
                       input_size, 
                       input_extension, 
                       bbox_extension, 
                       return_affine_transform=False):

    out_size = (int(input_size[0]*(1+2*input_extension[0])),
                int(input_size[1]*(1+2*input_extension[1])))
    margin = (input_extension[0]+bbox_extension[0]+2*input_extension[0]*bbox_extension[0],
              input_extension[1]+bbox_extension[1]+2*input_extension[1]*bbox_extension[1])

    out_img, M = crop_bbox(img, bbox, out_size,
                           margin=margin, one_based_bbox=True)

    if return_affine_transform:
        return out_img, M

    return out_img


def draw_text(img: np.ndarray,
              text: str,
              pos=(0, 0),
              font=cv2.FONT_HERSHEY_PLAIN,
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1),
                font, font_scale, text_color, font_thickness)

def getsize(font, text):
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top

def draw_labeled_bbox(img, bbox, text, bbox_color="blue",
                      text_box_position="top", font_size=(15, 50)):

    margin = 3  # margin between the text anx the text box edge
    text_color = "#FFFFFF"
    boundary_line_width = 2

    # if bbox is a set of points, convert it to a list of 2-tuples
    if type(bbox[0]) is not tuple:
        bbox = [(bbox[i], bbox[i+1]) for i in range(0, len(bbox), 2)]

    bbox_size = int(
        math.sqrt((bbox[0][0]-bbox[1][0])**2 + (bbox[0][1]-bbox[1][1])**2))

    if type(font_size) is tuple:
        font_size = int(min(font_size[1], max(bbox_size * 0.2, font_size[0])))

    draw = ImageDraw.Draw(img)
    with resources.path("farel.helpers.bbox.fonts", "FreeMono.ttf") as font_path:
        #print(font_path)
        font = ImageFont.truetype(str(font_path), font_size)

    #font = ImageFont.truetype("helpers/bbox/fonts/FreeMono.ttf", font_size)

    # find size of the box that encloses the text
    y_pos = 0
    x_max = 0
    y_max = 0
    for line in text:
        line_width, line_height = getsize(font,line)
        y_pos += line_height
        x_max = max(x_max, line_width-1+2*margin)
        y_max = max(y_max, y_pos-1 + margin)

    # define position of the text box
    if text_box_position == "top":
        x_min = bbox[0][0]
        y_min = bbox[0][1]-y_max
        x_max = max(x_max, bbox_size)
    elif text_box_position == "left":
        x_min = bbox[1][0]
        y_min = bbox[1][1]
    elif text_box_position == "bottom":
        x_min = bbox[3][0]
        y_min = bbox[3][1]
        x_max = max(x_max, bbox_size)

    x_max += x_min
    y_max += y_min

    # crop the region below the text box
    region = img.crop((x_min, y_min, x_max, y_max))

    # dim the region and past it back
    source = region.split()
    R, G, B = 0, 1, 2
    constant = 2  # constant by which each pixel is divided
    Red = source[R].point(lambda i: i/constant)
    Green = source[G].point(lambda i: i/constant)
    Blue = source[B].point(lambda i: i/constant)
    region = Image.merge(region.mode, (Red, Green, Blue))
    img.paste(region, (x_min, y_min, x_max, y_max))

    # draw the text to the text box
    x_pos = int(x_min)
    y_pos = int(y_min)
    for line in text:
        line_width, line_height = getsize(font,line)
        draw.text((x_pos+margin, y_pos+margin), line,
                  fill=text_color, anchor="lt", font=font)
        y_pos += line_height

    # draw the bbox
    draw.polygon(bbox, fill=None, outline=bbox_color,
                 width=boundary_line_width)

    return
