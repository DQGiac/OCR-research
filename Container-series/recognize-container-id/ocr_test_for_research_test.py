import pytesseract
from PIL import Image
# Đường dẫn đến tesseract.exe nếu cần
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
numbers = "0123456789"
def detect(net, frame, output_layers):
    """
    Return class id, confidence scores and bounding boxes for detected object (after NMS).
    This function is the same as using following built-in opencv functions:

    model = cv2.dnn_DetectionModel(net);
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True);
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    """
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    scale = 1/255
    size = (320, 320)
    # create input blob to prepare image for the network
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=size, mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    im_h, im_w = frame.shape[0:2]
    # run inference through the network and gather predictions from output layers
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()
    print("code_region_detector/detect(): YOLOv4 took {:.6f} seconds".format(end - start))

    # for each detection from each output layer, get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # convert yolo coords to opencv coords
                center_x = int(detection[0] * im_w)
                center_y = int(detection[1] * im_h)
                w = int(detection[2] * im_w)
                h = int(detection[3] * im_h)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # clean up
    clean_class_ids = []
    clean_confidences = []
    clean_boxes = []
    for i in indices:
        j = i[0]
        clean_class_ids.append(class_ids[j])
        clean_boxes.append(boxes[j])
        clean_confidences.append(confidences[j])
    return clean_class_ids, clean_boxes, clean_confidences

def error_check(formatted_code):
    """ Assume code is using BIC container code format. """
    fixed_code = list(formatted_code)
    n = len(fixed_code)
    start = False
    for i in range(len(fixed_code)):
        if fixed_code[i:i+2] == '\n' and not start:
            fixed_code = fixed_code[i + 3:]
        if fixed_code[i] in alphabet:
            for j in range(i + 1, min(len(fixed_code), i + 4)):
                if fixed_code[j] not in alphabet:
                    break
                if j == i + 3:
                    for ind in range(i + 4, len(fixed_code)):
                        if fixed_code[ind] not in numbers:
                            break
                        elif ind == len(fixed_code) - 1:
                            start = True
                            break
            if start:
                fixed_code = fixed_code[i:]
                break

    if not start:
        print("Code is not complete!")
        print(fixed_code)
    else:
        # First 4 characters are always letters
        for i in range(4):
            if fixed_code[i] == '1':
                fixed_code[i] = 'I'
            if fixed_code[i] == '4':
                fixed_code[i] = 'A'
            if fixed_code[i] == '6':
                fixed_code[i] = 'G'
            if fixed_code[i] == '8':
                fixed_code[i] = 'B'
        # The next 5 characters are always digits
        for i in range(4, 9):
            if fixed_code[i] == 'I':
                fixed_code[i] = '1'
            if fixed_code[i] == 'A':
                fixed_code[i] = '4'
            if fixed_code[i] == 'G':
                fixed_code[i] = '6'
            if fixed_code[i] == 'B':
                fixed_code[i] = '8'
        # This character is always letters (can only be G, R, U, P or T)
        if fixed_code[16] == '6':
            print(True)
            fixed_code[16] = 'G'
        # The following characters is usually 1
        if fixed_code[17] == 'I' or fixed_code[17] == '1':
            fixed_code[17] = '1'
    fixed_code = "".join(fixed_code)
    return fixed_code


def reformat_code(original_code):
    """ Reformat the text into better format. Format assumed to be BIC container code. """
    formatted_code = None
    n = len(original_code)
    formatted_code = original_code[0:n - 1]  # Remove weird last character
    formatted_code = formatted_code.replace(" ", "")  # Remove whitespace characters
    formatted_code = formatted_code.replace("\n", "")  # Remove whitespace character
    return formatted_code

def image_to_text(image_array):
    img = Image.fromarray(image_array)
    alphanumeric = alphabet + numbers + " "
    custom_oem_psm_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=' + alphanumeric
    # custom_oem_psm_config = r'--psm 10 --oem 3'
    text = pytesseract.image_to_string(img, config=custom_oem_psm_config, lang='eng')
    return text

import cv2
import numpy as np
import time
def rotate_image(thresh_img, debug=False):
    """ Rotate an image. Required input to be a binary image."""
    im_h, im_w = thresh_img.shape[0:2]
    if im_h > im_w:
        # Not yet implemented for vertical side code
        print("code_image_cleaner/rotate_image(): Not yet implemented for vertical side code")
        return thresh_img

    tmp = np.where(thresh_img > 0)
    row, col = tmp
    # note: column_stack is just vstack().T (aka transposed vstack)
    coords = np.column_stack((col, row))
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if debug:
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)
        debug_box_img = cv2.drawContours(thresh_img.copy(), [box_points], 0, (255, 255, 255), 2)
    # the v4.5.1 `cv2.minAreaRect` function returns values in the
    # range (0, 90]); as the rectangle rotates clockwise the
    # returned angle approach 90.
    if angle > 45:
        # if angle > 45 it will rotate left 90 degree into vertical standing form, so rotate another 270 degree
        # will bring it back to good. Otherwise, it will rotate nice.
        angle = 270 + angle

    # rotate the image
    (h, w) = thresh_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    print("code_image_cleaner/rotate_image(): rotated", angle)
    return rotated


def is_contour_bad(c, src_img):
    im_h, im_w = src_img.shape[0:2]
    box = cv2.boundingRect(c)
    x, y, w, h = box[0], box[1], box[2], box[3]
    # If image is a back code (width larger than height)
    if im_w > im_h:
        if h >= 0.6*im_h:  # likely to be a bar
            print("code_image_cleaner/is_contour_bad(): found a bar contour")
            return True
        if x < 0.4*im_w and y > 0.6*im_h:  # lower left unrelated symbols
            print("code_image_cleaner/is_contour_bad(): found a unrelated contour")
            return True
        if w*h < 0.002*im_h*im_w:  # Noise w/ area < 0.2% of image's area
            print("code_image_cleaner/is_contour_bad(): found a tiny noise contour")
            return True
        if x <= 1 or x >= (im_w-1) or y <= 1 or y >= (im_h-1):
            if w*h < 0.05*im_h*im_w:
                print(x, y, w, h, im_w, im_h)
                print("code_image_cleaner/is_contour_bad(): found a sus edge-touched small contour")
                return True
    # Else, it a side code
    else:
        if w*h < 0.001*im_h*im_w:  # Noise
            print("code_image_cleaner/is_contour_bad(): found a tiny noise contour")
            return True
        if x+w >= 0.5*im_w and y+h >= 0.4*im_h:
            print("code_image_cleaner/is_contour_bad(): found a unrelated contour")
            return True
    return False


def remove_noise(cnts, thresh, src_img, debug=False):
    print("===Start removing noise===")
    mask = np.ones(thresh.shape[:2], dtype="uint8") * 255
    # loop over the contours
    for c in cnts:
        # Draw contour for visualization
        if debug:
            box = cv2.boundingRect(c)
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(src_img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
        # if the contour is bad, draw it on the mask (to remove it later)
        if is_contour_bad(c, src_img):
            cv2.drawContours(mask, [c], -1, 0, -1)
    # remove the contours from the image and show the resulting images
    result = cv2.bitwise_and(thresh, thresh, mask=mask)
    print("=====Finish=====")
    return result


def otsu_threshold(src_img):
    """ NOT expected to return white text on black background"""
    gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh_img1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh_img2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 17)
    thresh_img = cv2.bitwise_and(thresh_img1, thresh_img2)
    return thresh_img


def make_sure_it_bbwt(thresh_img, depth=2):
    """ Make sure the thresh img has white text on black background """
    im_h, im_w = thresh_img.shape[0:2]
    # Calculate the pixel value of image border
    total_pixel_value = np.sum(thresh_img)
    center_img = thresh_img[depth:im_h-depth, depth:im_w-depth]
    center_pixel_value = np.sum(center_img)
    border_bw_value = (total_pixel_value - center_pixel_value) / (im_h*im_w - center_img.size)
    print("code_image_cleaner/is_it_bbwt():BBWT value:", border_bw_value)
    # If True mean it is not bbwt, and thresh must be invert
    if border_bw_value > 127:
        cv2.bitwise_not(thresh_img, thresh_img)


def process_image_for_ocr(src_img, debug=False):
    """
    Clean up other cluttering on the back code and return a binary image. Run this from other file.
    """

    # Binarization
    thresh = otsu_threshold(src_img)
    make_sure_it_bbwt(thresh)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Remove noise
    # clean = remove_noise(cnts, thresh, src_img, debug)
    # Rotate
    # rotated = rotate_image(thresh, debug)
    return thresh

src_img = cv2.imread("/content/BSIU.jpg")
src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
a = Image.fromarray(src_img)
a.save("/content/inhold.png")

im_rgb = (cv2.imread("/content/inhold.png"))
a = process_image_for_ocr(im_rgb, True)
a = cv2.bitwise_not(a)
b = Image.fromarray(a)
b.save("/content/inhold.png")

Image.open("inhold.png")

text = image_to_text(a)
print(text)
text = reformat_code(text)
text = error_check(text.strip())
print(text)