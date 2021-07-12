import numpy as np
from imutils import face_utils
import cv2
import math
from utils.terminal_text import bcolors as _bcolors


LIGHTPOINT_NUMBER = 79

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def landmarks_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2))
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def place_landmarks_single_paint(detector, predictor, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 1)
    landmarks = predictor(img_gray, faces[0])
    landmarks = face_utils.shape_to_np(landmarks)
    count = 0
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    for (sX, sY) in landmarks:
        cv2.putText(img, str(count), (sX, sY), font, 1, (0, 255, 255), 0, 0)
        cv2.circle(img, (sX, sY), 3, (0, 0, 255), -1)
        count += 1
    return landmarks

def place_landmarks_single(detector, predictor, img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray, 1)
    if(len(faces) != 0):
        landmarks = predictor(img_gray, faces[0])
        landmarks = face_utils.shape_to_np(landmarks)
        return landmarks, True
    return None, False


# calculate additional points from landmarks
def calc_additional_points(landmarks):
    points = np.empty((0, 2), dtype=int)


    # Right and left cheeks are calculated
    right_cheek0 = dot_median(landmarks[29], landmarks[0])
    right_cheek1 = dot_median(landmarks[51], landmarks[0])
    right_cheek2 = dot_median(landmarks[51], landmarks[2])
    right_cheek3 = dot_median(landmarks[51], landmarks[4])
    left_cheek0 = dot_median(landmarks[29], landmarks[16])
    left_cheek1 = dot_median(landmarks[51], landmarks[16])
    left_cheek2 = dot_median(landmarks[51], landmarks[14])
    left_cheek3 = dot_median(landmarks[51], landmarks[12])
    face_cheeks = np.array(
        [right_cheek0, right_cheek1, right_cheek2, right_cheek3, left_cheek0, left_cheek1, left_cheek2, left_cheek3],
        dtype=int)
    points = np.append(points, face_cheeks, axis=0)


    # Right and left brows
    left_brow = landmarks[26]
    right_brow = landmarks[17]
    mid_brow = dot_median(landmarks[21], landmarks[22])

    face_brows = np.array([right_brow, left_brow, mid_brow], dtype=int)
    points = np.append(points, face_brows, axis=0)

    # Mouth parts
    right_beard0 = dot_median(landmarks[7], landmarks[58])
    right_beard1 = dot_median(landmarks[7], landmarks[58])
    right_beard2 = dot_median(landmarks[6], landmarks[59])
    right_beard3 = dot_median(landmarks[5], landmarks[59])
    left_beard0 = dot_median(landmarks[9], landmarks[56])
    left_beard1 = dot_median(landmarks[9], landmarks[56])
    left_beard2 = dot_median(landmarks[10], landmarks[55])
    left_beard3 = dot_median(landmarks[11], landmarks[55])
    right_lip1 = dot_median(landmarks[31], landmarks[49])
    right_lip0 = dot_median(right_cheek3, right_lip1)
    right_lip2 = dot_median(landmarks[32], landmarks[50])
    mid_lip = dot_median(landmarks[33], landmarks[51])
    left_lip1 = dot_median(landmarks[35], landmarks[53])
    left_lip0 = dot_median(left_cheek3, left_lip1)
    left_lip2 = dot_median(landmarks[34], landmarks[52])
    mid_beard = dot_median(landmarks[8], landmarks[57])
    face_mouth = np.array(
        [mid_beard, right_beard0, right_beard1, right_beard2,
         right_beard3, left_beard0, left_beard1, left_beard2,
         left_beard3, right_lip0, right_lip1, right_lip2,
        mid_lip, left_lip0, left_lip1, left_lip2])
    points = np.append(points, face_mouth, axis=0)

    # Forehead points
    # using distance between eyebrows divided by 2
    proportional_distance = int(dot_distance(landmarks[21], landmarks[22]) / 2)
    right_forehead0 = np.array([right_brow[0], right_brow[1] - proportional_distance])
    right_forehead1 = np.array([landmarks[18][0], landmarks[18][1] - proportional_distance])
    right_forehead2 = np.array([landmarks[19][0], landmarks[19][1] - proportional_distance])
    right_forehead3 = np.array([landmarks[20][0], landmarks[20][1] - proportional_distance])
    right_forehead4 = np.array([landmarks[21][0], landmarks[21][1] - proportional_distance])
    mid_forehead = np.array([mid_brow[0], mid_brow[1] - proportional_distance])
    left_forehead0 = np.array([left_brow[0], left_brow[1] - proportional_distance])
    left_forehead1 = np.array([landmarks[25][0], landmarks[25][1] - proportional_distance])
    left_forehead2 = np.array([landmarks[24][0], landmarks[24][1] - proportional_distance])
    left_forehead3 = np.array([landmarks[23][0], landmarks[23][1] - proportional_distance])
    left_forehead4 = np.array([landmarks[22][0], landmarks[22][1] - proportional_distance])
    face_forehead = np.array(
        [right_forehead0, right_forehead1, right_forehead2, right_forehead3, right_forehead4, mid_forehead,
         left_forehead0, left_forehead1, left_forehead2, left_forehead3, left_forehead4])
    points = np.append(points, face_forehead, axis=0)

    # Nose points
    proportional_distance = int(dot_distance(landmarks[30], landmarks[31]) / 2)
    # print(_bcolors.FAIL + 'porportrional distance nose: {}'.format(proportional_distance) + _bcolors.ENDC)
    right_nose0 = np.array([landmarks[27][0] - proportional_distance, landmarks[27][1]])
    right_nose1 = np.array([landmarks[28][0] - proportional_distance, landmarks[28][1]])
    right_nose2 = np.array([landmarks[29][0] - proportional_distance, landmarks[29][1]])
    right_nose3 = np.array([landmarks[30][0] - proportional_distance, landmarks[30][1]])
    left_nose0 = np.array([landmarks[27][0] + proportional_distance, landmarks[27][1]])
    left_nose1 = np.array([landmarks[28][0] + proportional_distance, landmarks[28][1]])
    left_nose2 = np.array([landmarks[29][0] + proportional_distance, landmarks[29][1]])
    left_nose3 = np.array([landmarks[30][0] + proportional_distance, landmarks[30][1]])
    face_nose = np.array(
        [right_nose0, right_nose1, right_nose2, right_nose3, left_nose0, left_nose1, left_nose2, left_nose3])
    points = np.append(points, face_nose, axis=0)

    # Secondaty points cheecks
    right_cheek_right0 = dot_median(right_cheek0, landmarks[1])
    right_cheek_right01 = dot_median(right_cheek_right0, landmarks[0])
    right_cheek_right02 = dot_median(right_cheek_right0, landmarks[1])
    right_cheek_left0 = dot_median(right_nose0, right_cheek0)
    right_cheek_right1 = dot_median(right_cheek1, landmarks[2])
    right_cheek_right11 = dot_median(right_cheek_right1, landmarks[2])
    right_cheek_left1 = dot_median(right_nose1, right_cheek1)
    right_cheek_right2 = dot_median(right_cheek2, landmarks[3])
    right_cheek_right21 = dot_median(right_cheek_right2, landmarks[3])
    right_cheek_left2 = dot_median(right_nose2, right_cheek2)
    right_cheek_right3 = dot_median(right_cheek3, landmarks[4])
    right_cheek_right31 = dot_median(right_cheek_right3, landmarks[4])
    cross_right01 = dot_median(right_cheek0, right_cheek_right1)
    cross_right12 = dot_median(right_cheek1, right_cheek_right2)
    cross_right23 = dot_median(right_cheek2, right_cheek_right3)
    face_secondary_cheeks_right = np.array(
        [right_cheek_right0, right_cheek_right01, right_cheek_right02, right_cheek_left0,
         right_cheek_right1, right_cheek_right11, right_cheek_left1,
         right_cheek_right2, right_cheek_right21, right_cheek_left2,
         right_cheek_right3, right_cheek_right31, cross_right01,
         cross_right12, cross_right23])
    points = np.append(points, face_secondary_cheeks_right, axis=0)

    left_cheek_left0 = dot_median(left_cheek0, landmarks[15])
    left_cheek_left01 = dot_median(left_cheek_left0, landmarks[16])
    left_cheek_left02 = dot_median(left_cheek_left0, landmarks[15])
    left_cheek_right0 = dot_median(left_nose0, left_cheek0)
    left_cheek_left1 = dot_median(left_cheek1, landmarks[14])
    left_cheek_left11 = dot_median(left_cheek_left1, landmarks[14])
    left_cheek_right1 = dot_median(left_nose1, left_cheek1)
    left_cheek_left2 = dot_median(left_cheek2, landmarks[13])
    left_cheek_left21 = dot_median(left_cheek_left2, landmarks[13])
    left_cheek_right2 = dot_median(left_nose2, left_cheek2)
    left_cheek_left3 = dot_median(left_cheek3, landmarks[12])
    left_cheek_left31 = dot_median(left_cheek_left3, landmarks[12])
    cross_left01 = dot_median(left_cheek0, left_cheek_left1)
    cross_left12 = dot_median(left_cheek1, left_cheek_left2)
    cross_left23 = dot_median(left_cheek2, left_cheek_left3)

    face_secondary_cheeks_left = np.array(
        [left_cheek_left0, left_cheek_left01, left_cheek_left02, left_cheek_right0,
         left_cheek_left1, left_cheek_left11, left_cheek_right1,
         left_cheek_left2, left_cheek_left21, left_cheek_right2,
         left_cheek_left3, left_cheek_left31, cross_left01, cross_left12, cross_left23])
    points = np.append(points, face_secondary_cheeks_left, axis=0)

    # Seondary beard
    right_beard_lip = dot_median(right_beard1, landmarks[59])
    left_beard_lip = dot_median(left_beard1, landmarks[55])
    mid_beard_lip = dot_median(mid_beard, landmarks[57])

    face_secondary_beard = np.array([right_beard_lip, left_beard_lip, mid_beard_lip])
    points = np.append(points, face_secondary_beard, axis=0)

    return points


def dot_median(dot1, dot2):
    x = (dot1[0] + dot2[0]) / 2
    y = (dot1[1] + dot2[1]) / 2
    return np.array([x, y], dtype=int)



def extract_colors_grayscale(lightpoints, img, radius):
    """
    Get the average color on the lightpoints:
    1. create mask
    2. paint circle on the mask
    3. get average color after applying mask
    """
    colors = np.empty((0), dtype=int)

    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for lightpoint in lightpoints:
        circle_img = np.zeros((img_grey.shape[0], img_grey.shape[1]), np.uint8)
        cv2.circle(circle_img, (lightpoint[0], lightpoint[1]), radius, (255, 255, 255), -1)
        color = cv2.mean(img_grey, mask=circle_img)[0]
        color = np.array([color])
        colors = np.append(colors, color, axis=0)
    return colors


def extract_colors(lightpoints, img, radius):
    colors = np.empty((0, 3), dtype=int)
    for lightpoint in lightpoints:
        circle_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        cv2.circle(circle_img, (lightpoint[0], lightpoint[1]), radius, (255, 255, 255), -1)
        color = cv2.mean(img, mask=circle_img)[::-1]
        color = np.array([[color[1], color[2], color[3]]])
        colors = np.append(colors, color, axis=0)
    return colors

def dot_distance(dot1, dot2):
    return (math.sqrt((dot2[0] - dot1[0]) ** 2 + (dot2[1] - dot1[1]) ** 2))

def show_BGR(img, colors, lightpoints, label):
    """
    Show blue lighpoints
    """
    print(_bcolors.OKBLUE + "showing: {}".format(label) + _bcolors.ENDC)
    for point, color in zip(lightpoints, colors):
        cv2.circle(img, (point[0], point[1]), 5, color, -1)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def show_RGB(img, colors, lightpoints, label):
    """
    Show actual lightpoints
    """
    print(_bcolors.OKBLUE + "showing: {}".format(label) + _bcolors.ENDC)
    for point, color in zip(lightpoints, colors):
        cv2.circle(img, (point[0], point[1]), 5, color[::-1], -1)
    cv2.imshow("img", img)
    cv2.waitKey(0)

def show_Gray(img, colors, lightpoints, label):
    print(_bcolors.OKBLUE + "showing: {}".format(label) + _bcolors.ENDC)
    for point, color in zip(lightpoints, colors):
        color_rgb = [color, color, color]
        cv2.circle(img, (point[0], point[1]), 5, color_rgb, -1)
    cv2.imshow("img", img)
    cv2.waitKey(0)