import cv2
import face_recognition
from PIL import Image

image_name = 'faces\\square.jpg'

def dist(point1, point2):
    """Calculates the distance between two points.

    Parameters
    ----------
    point1 : tuple
        The first point
    point2 : tuple
        The second point

    Returns
    -------
    int
        The distance between the two points
    """
    x_squared = (point1[0] - point2[0]) ** 2
    y_squared = (point1[1] - point2[1]) ** 2
    return (x_squared + y_squared) ** (1/2)

def find_forehead_line_y():
    """Finds the coordinates of the forehead guideline.
    
    Uses haarcascade to identify a face in an image and places a square around
    the face. The y value of the top edge of the square is used as the y value
    of the forehead guideline.

    Returns
    -------
    int
        The y coordinate of the top edge of the square
    """
    # loads the classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # reads image
    img = cv2.imread(image_name)
    # converts image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detects the face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # gets the y value of the forehead line
    return faces[0][1]

def get_guideline_coordinates():
    """Gets the end points of the forehead, face length, ear, and jaw
    guidelines.
    
    Uses the face_recognition library to identify facial landmarks on a face in
    an image, and the coordinates of certain facial landmarks are recorded.

    Returns
    -------
    list
        A list of tuples containing the end points of the forehead, face length,
        ear, and jaw guidelines. 
    """

    # load image
    input_image = face_recognition.load_image_file(image_name)
    # generate facial landmarks
    face_landmarks = face_recognition.face_landmarks(input_image)
    
    # get all chin landmark coordinates
    chin_landmark = face_landmarks[0].get('chin')

    # gets the x value of the first chin landmark
    forehead_line_x1 = chin_landmark[2][0]
    # gets the x value of the last chin landmark
    forehead_line_x2 = chin_landmark[14][0]
    # get the y value of the forhead line
    forehead_line_y = find_forehead_line_y()

    # get forehead line end points
    forehead_line_x1_y = (forehead_line_x1, forehead_line_y)
    forehead_line_x2_y = (forehead_line_x2, forehead_line_y)
    #debugging
    print(f'forhead line p1: {forehead_line_x1_y}, forhead line p2: {forehead_line_x2_y}')
    # get face length end points
    face_length_line_x_y1 = (chin_landmark[8][0], forehead_line_y)
    face_length_line_x_y2 = (chin_landmark[8][0], chin_landmark[8][1])
    # get ear line end points
    ear_line_x1_y1 = chin_landmark[2]
    ear_line_x2_y2 = chin_landmark[14]
    #debugging
    print(f'ear line p1: {ear_line_x1_y1}, ear line p2: {ear_line_x2_y2}')
    # get jaw line end points
    jaw_line_x1_y1 = chin_landmark[4]
    jaw_line_x2_y2 = chin_landmark[12]
    print(f'jaw line p1: {jaw_line_x1_y1}, jaw line p2: {jaw_line_x2_y2}')

    return [forehead_line_x1_y, forehead_line_x2_y, face_length_line_x_y1,\
            face_length_line_x_y2, ear_line_x1_y1, ear_line_x2_y2,\
            jaw_line_x1_y1, jaw_line_x2_y2]

def get_guideline_lengths():
    """Calculates the lengths of the forehead, face length, ear, and jaw
    guidelines using the distance formula.
    
    Returns
    -------
    list
        A list of integers that are the lengths of all the guidelines.
    """
    # getting the coordinates of the guidelines
    guidelines = get_guideline_coordinates()
    
    # finding the distance between the coordinates
    forehead_line_length = dist(guidelines[0], guidelines[1])
    face_length_line = dist(guidelines[2], guidelines[3])
    ear_line_length = dist(guidelines[4], guidelines[5])
    jaw_line_length = dist(guidelines[6], guidelines[7])

    return [forehead_line_length, face_length_line, ear_line_length,\
            jaw_line_length]

def find_face_type():
    """The main function. Given an image with a single human face in it, finds
    the face type of the person.
    """
    # get the lengths of the guidelines
    guideline_lengths = get_guideline_lengths()

    forehead_line_length = guideline_lengths[0]
    face_length = guideline_lengths[1]
    ear_line_length = guideline_lengths[2]
    jaw_line_length = guideline_lengths[3]

    # the largest of the three guidelines becoemes the face width
    face_width = max(forehead_line_length, ear_line_length, jaw_line_length)

    tolerance = 40
    # debugging
    print(f'____________________________\n{image_name}\n')

    print(f'Forehead line length: {forehead_line_length}\nEar line length: ' + 
            f'{ear_line_length}\nJaw line length: {jaw_line_length}\n' +
            f'Face length: {face_length}\n')

    # primary trait conditions
    if forehead_line_length <= ear_line_length and\
            ear_line_length > jaw_line_length:
        print('You have an oval face')
    elif abs(forehead_line_length - jaw_line_length) <= tolerance and\
            abs(ear_line_length - jaw_line_length) <= tolerance:
        print('You have a square face')
    elif forehead_line_length - jaw_line_length >= tolerance and\
            ear_line_length - jaw_line_length >= tolerance:
        print('You have a diamond face')
    
    # secondary trait conditions
    if face_width/face_length <= 0.8:
        print('You have an oblong face')
    elif face_width/face_length >= 0.9:
        print('You have a round face')
    

    #debugging
    guidelines = get_guideline_coordinates()        

    image = cv2.imread(image_name)
    window_name = 'Image'
    color = (0, 255, 0)
    thickness = 4
    t1 = (int(guidelines[0][0]), int(guidelines[0][1]))
    p1 = (int(guidelines[1][0]), int(guidelines[1][1]))
    t2 = (int(guidelines[2][0]), int(guidelines[2][1]))
    p2 = (int(guidelines[3][0]), int(guidelines[3][1]))
    t3 = (int(guidelines[4][0]), int(guidelines[4][1]))
    p3 = (int(guidelines[5][0]), int(guidelines[5][1]))
    t4 = (int(guidelines[6][0]), int(guidelines[6][1]))
    p4 = (int(guidelines[7][0]), int(guidelines[7][1]))

    image = cv2.line(image, t1, p1, color, thickness)
    image = cv2.line(image, t2, p2, color, thickness)
    image = cv2.line(image, t3, p3, color, thickness)
    image = cv2.line(image, t4, p4, color, thickness)
    image = cv2.line(image, (10, 10), (50, 10), (255, 0, 0), thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey()

find_face_type()