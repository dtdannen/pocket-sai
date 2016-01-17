'''
Created on Jan 2, 2016

@author: Dustin
'''

# config variables
ZOOM_ON = True
SAVE_NEG_IMAGES = True
SAVE_WHITE_STONE_IMAGES = True

# Display variables
DRAW_LINES = False

IMAGES_DIR = 'C:\\Users\\Dustin\\Dropbox\\FunProjects\\RaspberryPiGo\\StaticGoBoardImages\\'
NEG_TRAINING_IMAGES_DIR = 'C:\\Users\\Dustin\\Dropbox\\FunProjects\\RaspberryPiGo\\Training\\neg\\'
POS_WHITE_TRAINING_IMAGES_DIR = 'C:\\Users\\Dustin\\Dropbox\\FunProjects\\RaspberryPiGo\\Training\\white\\'
STONE_IMAGES = ['WhiteStone2.jpg','BlackStone2.jpg']
CORNER_IMAGES = ['webcam-tlc1.jpg','webcam-trc1.jpg','webcam-blc1.jpg','webcam-brc1.jpg']
CORNER_POINTS = []#[[139,152],[508,156],[29,432],[634,432]]
PERSPECTIVE_CORNER_POINTS = []
INTERSECTION_FN = 'webcam-empty-board-transformed1.jpg'
THRESHOLD = 0.7
BOARD_SIZE = 19
FRAMES_CAPTURED = 0
INTERSECTIONS = [] # coordinates of intersections on the empty board
#STONES = [] # coordinates of stones on the board

MIN_DIST = 12 # minimum distance between stones and intersections (adjust based on physical board size)
BUFFER = 8

NUM_FRAMES_TO_KEEP = 50

BACKGROUND_IMAGE = None

FRAMES_CAPTURED_THUS_FAR = 0

total_clicks = 4
n_clicks = 0
points = []
curr_white_stone_clicks = 0
total_white_stone_clicks = 81

# dictionary where the key is an intersection and the value is the go board index (i.e. J5)
LABELS = {}

ACTIVE_INTERSECTIONS = [] # each element is an intersection, which is an array of an x,y

INTERSECTION_ACC_INTS = {} # key is str of x,y of coordinate, val is avg intensity of pixels in recetangle of intersection

NUM_FRAMES = 10
LAST_FRAMES = None

MOST_RECENT_IMG = None

STONES = {} # key is move, val is intersection
SINGLE_STONE_FRAME_THRESHOLD = 5
CURR_SINGLE_STONE_FRAME_COUNT = 0

DETECT_STONE_THRESHOLD = 70 # percent of intersection that needs to be white
