'''
Created on Jan 2, 2016

@author: Dustin
'''


def save_all_interesections_as_white_stones(img):
    '''
    After the intersections have been properly identified, this will crop the image into
    361 individual images, one for each intersection.
    '''
    w, h = 18,18
    count = 0
    global SAVE_NEG_IMAGES, INTERSECTIONS 
    if SAVE_WHITE_STONE_IMAGES:
        if len(INTERSECTIONS) == pow(BOARD_SIZE,2):
            for inter in INTERSECTIONS:
                x1 = inter[0] - (w / 2)
                y1 = inter[1] - (h / 2)
                x2 = inter[0] + (w / 2)
                y2 = inter[1] + (h / 2)
                cv2.imwrite(POS_WHITE_TRAINING_IMAGES_DIR+"white_"+str(count)+".jpg", crop(img,x1,y1,x2,y2))
                count+=1
            print("Just produced "+ str(count)+" white stone images")
            SAVE_WHITE_STONE_IMAGES = False


def save_all_intersections_as_neg_images(img):
    '''
    After the intersections have been properly identified, this will crop the image into
    361 individual images, one for each intersection.
    '''
    w, h = 18,18
    count = 0
    global SAVE_NEG_IMAGES, INTERSECTIONS 
    if SAVE_NEG_IMAGES:
        if len(INTERSECTIONS) == pow(BOARD_SIZE,2):
            for inter in INTERSECTIONS:
                x1 = inter[0] - (w / 2)
                y1 = inter[1] - (h / 2)
                x2 = inter[0] + (w / 2)
                y2 = inter[1] + (h / 2)
                cv2.imwrite(NEG_TRAINING_IMAGES_DIR+"neg_"+str(count)+".jpg", crop(img,x1,y1,x2,y2))
                count+=1
            print("Just produced "+ str(count)+" negative images")
            SAVE_NEG_IMAGES = False

def save_all_intersections_as_white_stones(img):
    '''
    After the intersections have been properly identified, this will crop the image into
    361 individual images, one for each intersection.
    '''
    w, h = 14,14
    count = 0
    global SAVE_WHITE_STONE_IMAGES, INTERSECTIONS 
    if SAVE_WHITE_STONE_IMAGES:
        if len(INTERSECTIONS) == pow(BOARD_SIZE,2):
            for inter in INTERSECTIONS:
                x1 = inter[0] - (w / 2)
                y1 = inter[1] - (h / 2)
                x2 = inter[0] + (w / 2)
                y2 = inter[1] + (h / 2)
                cv2.imwrite(POS_WHITE_TRAINING_IMAGES_DIR+"white_"+str(count)+".jpg", crop(img,x1,y1,x2,y2))
                count+=1
            print("Just produced "+ str(count)+" white stone images")
            SAVE_WHITE_STONE_IMAGES = False     
