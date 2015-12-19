'''
Created on Dec 10, 2015

@author: Dustin
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

IMAGES_DIR = 'C:\\Users\\Dustin\\Dropbox\\FunProjects\\RaspberryPiGo\\StaticGoBoardImages\\'
STONE_IMAGES = ['WhiteStone2.jpg','BlackStone2.jpg']
CORNER_IMAGES = ['TopLeftCorner1.jpg','TopRightCorner1.jpg','BottomLeftCorner1.jpg','BottomRightCorner1.jpg']

BOARD_SIZE = 19

def load_stone_images():
    template = cv2.imread(IMAGES_DIR+'WhiteStone1',0)

def main():
    ###### detect four corners and draw grid overlay
    # read in images of the corners
    tlc_template = cv2.imread(IMAGES_DIR + 'TopLeftCorner1.jpg',0)
    tlc_w, tlc_h = tlc_template.shape[::-1]
    tlc_w_offset = tlc_w / 2
    tlc_h_offset = tlc_h / 2
    trc_template = cv2.imread(IMAGES_DIR + 'TopRightCorner1.jpg',0)
    trc_w, trc_h = trc_template.shape[::-1]
    trc_w_offset = trc_w / 2
    trc_h_offset = trc_h / 2
    blc_template = cv2.imread(IMAGES_DIR + 'BottomLeftCorner1.jpg',0)
    blc_w, blc_h = blc_template.shape[::-1]
    blc_w_offset = blc_w / 2
    blc_h_offset = blc_h / 2
    brc_template = cv2.imread(IMAGES_DIR + 'BottomRightCorner1.jpg',0)
    brc_w, brc_h = brc_template.shape[::-1]
    brc_w_offset = brc_w / 2
    brc_h_offset = brc_h / 2
    
#     cv2.imshow(None,tlc_template)
#     cv2.waitKey(0)
#     cv2.imshow(None,trc_template)
#     cv2.waitKey(0)
#     cv2.imshow(None,blc_template)
#     cv2.waitKey(0)
#     cv2.imshow(None,brc_template)
#     cv2.waitKey(0)
#      
    # read in whole image of board
    img_rgb = cv2.imread(IMAGES_DIR + 'EmptyBoard1.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
     
    tlc_res = cv2.matchTemplate(img_gray,tlc_template,cv2.TM_CCOEFF_NORMED)
    trc_res = cv2.matchTemplate(img_gray,trc_template,cv2.TM_CCOEFF_NORMED)
    blc_res = cv2.matchTemplate(img_gray,blc_template,cv2.TM_CCOEFF_NORMED)
    brc_res = cv2.matchTemplate(img_gray,brc_template,cv2.TM_CCOEFF_NORMED)
     
    threshold = 0.9
    tlc_loc = np.where( tlc_res >= threshold)
    tlc_loc_pt = zip(*tlc_loc[::-1])[0]
    tlc_loc_pt_middle = tuple([tlc_loc_pt[0] + tlc_w_offset,tlc_loc_pt[1] + tlc_h_offset])
    #print(str(tlc_loc_pt))
    #print("new point = "+str(tlc_loc_pt_middle))
    trc_loc = np.where( trc_res >= threshold)
    trc_loc_pt = zip(*trc_loc[::-1])[0]
    trc_loc_pt_middle = tuple([trc_loc_pt[0] + trc_w_offset,trc_loc_pt[1] + trc_h_offset])
    blc_loc = np.where( blc_res >= threshold)
    blc_loc_pt = zip(*blc_loc[::-1])[0]
    blc_loc_pt_middle = tuple([blc_loc_pt[0] + blc_w_offset,blc_loc_pt[1] + blc_h_offset])
    brc_loc = np.where( brc_res >= threshold)
    brc_loc_pt = zip(*brc_loc[::-1])[0]
    brc_loc_pt_middle = tuple([brc_loc_pt[0] + brc_w_offset,brc_loc_pt[1] + brc_h_offset])
     
     
    # draw the boarders of the board's playing field
    cv2.line(img_rgb,tlc_loc_pt_middle,trc_loc_pt_middle,(0,255,0),2)
    cv2.line(img_rgb,trc_loc_pt_middle,brc_loc_pt_middle,(0,255,0),2)
    cv2.line(img_rgb,brc_loc_pt_middle,blc_loc_pt_middle,(0,255,0),2)
    cv2.line(img_rgb,blc_loc_pt_middle,tlc_loc_pt_middle,(0,255,0),2)
     
    # now calculate where the vertical lines on the go board meet the top border,
    # dividing it up into board size columns
    dist_between_top_columns_x = (trc_loc_pt_middle[0] - tlc_loc_pt_middle[0]) / BOARD_SIZE
    dist_between_top_columns_h = (trc_loc_pt_middle[1] - tlc_loc_pt_middle[1]) / BOARD_SIZE
    dist_between_bot_columns_x = (brc_loc_pt_middle[0] - blc_loc_pt_middle[0]) / BOARD_SIZE
    dist_between_bot_columns_h = (brc_loc_pt_middle[1] - blc_loc_pt_middle[1]) / BOARD_SIZE
    print("dist_between_top_columns_x ="+str(dist_between_top_columns_x))
    print("dist_between_top_columns_h ="+str(dist_between_top_columns_h))
    print("dist_between_bot_columns_x ="+str(dist_between_bot_columns_x))
    print("dist_between_bot_columns_h ="+str(dist_between_bot_columns_h))
    
    # now draw all vertical lines in between
    curr_top_x_offset = dist_between_top_columns_x
    curr_top_h_offset = dist_between_top_columns_h
    curr_bot_x_offset = dist_between_bot_columns_x
    curr_bot_h_offset = dist_between_bot_columns_h
    horizontal_scalar = 4
    for i in range(BOARD_SIZE-2): # subtract 2 because border lines already accounted for 
        top_line_pt = tuple([tlc_loc_pt_middle[0]+curr_top_x_offset+(i*horizontal_scalar),tlc_loc_pt_middle[1]+curr_top_h_offset])
        bot_line_pt = tuple([blc_loc_pt_middle[0]+curr_bot_x_offset+(i*horizontal_scalar),blc_loc_pt_middle[1]+curr_bot_h_offset])
        curr_top_x_offset += dist_between_top_columns_x#+(i*horizontal_scalar)
        curr_top_h_offset += dist_between_top_columns_h
        curr_bot_x_offset += dist_between_bot_columns_x#+(i*horizontal_scalar)
        curr_bot_h_offset += dist_between_bot_columns_h
        cv2.line(img_rgb,top_line_pt,bot_line_pt,(0,255,0),2)
     
    cv2.imwrite(IMAGES_DIR+'grid-mapping2.png',img_rgb)

    ###### detect stones
    
#     img_rgb = cv2.imread(IMAGES_DIR + 'MidGame1.jpg')
#     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#     for stone_fn in map(lambda x:IMAGES_DIR+x,STONE_IMAGES):
#          
#         template = cv2.imread(stone_fn,0)
#         w, h = template.shape[::-1]
#  
#         res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#         threshold = 0.85
#         loc = np.where( res >= threshold)
#         for pt in zip(*loc[::-1]):
#             cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#  
#     cv2.imwrite(IMAGES_DIR+'stone-detection2.png',img_rgb)
#     
    ###### edge detection
    #img = cv2.imread(IMAGES_DIR + 'EarlyGame1.jpg',0)
    #edges = cv2.Canny(img,100,200)
    
    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    #plt.show()

if __name__ == '__main__':
    main()