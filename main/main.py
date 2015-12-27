'''
Created on Dec 10, 2015

@author: Dustin
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import cmath
import copy
import sys
import math
import traceback

IMAGES_DIR = 'C:\\Users\\Dustin\\Dropbox\\FunProjects\\RaspberryPiGo\\StaticGoBoardImages\\'
STONE_IMAGES = ['WhiteStone2.jpg','BlackStone2.jpg']
CORNER_IMAGES = ['webcam-tlc1.jpg','webcam-trc1.jpg','webcam-blc1.jpg','webcam-brc1.jpg']
CORNER_POINTS = []#[[139,152],[508,156],[29,432],[634,432]]
INTERSECTION_FN = 'webcam-empty-board-transformed1.jpg'
THRESHOLD = 0.7
BOARD_SIZE = 19

LINES = []

total_clicks = 4
n_clicks = 0
points = []



def dist(x1,y1,x2,y2):
    return cmath.sqrt((pow(abs(x1-x2),2) + pow(abs(y1-y2),2)))

def perp(a):
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

def seg_intersect(img, a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    result = None
    if denom.astype(float) != 0:
        result = (num / denom.astype(float))*db + b1
    
    # do some nice boundary checking
    if result is not None:
        if math.isnan(result[0]) or math.isnan(result[1]) or math.isinf(result[0]) or math.isinf(result[0]) or result[0] < 0 or result[1] < 0 or result[0] > img.shape[1] or result[1] > img.shape[0]:
            return None 
    return result


def dostuff(img):
    img,tlc,trc,blc,brc = transform(img.copy())
    #draw the boarders of the board's playing field
    #cv2.line(img,tuple(tlc),tuple(trc),(0,255,0),2)
    #cv2.line(img,tuple(trc),tuple(brc),(0,255,0),2)
    #cv2.line(img,tuple(brc),tuple(blc),(0,255,0),2)
    #cv2.line(img,tuple(blc),tuple(tlc),(0,255,0),2)
    #img = drawintersections(img)
    #img = cv2.medianBlur(img,5)
    
    # crop image
    if len(CORNER_POINTS) == 4:
        buffer = 10
        x1 = tlc[0] - buffer
        x2 = trc[0] + buffer
        y1 = tlc[1] - buffer
        y2 = blc[1] + buffer
        img = img[y1:y2,x1:x2]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
#     plt.subplot(121),plt.imshow(img,cmap = 'gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#     plt.title('Edge Image'), plt.xticks([]), plt.yticks([])    
#     plt.show()
    #images = []
    lines = cv2.HoughLines(edges,1,np.pi/180,150)
    #print("found "+str(len(lines))+" lines")
    lines_pts = []
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            
            # an attempt to keep all lines within the image: doesn't work as it messes with slope
#             keep_in_width = lambda x: img.shape[1] - 10 if x >= img.shape[1] else x
#             keep_in_height = lambda x: img.shape[0] - 10 if x >= img.shape[0] else x
#             keep_in_origin = lambda x: 10 if x < 0 else x
#             check_x = lambda x: keep_in_origin(keep_in_width(x))
#             check_y = lambda x: keep_in_origin(keep_in_height(x))
#             x1 = check_x(x1)
#             y1 = check_y(y1)
#             x2 = check_x(x2)
#             y2 = check_y(y2)
             
            lines_pts.append([np.array([x1,y1]),np.array([x2,y2])])
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
            
            
    # now filter lines that are too too close to one another (duplicates)
    # for each line, find any where the end points are within a distance of 5 from each other
#     too_close_dist = 5
#     lines_a = [pt for pt in copy.copy(lines_pts)]
#     lines_b = [pt for pt in copy.copy(lines_pts)]
#     for i in range(len(lines_a)):
#         for j in range(len(lines_b)):
#             if i == j:
#                 continue
#             
            
    # find all intersections
    lines_a = [pt for pt in copy.copy(lines_pts)]
    lines_b = [pt for pt in copy.copy(lines_pts)]
    for i in range(len(lines_a)):
        for j in range(len(lines_b)):
            if i == j:
                continue
            lineA, lineB = lines_a[i], lines_b[j]
            result = seg_intersect(img, lineA[0],lineA[1],lineB[0],lineB[1])
            if result is not None: 
                cv2.circle(img,tuple([int(result[0]),int(result[1])]),2,(255,255,255))
            #if result is not None: print(str(result))
        #print("***** NEXT LINE *****")
    #print("-=-=-=-=-DONE-=-=-=-=-=\n\n")
        
    #        images.append(img.copy())
            
    
    #for image in images:
    #    cv2.imshow(None,image)
    #    cv2.waitKey(1000)
    
    return img

def on_mouse_click(event, x, y, flag, param):
    '''
    used to record points from clicking
    '''
    global n_clicks, points
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Point %s captured: (%s,%s)' % (n_clicks+1,x,y)
        points.append([x, y])
        n_clicks += 1

def setcornersbyclicking(img):
    global CORNER_POINTS
    while n_clicks <= total_clicks-1:
        # displays the image
        cv2.imshow("Click", img)
        #cv.ShowImage("Click", cvimage)
        #calls the callback function "on_mouse_click'when mouse is clicked inside window
        cv2.setMouseCallback("Click", on_mouse_click, param=1)
        #cv.SetMouseCallback("Click", on_mouse_click, param=1)
        #cv.WaitKey(1000)
        cv2.waitKey(1000)

    CORNER_POINTS = points
    #return points

def getvideo():
    frames_to_save = 2
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(1)
    
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
        img = frame
        # now perform pattern matching to get corners
        #for frame_count in range(frames_to_save):
        #    cv2.imwrite(IMAGES_DIR+"webcam-empty-board"+str(frame_count)+".jpg", frame)
        
        try:
            setcornersbyclicking(frame)
            img = dostuff(frame)
            pass
        except:
            img = frame
            print "Unexpected error:", sys.exc_info()[0]
            traceback.print_exc()
        
    else:
        rval = False
    
    while rval:
        cv2.imshow("preview", img)
        rval, frame = vc.read()
        try:
            img = dostuff(frame)
            pass
            #cv2.imwrite(IMAGES_DIR+"webcam-empty-board-transformed"+str(frame_count)+".jpg", img)
        except:
            img = frame
            print "Unexpected error:", sys.exc_info()[0]
        
        key = cv2.waitKey(200)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

def load_stone_images():
    template = cv2.imread(IMAGES_DIR+'WhiteStone1',0)

def drawintersections(img):
    intersection_template = cv2.imread(IMAGES_DIR+INTERSECTION_FN,0)
    w, h = intersection_template.shape[::-1]
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img,intersection_template,cv2.TM_CCOEFF_NORMED)
    
    loc = np.where( res >= THRESHOLD)
    print("loc is "+str(loc))
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    return img

def getcornerpts(main_img):
    tlc_template = cv2.imread(IMAGES_DIR + CORNER_IMAGES[0],0)
    tlc_w, tlc_h = tlc_template.shape[::-1]
    tlc_w_offset = tlc_w / 2
    tlc_h_offset = tlc_h / 2
    trc_template = cv2.imread(IMAGES_DIR + CORNER_IMAGES[1],0)
    trc_w, trc_h = trc_template.shape[::-1]
    trc_w_offset = trc_w / 2
    trc_h_offset = trc_h / 2
    blc_template = cv2.imread(IMAGES_DIR + CORNER_IMAGES[2],0)
    #print("blc_template.shape = "+str(blc_template.shape[::-1]))
    blc_w, blc_h = blc_template.shape[::-1]
    #print("blc_w, blc_h = "+str(blc_w)+","+str(blc_h))
    blc_w_offset = blc_w / 2
    blc_h_offset = blc_h / 2
    brc_template = cv2.imread(IMAGES_DIR + CORNER_IMAGES[3],0)
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
    img_rgb = main_img
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
     
    tlc_res = cv2.matchTemplate(img_gray,tlc_template,cv2.TM_CCOEFF_NORMED)
    trc_res = cv2.matchTemplate(img_gray,trc_template,cv2.TM_CCOEFF_NORMED)
    blc_res = cv2.matchTemplate(img_gray,blc_template,cv2.TM_CCOEFF_NORMED)
    brc_res = cv2.matchTemplate(img_gray,brc_template,cv2.TM_CCOEFF_NORMED)
     
    tlc_loc = np.where( tlc_res >= THRESHOLD)
    tlc_loc_pt = zip(*tlc_loc[::-1])[0]
    tlc_loc_pt_middle = tuple([tlc_loc_pt[0] + tlc_w_offset,tlc_loc_pt[1] + tlc_h_offset])
    #print(str(tlc_loc_pt))
    #print("new point = "+str(tlc_loc_pt_middle))
    trc_loc = np.where( trc_res >= THRESHOLD)
    #print(str(trc_loc))
    trc_loc_pt = zip(*trc_loc[::-1])[0]
    trc_loc_pt_middle = tuple([trc_loc_pt[0] + trc_w_offset,trc_loc_pt[1] + trc_h_offset])
    blc_loc = np.where( blc_res >= THRESHOLD)
    blc_loc_pt = zip(*blc_loc[::-1])[0]
    blc_loc_pt_middle = tuple([blc_loc_pt[0] + blc_w_offset,blc_loc_pt[1] + blc_h_offset])
    brc_loc = np.where( brc_res >= THRESHOLD)
    brc_loc_pt = zip(*brc_loc[::-1])[0]
    brc_loc_pt_middle = tuple([brc_loc_pt[0] + brc_w_offset,brc_loc_pt[1] + brc_h_offset])

    return tlc_loc_pt_middle, trc_loc_pt_middle, blc_loc_pt_middle, brc_loc_pt_middle

def transform(img):
    
    rows,cols,ch = img.shape
    
    # get the original four corners via template matching
    #tlc,trc,blc,brc = getcornerpts(img)
    tlc,trc,blc,brc = CORNER_POINTS
    #print("corner_tlc is "+str(tlc))
    #print("corner_trc is "+str(trc))
    #print("corner_blc is "+str(blc))
    #print("corner_brc is "+str(brc))
    origin_x = tlc[0]
    origin_y = tlc[1]
    bottom_y = blc[1]
    right_x = trc[0]
    pts1 = np.float32([tlc,trc,blc,brc])
    #todo make this more robust
    new_tlc = tlc
    new_trc = [right_x,origin_y]
    new_blc = [origin_x,bottom_y]
    new_brc = [right_x,bottom_y]
    #print("new_tlc is "+str(new_tlc))
    #print("new_trc is "+str(new_trc))
    #print("new_blc is "+str(new_blc))
    #print("new_brc is "+str(new_brc))
    pts2 = np.float32([new_tlc,new_trc,new_blc,new_brc])
    #pts1 = np.float32([[248,757],[1309,741],[84,1877],[1449,1886]])
    #pts2 = np.float32([[248,757],[1309,757],[248,1877],[1309,1877]])
     
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(right_x+200,bottom_y+200))
    
    #plt.subplot(121),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(dst),plt.title('Output')
    #plt.show()
    
    return dst,new_tlc,new_trc,new_blc,new_brc
    

def computegrid(img,tlc,trc,blc,brc):
    '''
    Given an image and 4 coordinates representing each corner, return
    a grid of locations corresponding to the Go Board intersections. The
    return value is a 2d array, where element arr[0][0] is the top left corner
    intersection of the board. The value of the array is the pixel location.
    
    Assumption: given img has been transformed to remove any skew, so that the distances
    between columns and rows is consistent. In order to do this, call transform on the
    img before returning it. You need to also return the new coordinates.
    '''
    
    # grid is a mapping of coordinates (0,0) to pixels in the image
    grid = []
    
    col_dist = (trc[0] - tlc[0]) / BOARD_SIZE
    row_dist = (blc[1] - tlc[1]) / BOARD_SIZE
    print("col_dist is "+str(col_dist)+",row_dist is "+str(row_dist))
    origin_x = tlc[0]
    origin_y = tlc[1]
    
    row_y_offset = 0
    
    for r in range(BOARD_SIZE+1):
        row = []
        col_x_offset = 0
        for c in range(BOARD_SIZE+1):
            row.append(tuple([origin_x+col_x_offset,origin_y+row_y_offset]))
            print("just appended point: "+str([origin_x+col_x_offset,origin_y+row_y_offset]))
            col_x_offset += col_dist
        grid.append(row)
        row_y_offset += row_dist
        
    return grid

def showgrid(img, grid):
    '''
    Show the new image with lines draw on it
    '''

    for row in grid:
        for pt in row: # pt is the same as col
            print("drawing circle at pt "+str(pt))
            cv2.circle(img,pt,3,(255,255,255),3)

    plt.subplot(111),plt.imshow(img),plt.title('Input')
    #plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()

def main():
    ###### detect four corners and draw grid overlay
    # read in images of the corners
    img = cv2.imread(IMAGES_DIR + 'MidGame1.jpg')
    print("Transforming image...")
    img,new_tlc,new_trc,new_blc,new_brc = transform(img)
    print("Computing Grid...") 
    grid = computegrid(img, new_tlc, new_trc, new_blc, new_brc)
    print("Drawing image with grid...") 
    showgrid(img, grid) 
     
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
#         THRESHOLD = 0.85
#         loc = np.where( res >= THRESHOLD)
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
    #main()
    #transform()
    getvideo()