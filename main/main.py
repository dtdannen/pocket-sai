'''
Created on Dec 10, 2015

@author: Dustin
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion
import time
import cmath
import copy
import sys
import math
import traceback
from cv2 import pyrUp
import collections
import config
import itertools
from PIL import ImageGrab


def getvideo():
    video_window_name = "Go Board Live Video Stream" 
    cv2.namedWindow(video_window_name)
    cv2.setMouseCallback(video_window_name, mouseclick_show_histogram, param=1)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    fgbg.setDetectShadows(False)
    print(str(dir(fgbg)))
    # complicated, hacky logic, just ignore - only used to collect training data
    #will_save_white_stone_images = False
    #global SAVE_WHITE_STONE_IMAGES
    #if SAVE_WHITE_STONE_IMAGES:
    #    will_save_white_stone_images = True
    #SAVE_WHITE_STONE_IMAGES = False
    # end hacky logic
    
    rval, frame = get_next_frame()
   
    if config.ROTATE_ON: frame = rotate(frame)
    img = frame
    fgmask = fgbg.apply(frame)
    #FRAMES_CAPTURED+=1
    try:
        #print("about to set corners by clicking")
        setcornersbyclicking(frame)
        img = dostuff(frame)
#             if BACKGROUND_IMAGE is None:
#                 #BACKGROUND_IMAGE = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
#                 BACKGROUND_IMAGE = img.copy()
#                 #cv2.imshow("background is",BACKGROUND_IMAGE)
        #fgbg = cv2.createBackgroundSubtractorMOG2()
        if config.ZOOM_ON: 
            display_img = cv2.pyrUp(img.copy())
        else:
            display_img = img.copy()
        pass
    except:
        img = frame
        print "Unexpected error:", sys.exc_info()[0]
        traceback.print_exc()
        raise Exception("meh")
    

    while rval:
        
        cv2.imshow(video_window_name, display_img)
        rval, frame = get_next_frame()
        if config.ROTATE_ON: frame = rotate(frame)
        fgmask = fgbg.apply(frame)
        fgmask = transform_and_crop(fgmask,True)
        if len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
            add_new_stone(active_intersections(fgmask))
        #fgmask = draw_intersections2(fgmask,True)
        if config.ROTATE_ON: fgmask = rotate(fgmask)
        cv2.imshow("background subtraction", fgmask)
        config.FRAMES_CAPTURED+=1
        try:
            img = dostuff(frame)
            #kernel = np.ones((5,5),np.uint8)
            #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            #img = cv2.dilate(img,kernel,iterations = 1)
            display_img = img.copy()
            if len(config.INTERSECTIONS_TO_COORDINATES) == 0 and len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
                map_intersections_to_coordinates(show=True,image=display_img) 
                pass
            #display_avg_color_at_inters(display_img)
            config.MOST_RECENT_IMG = img
            add_img_to_last_frames(img) 
            #save_all_intersections_as_neg_images(img)
            #save_all_intersections_as_white_stones(img)
#             global BACKGROUND_IMAGE 
#             if BACKGROUND_IMAGE is not None:
#                 #cv2.imshow("new_img is",img.copy())
#                 #subtraction_img = BACKGROUND_IMAGE-cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
#                 subtraction_img = BACKGROUND_IMAGE-img.copy()
#                 #subtraction_img = cv2.fastNlMeansDenoisingColored(subtraction_img,None,10,10,7,21)
#                 #subtraction_img = cv2.blur(subtraction_img,(7,7))
#                 subtraction_img = cv2.cvtColor(subtraction_img,cv2.COLOR_BGR2GRAY)
#                 #(thresh, subtraction_img_bw) = cv2.threshold(subtraction_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#                 thresh = 127
#                 subtraction_img_bw = cv2.threshold(subtraction_img, thresh, 255, cv2.THRESH_BINARY)[1]
#                 #circles = cv2.HoughCircles(subtraction_img,cv2.HOUGH_GRADIENT,1,MIN_DIST,
#                 #                param1=50,param2=30,minRadius=int(MIN_DIST/1.5),maxRadius=int(1.2*MIN_DIST))
#                 #backtorgb = cv2.cvtColor(subtraction_img,cv2.COLOR_GRAY2RGB)
#                 #raw_circles = circles
#                 #if circles is not None:
#                 #    circles = np.uint16(np.around(circles))
#                 #    for circle in circles:
#                 #        circle = np.array(circle[0]).tolist()   
#                 #        cv2.circle(backtorgb,(circle[0],circle[1]),circle[2],(0,255,0),2)
#                 #        # draw the center of the circle
#                 #        cv2.circle(backtorgb,(circle[0],circle[1]),2,(0,0,255),3)
#                              
#                 #subtraction_img = cv2.fastNlMeansDenoising()
#                 #ret,th1 = cv2.threshold(subtraction_img,127,255,cv2.THRESH_BINARY)
#                 cv2.imshow("mask",subtraction_img_bw)
            # display circles around stones
#             if len(STONES) > 0:
#                 for i in STONES:
#                     # draw the outer circle
#                     #cir = 
#                     #print("i is "+str(i)+" and has type "+str(type(i)))
#                     cv2.circle(display_img,(i[0],i[1]),i[2],(0,255,0),2)
#                     # draw the center of the circle
#                     cv2.circle(display_img,(i[0],i[1]),2,(0,0,255),3)
            
            # display intersections
#             letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
#             labels = []
#             for num in range(1,20):
#                 for letter in letters:
#                     labels.append(letter+str(num))
#             print(str(labels))
            
            sort_intersections()
            i = 0
            
            update_intersection_intensity_avgs(img)
            
            #display_img = display_intersection_avgs(img)
            display_img = display_intersection_move_number(img)
            
            for inter in config.INTERSECTIONS:
                cv2.rectangle(display_img,tuple([int(inter[0]-6),int(inter[1]-6)]),tuple([int(inter[0]+6),int(inter[1]+6)]),(0,0,255),1)
                #key = str(inter[0])+","+str(inter[1])
                #cv2.putText(img,INTERSECTION_AVG_INTS[key] ,tuple([inter[0]-5,inter[1]+3]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,color=(255,50,50))
                #i+=1
                
            #display_img = check_new_stones(display_img)
            #global LABELS
            #print("LABELS is "+str(LABELS))
            #for label,inter in LABELS.items():
            #    cv2.rectangle(display_img,tuple([int(inter[0]-5),int(inter[1]-5)]),tuple([int(inter[0]+5),int(inter[1]+5)]),(0,0,255),1)
            #    cv2.putText(display_img,label,tuple([inter[0]-5,inter[1]+3]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,color=(50,50,255))
            
            # zoom in 
            if config.ZOOM_ON:
                display_img = cv2.pyrUp(display_img)
            else:
                display_img = display_img
            pass
            #cv2.imwrite(IMAGES_DIR+"webcam-empty-board-transformed"+str(frame_count)+".jpg", img)
        except:
            img = frame
            print "Unexpected error:", sys.exc_info()[0]
            traceback.print_exc()
            raise Exception("meh")
        
      
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
        elif key == ord('w'):
            print("just pressed w")
#             if will_save_white_stone_images:
#                 # this will trigger collection of white stone images
#                 SAVE_WHITE_STONE_IMAGES = True
#             
    cv2.destroyWindow("preview")

def get_next_frame():
    '''
    Returns the next frame. The function takes care
    of handling the actual getting of the frame. If
    the frame is from the webcam its different than
    from the desktop.
    '''

    if config.VIDEO_SOURCE == "WEBCAM":
        if config.WEBCAM_VC is None:
            config.WEBCAM_VC = cv2.VideoCapture(1)
            
        if config.WEBCAM_VC.isOpened(): 
            rval, frame = config.WEBCAM_VC.read() # get the next frame 
            return rval, frame
    elif config.VIDEO_SOURCE == "DESKTOP":
        frame = ImageGrab.grab()
        frame = np.array(frame)
        
        return True, frame
    
    return False, None

def sort_intersections():
    if len(config.PERSPECTIVE_CORNER_POINTS) == 4 and len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
        tlc,trc,blc,brc = config.PERSPECTIVE_CORNER_POINTS
        tlc_inter, trc_inter, blc_inter, brc_inter = get_closest_inter(tlc),get_closest_inter(trc),get_closest_inter(blc),get_closest_inter(brc)
        config.LABELS['A1']=tlc_inter
        config.LABELS['A19']=trc_inter
        config.LABELS['T1']=blc_inter
        config.LABELS['T19']=brc_inter

def get_closest_inter(pt):
    
    min_d = 1000
    min_inter = None
    for inter in config.INTERSECTIONS:
        curr_dist = dist(pt[0],pt[1],inter[0],inter[1]) 
        if curr_dist < min_d:
            min_inter = inter
            min_d = curr_dist
    return min_inter

def dist(x1,y1,x2,y2):
    #print("abs(x1-x2) =" + str(abs(x1-x2)))
    #print("abs(y1-y2) =" + str(abs(y1-y2)))
    result = cmath.sqrt((pow(abs(x1-x2),2) + pow(abs(y1-y2),2)))
    #print("result is "+str(result.real)) 
    return result.real

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

def find_intersections(img):
    if len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
        return img
    elif len(config.INTERSECTIONS) > pow(config.BOARD_SIZE,2):
        raise Exception("Whoops too many intersections")
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray", gray)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,150)    
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
             
            lines_pts.append([np.array([x1,y1]),np.array([x2,y2])])
            if config.DRAW_LINES: cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)      
            
    # find all intersections, while ignoring duplicates
    lines_a = [pt for pt in copy.copy(lines_pts)]
    lines_b = [pt for pt in copy.copy(lines_pts)]
    for i in range(len(lines_a)):
        for j in range(len(lines_b)):
            if i == j:
                continue
            lineA, lineB = lines_a[i], lines_b[j]
            result = seg_intersect(img, lineA[0],lineA[1],lineB[0],lineB[1])
            if result is not None:
                # now compare against all intersections and make sure this one isn't a duplicate
                curr_i = [int(result[0]),int(result[1])]
                duplicate = False
                for prev_i in config.INTERSECTIONS:
                    if dist(prev_i[0], prev_i[1], curr_i[0], curr_i[1]) < config.MIN_DIST:
                        duplicate = True
                if not duplicate:
                    config.INTERSECTIONS.append([int(result[0]),int(result[1])])    
    return img

# def find_stones(img):
#     #global STONES
#     img = cv2.medianBlur(img,5)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 
#     circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,config.MIN_DIST,
#                                 param1=50,param2=30,minRadius=int(config.MIN_DIST/1.5),maxRadius=int(1.2*config.MIN_DIST))
#     
#     raw_circles = circles
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for circle in circles:
#             circle = np.array(circle[0]).tolist()   
#             if circle not in config.STONES:
#                 config.STONES.append(circle)
#                 print("Just added stone: "+str(circle))
#     return img

def crop(img,x1,y1,x2,y2):
    '''
    returns a new img from the given image of the square formed from the coordinates
    '''
    new_img = img.copy()
    new_img = new_img[y1:y2,x1:x2]
    return new_img

def get_num_white_pixels(img,inter,dist=10):
    '''
    ONLY TAKES A BINARY IMAGE
    Returns the number of white pixels within a certain distance of the intersection
    '''

    x_min = inter[0] - (dist / 2)
    x_max = inter[0] + (dist / 2)
    y_min = inter[1] - (dist / 2)
    y_max = inter[1] + (dist / 2)
    num_white_pixels = 0
    num_black_pixels = 0
   
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            if y < int(img.shape[0]) and x < int(img.shape[1]):
                #print("adding pixel for "+str(x)+","+str(y))
                pixel = img[y,x]
                
                if pixel > 200:
                    num_white_pixels+=1
                else:
                    num_black_pixels+=1 
            y+=1
        x+=1
    return num_white_pixels,num_black_pixels

def get_avg_intensity_inter(img,interx,intery):
    # get all surrounding pixels
    dist = 12
    x_min = interx - (dist / 2)
    x_max = interx + (dist / 2)
    y_min = intery - (dist / 2)
    y_max = intery + (dist / 2)
    acc_intensity = 0
    count = 0
    #print("img.shape[0]="+str(img.shape[0])+", "+"img.shape[1]="+str(img.shape[1]))
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            if y < int(img.shape[0]) and x < int(img.shape[1]):
                #print("adding pixel for "+str(x)+","+str(y))
                pixel = img[y,x]
                acc_intensity += (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3 
                count += 1
            y+=1
        x+=1
    avg_for_square = -1
    if count > 0:
        avg_for_square = acc_intensity / count

    return avg_for_square

def display_avg_color_at_inters(img):
    # get all pixels in a square around the intersection
    dist = 12
    #print("img.shape="+str(img.shape))
    if len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
        for inter in config.INTERSECTIONS:
            #print("inter is "+str(inter))
            #if inter[1] < img.shape[0] and inter[0] < img.shape[1]:
            
            # get all surrounding pixels
            x_min = inter[0] - (dist / 2)
            x_max = inter[0] + (dist / 2)
            y_min = inter[1] - (dist / 2)
            y_max = inter[1] + (dist / 2)
            acc_intensity = 0
            count = 0
            #print("img.shape[0]="+str(img.shape[0])+", "+"img.shape[1]="+str(img.shape[1]))
            x = x_min
            while x < x_max:
                y = y_min
                while y < y_max:
                    if y < int(img.shape[0]) and x < int(img.shape[1]):
                        #print("adding pixel for "+str(x)+","+str(y))
                        pixel = img[y,x]
                        acc_intensity += (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) / 3 
                        count += 1
                    y+=1
                x+=1
            avg_for_square = -1
            if count > 0:
                avg_for_square = acc_intensity / count
            txtstr = str(avg_for_square)
            cv2.rectangle(img,tuple([int(inter[0]-(dist / 2)),int(inter[1]-(dist / 2))]),tuple([int(inter[0]+(dist / 2)),int(inter[1]+(dist / 2))]),(0,220,0),1)
            cv2.putText(img,txtstr,tuple([inter[0]-(dist / 2),inter[1]+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,color=(255,50,50))
            #cv2.imshow("nothing",img)
            #cv2.waitKey(1000)
    
    #print("the value at "+str(type(img)))

def update_intersection_intensity_avgs(img):
    if len(config.INTERSECTIONS) == 361:
        config.FRAMES_CAPTURED_THUS_FAR += 1
        # get avg value for each intersection
        for inter in config.INTERSECTIONS:
            curr_avg = get_avg_intensity_inter(img, inter[0], inter[1]) 
            key = str(inter[0]) + "," + str(inter[1])
            # add avg to running accumulator
            #print("wtf - "+str(type(INTERSECTION_ACC_INTS)))
            if key in config.INTERSECTION_ACC_INTS.keys():
                prev_avgs = config.INTERSECTION_ACC_INTS[key]
                if len(prev_avgs) < config.NUM_FRAMES_TO_KEEP:
                    prev_avgs.insert(0, curr_avg)
                    config.INTERSECTION_ACC_INTS[key] = prev_avgs
                else:
                    prev_avgs.pop()
                    prev_avgs.insert(0,curr_avg)
                    config.INTERSECTION_ACC_INTS[key]= prev_avgs
            else:
                config.INTERSECTION_ACC_INTS[key] = [curr_avg]
        
        
    
def display_intersection_avgs(img):
    display_img = img.copy()
    dist = 12
    
    divide_val = config.NUM_FRAMES_TO_KEEP
    if config.FRAMES_CAPTURED < config.NUM_FRAMES_TO_KEEP:
        divide_val = config.FRAMES_CAPTURED
    
    if len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
        for inter in config.INTERSECTIONS:
            key = str(inter[0])+","+str(inter[1])
            val = sum(config.INTERSECTION_ACC_INTS[key])
            val = val / divide_val    
            cv2.putText(display_img,str(val),tuple([inter[0]-(dist / 2),inter[1]+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,color=(255,50,50))
    return display_img

def display_intersection_move_number(img):
    display_img = img.copy()
    dist = 12
    if len(config.INTERSECTIONS_TO_COORDINATES) > 0:
        print("mapping is "+str(config.INTERSECTIONS_TO_COORDINATES))
    if len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2):
        for move_num,inter in config.STONES.items():
            val = str(move_num)
            #inter_str = str(inter[0])+","+str(inter[1])
            if tuple(inter) in config.INTERSECTIONS_TO_COORDINATES.keys():
                alphaCord = config.INTERSECTIONS_TO_COORDINATES[tuple(inter)]
                cv2.putText(display_img,str(alphaCord),tuple([inter[0]-(dist / 2)+3,inter[1]+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,color=(255,255,0))    
            else:
                cv2.putText(display_img,str(val),tuple([inter[0]-(dist / 2)+3,inter[1]+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,color=(50,50,255))
                print("loc = "+str(inter))
    return display_img

def display_intersection_coordinates(img):
    display_img = img.copy()
    dist = 12
    for inter,coord in config.INTERSECTIONS_TO_COORDINATES.items():
        cv2.putText(display_img,str(coord),tuple([inter[0]-(dist / 2)+3,inter[1]+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,color=(255,255,0))
        
    return display_img

def transform_and_crop(img, binary_image=False):
    # transform first
    img,tlc,trc,blc,brc = transform(img.copy(),binary_image=binary_image)
    
    # crop (so we only look at the board)
    if len(config.CORNER_POINTS) == 4:
        x1 = tlc[0] - config.BUFFER
        x2 = trc[0] + config.BUFFER
        y1 = tlc[1] - config.BUFFER
        y2 = blc[1] + config.BUFFER
        img = crop(img,x1,y1,x2,y2)
        
        h,w = img.shape[0], img.shape[1]
    
    return img
        
def dostuff(img):
    # transform first
    img,tlc,trc,blc,brc = transform(img.copy())
    
    # crop (so we only look at the board)
    if len(config.CORNER_POINTS) == 4:
        x1 = tlc[0] - config.BUFFER
        x2 = trc[0] + config.BUFFER
        y1 = tlc[1] - config.BUFFER
        y2 = blc[1] + config.BUFFER
        img = crop(img,x1,y1,x2,y2)
        
        h,w = img.shape[0], img.shape[1]
        
    config.PERSPECTIVE_CORNER_POINTS = [[config.BUFFER,config.BUFFER],[w-config.BUFFER,config.BUFFER],[config.BUFFER,h-config.BUFFER],[w-config.BUFFER,h-config.BUFFER]]
        
    # find the intersections (only runs unless all intersections have been found)
    img = find_intersections(img)
    
    
    #cv2.CascadeClassifier()
    # locate all stones
    #img = find_stones(img)
    
    #draw the boarders of the board's playing field
    #cv2.line(img,tuple(tlc),tuple(trc),(0,255,0),2)
    #cv2.line(img,tuple(trc),tuple(brc),(0,255,0),2)
    #cv2.line(img,tuple(brc),tuple(blc),(0,255,0),2)
    #cv2.line(img,tuple(blc),tuple(tlc),(0,255,0),2)
    #img = drawintersections(img)
    #img = cv2.medianBlur(img,5)
    
    
    
#     plt.subplot(121),plt.imshow(img,cmap = 'gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#     plt.title('Edge Image'), plt.xticks([]), plt.yticks([])    
#     plt.show()
    #images = []
    
    #print("found "+str(len(lines))+" lines")
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
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Point %s captured: (%s,%s)' % (config.n_clicks+1,x,y)
        config.points.append([x, y])
        config.n_clicks += 1

def setcornersbyclicking(img):
    while config.n_clicks <= config.total_clicks-1:
        # displays the image
        cv2.imshow("Click", img)
        #cv.ShowImage("Click", cvimage)
        #calls the callback function "on_mouse_click'when mouse is clicked inside window
        cv2.setMouseCallback("Click", on_mouse_click, param=1)
        #cv.SetMouseCallback("Click", on_mouse_click, param=1)
        #cv.WaitKey(1000)
        cv2.waitKey(1000)

    config.CORNER_POINTS = config.points
    #return points


def mouseclick_show_histogram(event, x, y, flag, param):
    '''
    used to record points from clicking
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        
        #print("eventchecks out")
        inter = get_closest_inter([x,y])
        # get image around inter
        dist = 6
        x1 = inter[0] - (dist / 2)
        y1 = inter[1] - (dist / 2)
        x2 = inter[0] + (dist / 2)
        y2 = inter[1] + (dist / 2)
                 
        img = crop(config.MOST_RECENT_IMG,x1,y1,x2,y2)
        #print("just cropped img")
        
        color = ('b','g','r')
        
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.ylim([0,25])
        plt.savefig('../histograms/inter_'+str(inter[0])+"_"+str(inter[1])+'_'+str(time.strftime("%H_%M_%S", time.gmtime()))+'.png', bbox_inches='tight')
        plt.close()
        
#         plt.hist(img.ravel(),256,[0,256])
#         plt.show()
        #print("just showed it")
                    
def rotate(img):
    img = img.copy()
    rows,cols = None,None
    try:
        rows,cols,ch = img.shape
    except:
        rows,cols = img.shape
        
    M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
    dst = cv2.warpAffine(img,M,(rows+100,cols+100))
    return dst
                    
def add_img_to_last_frames(img): 
    
    if config.LAST_FRAMES is None:
        # initialize LAST_FRAMES
        starting_frames = []
        for i in range(config.NUM_FRAMES):
            starting_frames.append(img.copy())
        LAST_FRAMES = collections.deque(starting_frames,config.NUM_FRAMES)                   
    else:
        LAST_FRAMES.appendleft(img)

def get_last_average_img():
    if config.LAST_FRAMES is not None:
        config.LAST_FRAMES

# def check_new_stones(img):
#     change_threshold = 70
#     
#     # get curr_val for each intersection
#     display_img = img.copy()
#     dist = 12
#     #print("num saved intersection data is "+str(len(INTERSECTION_ACC_INTS.values()[0]))+" and frames captured is "+str(FRAMES_CAPTURED))
#     if len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2) and len(config.INTERSECTION_ACC_INTS.values()[0]) == config.NUM_FRAMES_TO_KEEP and config.FRAMES_CAPTURED > config.NUM_FRAMES_TO_KEEP*2:
#         for inter in config.INTERSECTIONS:
#             curr_avg = get_avg_intensity_inter(img, inter[0], inter[1])
#              
#             # get prev_avg
#             key = str(inter[0])+","+str(inter[1])
#             prev_avg = sum(config.INTERSECTION_ACC_INTS[key])
#             prev_avg = prev_avg / config.NUM_FRAMES_TO_KEEP
#             
#             if abs(prev_avg-curr_avg) > change_threshold:
#                 #print("prev avg was "+str(prev_avg)+" and curr_avg is "+str(curr_avg))
#                 config.STONES.append([inter[0],inter[1]])
#                 cv2.circle(display_img,tuple([inter[0],inter[1]]),5,(0,255,0),2)
#     return display_img

def simple_get_video_bg_subtract():
    video_window_name = "Go Board Live Video Stream" 
    cv2.namedWindow(video_window_name)
    vc = cv2.VideoCapture(1)
    cv2.setMouseCallback(video_window_name, mouseclick_show_histogram, param=1)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    #cv2.imshow(video_window_name, display_img)
    
    while(1):
        ret, frame = vc.read()
    
        fgmask = fgbg.apply(frame)
        #fgmask_gray = cv2.cvtColor() 
        circles = cv2.HoughCircles(fgmask,cv2.HOUGH_GRADIENT,1,config.MIN_DIST,
                                param1=50,param2=30,minRadius=int(config.MIN_DIST/1.5),maxRadius=int(1.2*config.MIN_DIST))
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles:
                circle = np.array(circle[0]).tolist()
                cv2.circle(frame,(circle[0],circle[1]),circle[2],(0,255,0),2)
            cv2.imshow("Circles",frame)
        
        cv2.imshow(video_window_name,fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    
    vc.release()
    cv2.destroyAllWindows()

def draw_intersections2(display_img,binary_image=False):
    display_img = display_img.copy()
    for inter in config.INTERSECTIONS:
        if binary_image:
            cv2.rectangle(display_img,tuple([int(inter[0]-6),int(inter[1]-6)]),tuple([int(inter[0]+6),int(inter[1]+6)]),(255,255,255),1)
        else:
            cv2.rectangle(display_img,tuple([int(inter[0]-6),int(inter[1]-6)]),tuple([int(inter[0]+6),int(inter[1]+6)]),(0,0,255),1)
        #key = str(inter[0])+","+str(inter[1])
        #cv2.putText(img,INTERSECTION_AVG_INTS[key] ,tuple([inter[0]-5,inter[1]+3]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2,color=(255,50,50))
        #i+=1
              
    return display_img

def active_intersections(img):
    '''
    Checks binary image within each intersection, if enough pixels are white, that
    intersection is 'activated'
    '''
    # IMPORTANT ASSUMPTION: img is binary
    active_inters = []
    for inter in config.INTERSECTIONS:
        # get the all the pixels in this intersection
        num_white_pixels, num_black_pixels = get_num_white_pixels(img,inter)
        percent = (num_white_pixels*1.0) / (1.0*(num_white_pixels+num_black_pixels))
        percent *= 100
        if percent > config.DETECT_STONE_THRESHOLD:
            #print("num_white is "+str(num_white_pixels)+" num black "+str(num_black_pixels))
            #print("percent is "+str(percent))
            active_inters.append(inter)
            #print("num_white_pixels = "+str(num_white_pixels))
    
    return active_inters

def add_new_stone(active_inters):
    '''
    To be called each frame, if there is one active intersection for config.STONE_FRAME_THRESHOLD
    then add a new stone!
    '''
    #if config.CURR_SINGLE_STONE_FRAME_COUNT > 5:
    #    print("CURR_SINGLE_STONE_FRAME_COUNT = "+str(config.CURR_SINGLE_STONE_FRAME_COUNT))
    if len(active_inters) == 1:
        config.CURR_SINGLE_STONE_FRAME_COUNT +=1
        
        if config.CURR_SINGLE_STONE_FRAME_COUNT >= config.SINGLE_STONE_FRAME_THRESHOLD:
            if len(config.STONES) == 0:
                config.STONES[1] = active_inters[0]
                print("just added stone "+str(active_inters[0]))
            else:
                if not active_inters[0] in config.STONES.values(): 
                    last_move_number = max(config.STONES.keys())
                    config.STONES[last_move_number+1] = active_inters[0]
                    print("just added stone "+str(active_inters[0]))

    else:
        config.CURR_SINGLE_STONE_FRAME_COUNT = 0 # reset

def map_intersections_to_coordinates(show=False,image=None):
    '''
    Letters go across the top from A to T, left to right
    Numbers go from top to bottom, 19 to 1
    
    If show is true and image is an actual image,
    then we will display images showing the intersections and labels as we process them.
    '''
    
    
    # This function isn't the most efficient, but only runs once
    dist = 12
    # double check
    if not (len(config.INTERSECTIONS_TO_COORDINATES) == 0 and len(config.INTERSECTIONS) == pow(config.BOARD_SIZE,2)):
        return # do nothing
    
    # find the top left intersection
    curr_x = 10000 # x max
    curr_y = 10000 # y max
    for (inter_x,inter_y) in config.INTERSECTIONS:
        if inter_x < curr_x: # if smaller
            curr_x = inter_x # assign
        if inter_y < curr_y: # if smaller
            curr_y = inter_y # assign
            
    # now we have the top left intersection
    # give the top left intersection its correct label 
    config.INTERSECTIONS_TO_COORDINATES[(curr_x,curr_y)] = 'A19'
    if show:
        label = config.INTERSECTIONS_TO_COORDINATES[(curr_x,curr_y)]
        print("label is "+str(label)+" curr_x is "+str(curr_x) + " curr_y is "+str(curr_y))
        cv2.putText(image,label,tuple([curr_x-(dist / 2),curr_y+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,color=(50,190,50))
        cv2.imshow("labeling intersections", image)
    #time.sleep(5)
    #key = str(curr_x)+","+str(curr_y)
    
    
    # store the top left, we will start our search based on it
    curr_inter = tuple([curr_x,curr_y])
    next_row_start_inter = curr_inter
    inters_left = copy.copy(config.INTERSECTIONS) # remaining intersections that need to be assigned
    
    
    keep_going_x = True
    keep_going_y = True
    next_inter_x = tuple(get_next_closest_inter_x(curr_inter))
    print("next_inter_x = "+str(next_inter_x))
    inters_left_in_this_row = config.BOARD_SIZE 
    min_dist_btwn_inters = 10 # needed as a margin
    
    letters = ['A','B','C','D','E','F','G','H','J','K','L','M','N','O','P','Q','R','S','T']
    curr_letter_i = 1 # column
    curr_num = 19 # row
    while next_inter_x and curr_num > 0:
        config.INTERSECTIONS_TO_COORDINATES[next_inter_x] = letters[curr_letter_i]+str(curr_num)
        #del inters_left[next_inter_x]
        
        if show:
            label = config.INTERSECTIONS_TO_COORDINATES[next_inter_x]
            print("label is "+str(label)+" inter is "+str(next_inter_x))
            cv2.putText(image,label,tuple([next_inter_x[0]-(dist / 2),next_inter_x[1]+(dist / 4)]),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,color=(50,190,50))
            cv2.imshow("labeling intersections", image)
            cv2.waitKey(0)
        
        
        curr_letter_i = curr_letter_i + 1
        if curr_letter_i == len(letters) and curr_num == 1:
            return
        elif curr_letter_i == len(letters):
            curr_letter_i = 0
            curr_num = curr_num - 1
            next_row_start_inter = tuple(get_next_closest_inter_y(next_row_start_inter))
            next_inter_x = next_row_start_inter
        else:
            next_inter_x = tuple(get_next_closest_inter_x(next_inter_x))
        
    
#     
#     # process the first row
#     for i in range(1,inters_left_in_this_row): # start at 1 because we already processed top left
#         # get the next closest inter pixel coord that has smallest x value and smallest y value
#         for 
#     
#     # do a custom sort for all the intersections
#     # sort based on their x and y values
#     go_coords = []
#     letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T']
#     for i in itertools.product(letters,range(1,config.BOARD_SIZE+1)):
#         go_coords.append(i)
#     
#     all_inter_pairs = []
#     for i in itertools.product(config.INTERSECTIONS,config.INTERSECTIONS):
#         if i[0] != i[1]:
#             all_inter_pairs.append(i)
#              
#     min_x_btwn = min([abs(a[0]-b[0]) for a,b in all_inter_pairs])
#     min_y_btwn = min([abs(a[1]-b[1]) for a,b in all_inter_pairs])
#     
#     # get all inters in the first column
#     curr_column_inters = []
#     for inter in config.INTERSECTIONS:
#         pass
#                          
#     # process first column
#     
#     # find the next smallest x only
#     
#     
#     
#     #cust_sort_func = lambda x: (int(x.split(",")[1])*state.dim) + int(x.split(",")[0])
#     
#         
#     curr_mapping = {}
#     go_coords_i = 0
#     for inter in config.INTERSECTIONS:
#         x = inter[0] # x value
#         y = inter[1] # y value
#         # now assign this intersection the next available mapping
#         curr_inter = tuple([x,y])
#         curr_mapping[curr_inter] = go_coords[go_coords_i]
#         go_coords_i += 1
#     
#     # now do sorting:
#     # go column by column and swap if possible
#     # numbers are rows, letters are columns
#     for col in letters:
#         col_mapping = {k:v for k,v in curr_mapping.items() if col in v}
#         # now col_mappings has only the mappings for this letter
#         # check to see if any of these can be swapped
#         min_inter_y = min([y for x,y in col_mapping.keys()])
#         #min_go_num = 
#         for inter,go_coord in col_mapping:
#             pass
        
#     while keep_going_x or keep_going_y:
#         if keep_going_x:
#             next_inter_x = get_next_closest_inter_x(curr_inter, inters_left)
#             if next_inter_x:
#                 del inters_left[next_inter_x]
#                 keep_going_x = True
#             else:
#                 keep_going_x = False
#             
            

#     del inters_left[(curr_x,curr_y)]
#     curr_inter = tuple([curr_x,curr_y])
#     while len(config.INTERSECTIONS_TO_COORDINATES) < len(config.INTERSECTIONS):
#         
#     
#     
#     print(str(config.INTERSECTIONS_TO_COORDINATES))
#     
def get_next_closest_inter_x(start_inter):
    '''
    This function will always be looking for the right most intersection to the current intersection
    '''
    
    max_y_dist = 8 # ensure we only look at intersections in this row
    
    # returns the next closest intersection along the x axis
    closest_inter = False
    closest_x_so_far = 50
    end_y_dist = -1
    for curr_inter in config.INTERSECTIONS:
        y_dist = abs(curr_inter[1] - start_inter[1])
        x_dist = abs(curr_inter[0] - start_inter[0])
        if curr_inter[0] > start_inter[0] and x_dist < closest_x_so_far and y_dist < max_y_dist:
            closest_inter = curr_inter
            closest_x_so_far = x_dist
            end_y_dist = y_dist
            print("closest_x_so_far is "+str(closest_x_so_far))
    print("y_dist is "+str(end_y_dist))
    return closest_inter

def get_next_closest_inter_y(start_inter):
    '''
    This function will always be looking for the right most intersection to the current intersection
    '''
    
    max_x_dist = 8 # ensure we only look at intersections in this row
    
    # returns the next closest intersection along the x axis
    closest_inter = False
    closest_y_so_far = 50
    end_x_dist = -1
    for curr_inter in config.INTERSECTIONS:
        y_dist = abs(curr_inter[1] - start_inter[1])
        x_dist = abs(curr_inter[0] - start_inter[0])
        if curr_inter[1] > start_inter[1] and y_dist < closest_y_so_far and x_dist < max_x_dist:
            closest_inter = curr_inter
            closest_y_so_far = y_dist
            end_x_dist = x_dist
            print("closest_y_so_far is "+str(closest_y_so_far))
    print("y_dist is "+str(end_x_dist))
    return closest_inter

def load_stone_images():
    template = cv2.imread(config.IMAGES_DIR+'WhiteStone1',0)

def drawintersections(img):
    intersection_template = cv2.imread(config.IMAGES_DIR+config.INTERSECTION_FN,0)
    w, h = intersection_template.shape[::-1]
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img,intersection_template,cv2.TM_CCOEFF_NORMED)
    
    loc = np.where( res >= config.THRESHOLD)
    print("loc is "+str(loc))
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    
    return img

def getcornerpts(main_img):
    tlc_template = cv2.imread(config.IMAGES_DIR + config.CORNER_IMAGES[0],0)
    tlc_w, tlc_h = tlc_template.shape[::-1]
    tlc_w_offset = tlc_w / 2
    tlc_h_offset = tlc_h / 2
    trc_template = cv2.imread(config.IMAGES_DIR +config.CORNER_IMAGES[1],0)
    trc_w, trc_h = trc_template.shape[::-1]
    trc_w_offset = trc_w / 2
    trc_h_offset = trc_h / 2
    blc_template = cv2.imread(config.IMAGES_DIR + config.CORNER_IMAGES[2],0)
    #print("blc_template.shape = "+str(blc_template.shape[::-1]))
    blc_w, blc_h = blc_template.shape[::-1]
    #print("blc_w, blc_h = "+str(blc_w)+","+str(blc_h))
    blc_w_offset = blc_w / 2
    blc_h_offset = blc_h / 2
    brc_template = cv2.imread(config.IMAGES_DIR + config.CORNER_IMAGES[3],0)
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
     
    tlc_loc = np.where( tlc_res >= config.THRESHOLD)
    tlc_loc_pt = zip(*tlc_loc[::-1])[0]
    tlc_loc_pt_middle = tuple([tlc_loc_pt[0] + tlc_w_offset,tlc_loc_pt[1] + tlc_h_offset])
    #print(str(tlc_loc_pt))
    #print("new point = "+str(tlc_loc_pt_middle))
    trc_loc = np.where( trc_res >= config.THRESHOLD)
    #print(str(trc_loc))
    trc_loc_pt = zip(*trc_loc[::-1])[0]
    trc_loc_pt_middle = tuple([trc_loc_pt[0] + trc_w_offset,trc_loc_pt[1] + trc_h_offset])
    blc_loc = np.where( blc_res >= config.THRESHOLD)
    blc_loc_pt = zip(*blc_loc[::-1])[0]
    blc_loc_pt_middle = tuple([blc_loc_pt[0] + blc_w_offset,blc_loc_pt[1] + blc_h_offset])
    brc_loc = np.where( brc_res >= config.THRESHOLD)
    brc_loc_pt = zip(*brc_loc[::-1])[0]
    brc_loc_pt_middle = tuple([brc_loc_pt[0] + brc_w_offset,brc_loc_pt[1] + brc_h_offset])

    return tlc_loc_pt_middle, trc_loc_pt_middle, blc_loc_pt_middle, brc_loc_pt_middle

def transform(img,binary_image=False):
    rows,cols,ch = None,None,None
    if binary_image:
        rows,cols = img.shape
    else:
        rows,cols,ch = img.shape
            
    
    # get the original four corners via template matching
    #tlc,trc,blc,brc = getcornerpts(img)
    tlc,trc,blc,brc = config.CORNER_POINTS
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
    
    col_dist = (trc[0] - tlc[0]) / config.BOARD_SIZE
    row_dist = (blc[1] - tlc[1]) / config.BOARD_SIZE
    print("col_dist is "+str(col_dist)+",row_dist is "+str(row_dist))
    origin_x = tlc[0]
    origin_y = tlc[1]
    
    row_y_offset = 0
    
    for r in range(config.BOARD_SIZE+1):
        row = []
        col_x_offset = 0
        for c in range(config.BOARD_SIZE+1):
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

# def main():
#     ###### detect four corners and draw grid overlay
#     # read in images of the corners
#     img = cv2.imread(config.IMAGES_DIR + 'MidGame1.jpg')
#     print("Transforming image...")
#     img,new_tlc,new_trc,new_blc,new_brc = transform(img)
#     print("Computing Grid...") 
#     grid = computegrid(img, new_tlc, new_trc, new_blc, new_brc)
#     print("Drawing image with grid...") 
#     showgrid(img, grid) 
#      
#     # draw the boarders of the board's playing field
#     cv2.line(img_rgb,tlc_loc_pt_middle,trc_loc_pt_middle,(0,255,0),2)
#     cv2.line(img_rgb,trc_loc_pt_middle,brc_loc_pt_middle,(0,255,0),2)
#     cv2.line(img_rgb,brc_loc_pt_middle,blc_loc_pt_middle,(0,255,0),2)
#     cv2.line(img_rgb,blc_loc_pt_middle,tlc_loc_pt_middle,(0,255,0),2)
#      
#     # now calculate where the vertical lines on the go board meet the top border,
#     # dividing it up into board size columns
#     dist_between_top_columns_x = (trc_loc_pt_middle[0] - tlc_loc_pt_middle[0]) / BOARD_SIZE
#     dist_between_top_columns_h = (trc_loc_pt_middle[1] - tlc_loc_pt_middle[1]) / BOARD_SIZE
#     dist_between_bot_columns_x = (brc_loc_pt_middle[0] - blc_loc_pt_middle[0]) / BOARD_SIZE
#     dist_between_bot_columns_h = (brc_loc_pt_middle[1] - blc_loc_pt_middle[1]) / BOARD_SIZE
#     print("dist_between_top_columns_x ="+str(dist_between_top_columns_x))
#     print("dist_between_top_columns_h ="+str(dist_between_top_columns_h))
#     print("dist_between_bot_columns_x ="+str(dist_between_bot_columns_x))
#     print("dist_between_bot_columns_h ="+str(dist_between_bot_columns_h))
#     
#     # now draw all vertical lines in between
#     curr_top_x_offset = dist_between_top_columns_x
#     curr_top_h_offset = dist_between_top_columns_h
#     curr_bot_x_offset = dist_between_bot_columns_x
#     curr_bot_h_offset = dist_between_bot_columns_h
#     horizontal_scalar = 4
#     for i in range(BOARD_SIZE-2): # subtract 2 because border lines already accounted for 
#         top_line_pt = tuple([tlc_loc_pt_middle[0]+curr_top_x_offset+(i*horizontal_scalar),tlc_loc_pt_middle[1]+curr_top_h_offset])
#         bot_line_pt = tuple([blc_loc_pt_middle[0]+curr_bot_x_offset+(i*horizontal_scalar),blc_loc_pt_middle[1]+curr_bot_h_offset])
#         curr_top_x_offset += dist_between_top_columns_x#+(i*horizontal_scalar)
#         curr_top_h_offset += dist_between_top_columns_h
#         curr_bot_x_offset += dist_between_bot_columns_x#+(i*horizontal_scalar)
#         curr_bot_h_offset += dist_between_bot_columns_h
#         cv2.line(img_rgb,top_line_pt,bot_line_pt,(0,255,0),2)
#      
#     cv2.imwrite(IMAGES_DIR+'grid-mapping2.png',img_rgb)
# 
#     ###### detect stones
#     
# #     img_rgb = cv2.imread(IMAGES_DIR + 'MidGame1.jpg')
# #     img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# #     for stone_fn in map(lambda x:IMAGES_DIR+x,STONE_IMAGES):
# #          
# #         template = cv2.imread(stone_fn,0)
# #         w, h = template.shape[::-1]
# #  
# #         res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# #         THRESHOLD = 0.85
# #         loc = np.where( res >= THRESHOLD)
# #         for pt in zip(*loc[::-1]):
# #             cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# #  
# #     cv2.imwrite(IMAGES_DIR+'stone-detection2.png',img_rgb)
# #     
#     ###### edge detection
#     #img = cv2.imread(IMAGES_DIR + 'EarlyGame1.jpg',0)
#     #edges = cv2.Canny(img,100,200)
#     
#     #plt.subplot(121),plt.imshow(img,cmap = 'gray')
#     #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#     #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#     
#     #plt.show()

if __name__ == '__main__':
    #main()
    #transform()
    getvideo()
    #simple_get_video_bg_subtract()