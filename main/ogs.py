'''
Created on Dec 24, 2015

@author: Dustin
'''

from PIL import ImageGrab
import cv2
import numpy as np
from matplotlib import pyplot as plt
import win32file, win32api, win32con
import os
import time

def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)

def flip(x,y):
    '''
    Given a point that assumes the origin is bottom, 
    '''

class OGSBot():
    '''
    This class is all about creating and playing a match on OGS (online-go.com)
    '''

    OGS_IMAGES_DIR = 'C:\\Users\\Dustin\\Dropbox\\FunProjects\\RaspberryPiGo\\OGSBotImages\\'

    def __init__(self):
        '''
        Constructor
        '''
        self.delay = 0.3
        
    def open_chrome(self):
        # get current image
        img = ImageGrab.grab()
        img_h = img.height
        img_w = img.width
        print("height is "+str(img_h)+", width is "+str(img_w))
        
        #plt.subplot(111),plt.imshow(img),plt.title('Raw ScreenShot')
        #plt.show()
        
        # find google chrome icon
        img_rgb = np.array(img)
        cv_img = img_rgb.astype(np.uint8)
        cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        #plt.subplot(111),plt.imshow(cv_gray),plt.title('ScreenShot now in opencv format')
        #plt.show()
        #cv2.imshow(None, cv_gray)
        #cv2.waitKey()
        #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        chrome_icon_template = cv2.imread(self.OGS_IMAGES_DIR + 'GoogleChromeDesktopIcon.jpg',0)
        #plt.subplot(111),plt.imshow(chrome_icon_template),plt.title('chrome icon')
        #plt.show()
        #cv2.imshow(None, chrome_icon_template)
        #cv2.waitKey()
         
        res = cv2.matchTemplate(cv_gray,chrome_icon_template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(res >= threshold)
        print("loc is "+str(loc))
        print("chrome_icon_template.shape = "+str(chrome_icon_template.shape))
        h,w  = chrome_icon_template.shape
        
        
        
        #print("w,h,d = "+str(w)+","+str(h)+","+str(d))
        print("loc = "+str(loc))
        loc = [loc[1][0], loc[0][0]]
        print("loc = "+str(loc))
        loc = [loc[0] + (w/2), loc[1] + (h/2)]
        print("about to set cursor")
        
        cv2.rectangle(img_rgb, tuple(loc), (loc[0] + w, loc[1] + h), (0,0,255), 2)
        plt.subplot(111),plt.imshow(img_rgb),plt.title('ScreenShot')
        plt.show()
        
        time.sleep(self.delay)
        win32api.SetCursorPos((loc[0],img_h - loc[1]))
        print("about to click")
        time.sleep(self.delay)
        click(loc[0], loc[1])
        
        
        
        
    def go(self):
        self.open_chrome()


def main():
    OGSBot().go()
    
if __name__ == '__main__':
    main()
    