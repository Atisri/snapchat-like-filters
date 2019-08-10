import numpy as np
import cv2
from skimage import exposure

from utils import CFEVideoConf, image_resize

cap = cv2.VideoCapture(0)

save_path = 'saved-media/recording.mp4'
frames_per_seconds = 24
config= CFEVideoConf(cap, filepath=save_path, res='720p')
out= cv2.VideoWriter(save_path, config.video_type, frames_per_seconds, config.dims)
face_cascade= cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
eyes_cascade= cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')
nose_cascade= cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')
glasses= cv2.imread("images/fun/yellow_shades.png", -1)
mustache= cv2.imread('images/fun/mustache.png',-1)
bg= cv2.imread('images/fun/bg1.jpg',-1)
bg2 = cv2.imread('images/fun/bg2.jpg',-1)
#cap1 = cv2.imread('images/fun/cap.png',-1)
teeth = cv2.imread('images/fun/teeth3.png',-1)


print('press 1 to unlock avtar 1' )
print('press anything accept 1 to uncock avtar 2')

    
c=int(input())

while(True):
    _, frame = cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
   
    
    if c==1:

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+h] 
            roi_color = frame[y:y+h, x:x+h]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
    
            eyes = eyes_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=5)
            for (ex, ey, ew, eh) in eyes:
                #cv2.rectangle(roi_color, (ex, ey), ((ex + ew), ey + eh), (0, 255, 0), 3)
                roi_eyes = roi_gray[ey: ey + eh, ex: (ex + ew)]
                glasses2 = image_resize(glasses.copy(), width=2*ew)
    
                gw, gh, gc = glasses2.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        #print(glasses[i, j]) 
                        if glasses2[i, j][3] != 0: 
                            roi_color[ey + i, ex + j] = glasses2[i, j]
                            
        
    
            nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (nx, ny, nw, nh) in nose:
                #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
                roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
                mustache2 = image_resize(mustache.copy(), width=nw)
    #
                mw, mh, mc = mustache2.shape
                for i in range(0, mw):
                    for j in range(0, mh):
                        #print(glasses[i, j]) 
                        if mustache2[i, j][3] != 0: 
                            roi_color[ny + int(nh/2.0) + i, nx + j] = mustache2[i, j]
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        frame= exposure.adjust_log(frame,1)
        
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        blurImage=cv2.medianBlur(grayImage,5)
    
        edges = cv2.adaptiveThreshold(blurImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)
    
        #colorImage = cv2.bilateralFilter(frame, 9, 300, 300)
        colorImage=frame
        m=np.ones((3,3),np.uint8)
        edges=cv2.erode(edges,m,iterations=1)
        edges=cv2.dilate(edges,m,iterations=1)
        rows=np.size(frame,0)
        cols=np.size(frame,1)
        waterColor = cv2.bitwise_and(colorImage, colorImage, mask=edges)
    
        back3=cv2.resize(bg,(cols,rows))
            
        waterColor=cv2.addWeighted(waterColor,0.7,back3, 0.3,0)
    else:
        
        for (x, y, w, h) in faces:
            roi_gray    = gray[y:y+h, x:x+h] # rec
            roi_color   = frame[y:y+h, x:x+h]
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
    
    
            nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (nx, ny, nw, nh) in nose:
                #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
                roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
                teeth2 = image_resize(teeth.copy(), width=nw)
    #
                mw, mh, mc = teeth2.shape
                for i in range(0, mw):
                    for j in range(0, mh):
                        #print(glasses[i, j]) 
                        if teeth2[i, j][3] != 0: 
                            roi_color[ny +int(nh/2) + i, nx + j] = teeth2[i, j]
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        frame= exposure.adjust_log(frame,1)
        
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        blurImage=cv2.medianBlur(grayImage,5)
    
        edges = cv2.adaptiveThreshold(blurImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 3)
        colorImage=frame
        #colorImage = cv2.bilateralFilter(frame, 9, 300, 300)
    
        m=np.ones((3,3),np.uint8)
        edges=cv2.erode(edges,m,iterations=1)
        edges=cv2.dilate(edges,m,iterations=1)
        rows=np.size(frame,0)
        cols=np.size(frame,1)
        waterColor = cv2.bitwise_and(colorImage, colorImage, mask=edges)
    
        back3=cv2.resize(bg2,(cols,rows))
            
        waterColor=cv2.addWeighted(waterColor,0.7,back3, 0.3,0)

   
    
    out.write(waterColor)
    cv2.imshow('frame',waterColor)
    
    if cv2.waitKey(20) & 0xFF == ord('c'):
        cv2.imwrite('a.jpg',waterColor)
        break


cap.release()
out.release()
cv2.destroyAllWindows()