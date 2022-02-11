import cv2
import  time
import imutils
cam=cv2.VideoCapture(0)
count=0
time.sleep(1)
firstFrame=None
area=500

while True:
    
    _,img=cam.read()
    text="Normal"
    imutils.resize(img,width=500)
    grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gaussianImg=cv2.GaussianBlur(img,(21,21),0)
    if firstFrame is None:
        firstFrame=gaussianImg
        continue
    imgDiff=cv2.absdiff(firstFrame,gaussianImg)
    thresImg=cv2.threshold(imgDiff,25,255,cv2.THRESH_BINARY)[1]
    thresImg=cv2.dilate(thresImg,None,iterations=2)
    cnts=cv2.findContours(thresImg.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue
        (x,y,w,h)=cv2.boundingRect(c)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text="Moving Object Detected"
        count+=1
    cv2.putText(img,str(count), (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

    print(text,count)
    cv2.putText(img,text, (10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    cv2.imshow("camerafeed",img)
    
    key=cv2.waitKey(1)
    if key== ord("q"):
        break
    
cam.release()
cv2.destroyAllWindows()
