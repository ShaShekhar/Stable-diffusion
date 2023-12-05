import numpy as np
import cv2
events = [i for i in dir(cv2) if 'EVENT' in i]
#print(events)
drawing = False
ix, iy = -1, -1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(imgwithalpha, (x,y), 40, (0,255,0,255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(imgwithalpha, (x,y), 40, (0,255,0,255), -1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
img = cv2.imread("Img-replace/IMG_20231129_174657.jpg")
R,G,B = cv2.split(img)
alpha = np.ones_like(R)*0
imgwithalpha = cv2.merge((R,G,B,alpha))
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', imgwithalpha)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        r,g,b,a = cv2.split(imgwithalpha)
        a_resize = cv2.resize(a, (512,512), interpolation=cv2.INTER_AREA)
        img_resize = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        cv2.imwrite('img_mask.png', a_resize)
        cv2.imwrite('img.png', img_resize)
        break
cv2.destroyAllWindows()
