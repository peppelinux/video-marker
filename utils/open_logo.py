import cv2
file_name = "python.png"

src = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

# Save the transparency channel alpha
*_, alpha = cv2.split(src)

gray_layer = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# ... Your image processing

# Duplicate the grayscale image to mimic the BGR image and finally add the transparency
dst = cv2.merge((gray_layer, gray_layer, gray_layer, alpha))
#cv2.imwrite("result.png", dst)

while 1:
    cv2.imshow('watermark',dst)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
