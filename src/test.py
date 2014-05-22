import cv2
import brivasmodule
import numpy
image = '//Jatin/Brivas/users/gujju/block_1/gujju.jpg'
frame = cv2.imread(image)

#cv2.imshow('jazz',frame)
left = numpy.array([])
right = numpy.array([])
x = brivasmodule.detect(frame,image)

print x.shape
print x
cv2.imshow('left',x)
cv2.waitKey(0)
#print 'return value: ', x
    
    
    # g++ -fPIC -g -c -Wall -I/usr/local/include/opencv  -I/usr/local/include -L/usr/local/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ts -lopencv_video liblbp.cpp
#   g++ --shared -L/usr/local/lib -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ts -lopencv_video main.o findEyeCenter.o findEyeCorner.o flandmark_detector.o helpers.o liblbp.o -o mylib.so
#g++ -fPIC -g -c -Wall -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core  -I/usr/local/include/opencv  -I/usr/local/include -I/usr/include/python2.7 -L/usr/local/lib -lboost_python  -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ts -lopencv_video main.cpp

#g++ -fPIC -g -c -Wall  -I/usr/local/include/opencv  -I/usr/local/include -I/usr/include/python2.7 -L/usr/local/lib -lboost_python  -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ts -lopencv_video main.cpp
#g++ --shared -W1,--export-dynamic -L/usr/local/lib -lboost_python -lpython2.7 -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_objdetect -lopencv_ts -lopencv_video main.o findEyeCenter.o findEyeCorner.o flandmark_detector.o helpers.o liblbp.o -o brivasmodule.so
