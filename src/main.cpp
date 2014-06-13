/*
 gaze tracking
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <Python.h>
#include <vector>
#include "numpy/ndarrayobject.h"
using namespace std;
using namespace cv;
#define PI 3.14159265
int desktop_x;
#include <cstring>
#include "flandmark_detector.h"
int dx;
int dy;

#include<boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;




IplImage *imgScribble = NULL;
cv::String face_cascade_name =
		"haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
std::string main_window_name = "Capture - Face detection";
std::string face_window_name = "Capture - Face";
cv::Mat debugImage;
cv::Mat skinCrCbHist = cv::Mat::zeros(cv::Size(256, 256), CV_8UC1);

cv::RNG rng(12345);
void findEyes(cv::Mat frame_gray, cv::Rect face, double *landmarks, int dx,
		int dy, std::string line,cv::Mat& leftEye,cv::Mat& rightEye);

// The following conversion functions are taken from OpenCV's cv2.cpp file inside modules/python/src2 folder.
static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
        {
            _sizes[i] = sizes[i];
        }

        if( cn > 1 )
        {
            /*if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else*/
                _sizes[dims++] = cn;
        }

        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);

        if(!o)
        {
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        }
        refcount = refcountFromPyObject(o);

        npy_intp* _strides = PyArray_STRIDES(o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA(o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};

NumpyAllocator g_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

static int pyopencv_to(const PyObject* o, Mat& m, const char* name = "<unknown>", bool allowND=true)
{
    //NumpyAllocator g_numpyAllocator;
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array", name);
        return false;
    }

    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("%s data type = %d is not supported", name, typenum);
        return false;
    }

    int ndims = PyArray_NDIM(o);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(o), step);

    if( m.data )
    {
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_numpyAllocator;

    if( transposed )
    {
        Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return true;
}

static PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
}




// -------------------------------------------------------------------------------------
// detect face using landmark detection and crops the face in a image. call findEyes function
// to detect eye corners and pupil in croppedimage
void detectFaceInImage(IplImage *orig, IplImage* input,
		CvHaarClassifierCascade* cascade, FLANDMARK_Model *model, int *bbox,
		double *landmarks, std::string line,cv::Mat & leftEye,cv::Mat & rightEye) {
	// Smallest face size.
	CvSize minFeatureSize = cvSize(40, 40);
	int flags = CV_HAAR_DO_CANNY_PRUNING;
	// How detailed should the search be.
	float search_scale_factor = 1.1f;
	CvMemStorage* storage;
	CvSeq* rects;
	int nFaces;

	storage = cvCreateMemStorage(0);
	cvClearMemStorage(storage);

	// Detect all the faces in the greyscale image.
	rects = cvHaarDetectObjects(input, cascade, storage, search_scale_factor, 2,
			flags, minFeatureSize);
	nFaces = rects->total;

	double t = (double) cvGetTickCount();
	for (int iface = 0; iface < (rects ? nFaces : 0); ++iface) {
		CvRect *r = (CvRect*) cvGetSeqElem(rects, iface);

		bbox[0] = r->x;
		bbox[1] = r->y;
		bbox[2] = r->x + r->width;
		bbox[3] = r->y + r->height;

		flandmark_detect(input, bbox, model, landmarks);

		//pupil 
		dx = bbox[0];
		dy = bbox[1];
		// crop the image based on region_of_interest
		cv::Rect region_of_interest = cv::Rect(bbox[0], bbox[1], r->width,
				r->height);
		cv::Mat image(input);
		cv::Mat image_rois = image(region_of_interest);
		createCornerKernels();
		ellipse(skinCrCbHist, cv::Point(113, int(155.6)), cv::Size(int(23.4), int(15.2)), 43.0,
				0.0, 360.0, cv::Scalar(255, 255, 255), -1);
		std::vector < cv::Rect > faces;

		// function call for pupil and eye corner detection
		findEyes(image, region_of_interest, landmarks, dx, dy, line,leftEye,rightEye);
		/*imshow("ROI",leftEye);
		cvWaitKey(0);
		imshow("ROI",rightEye);
		cvWaitKey(0);*/
		releaseCornerKernels();
		// display landmarks
		cvRectangle(orig, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]),
				CV_RGB(255, 0, 0));
		cvRectangle(orig, cvPoint(model->bb[0], model->bb[1]),
				cvPoint(model->bb[2], model->bb[3]), CV_RGB(0, 0, 255));
		cvCircle(orig, cvPoint((int) landmarks[0], (int) landmarks[1]), 3,
				CV_RGB(0, 0, 255), CV_FILLED);
		for (int i = 2; i < 2 * model->data.options.M; i += 2) {
			cvCircle(orig, cvPoint(int(landmarks[i]), int(landmarks[i + 1])), 3,
					CV_RGB(255, 0, 0), CV_FILLED);
		}

	}

	t = (double) cvGetTickCount() - t;
	int ms = cvRound(t / ((double) cvGetTickFrequency() * 1000.0));

	if (nFaces > 0) {
		/*printf(
				"Faces detected: %d; Detection of facial landmark on all faces took %d ms\n",
				nFaces, ms);*/
	} else {
		printf("No Face\n");
	}
	cvReleaseMemStorage(&storage);

}

PyObject* detect(PyObject* myframe,char* n) {

	char flandmark_window[] = "flandmark_example2";
	// load classifiers - Haar Cascade file, used for Face Detection.
	char faceCascadeFilename[] =
			"haarcascade_frontalface_alt.xml";
	// Load the HaarCascade classifier for face detection.
	CvHaarClassifierCascade* faceCascade;

	faceCascade = (CvHaarClassifierCascade*) cvLoad(faceCascadeFilename, 0, 0,
			0);
	if (!faceCascade) {
		printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
		exit(1);
	}

	double t;
	int ms;

	//string path = "/Jatin/Brivas/users/gujju/block_1/gujju.jpg";
    string path(n);
    
	Mat image;
	pyopencv_to(myframe,image); // eye patches obtained from findeyecenter is used are saved in location defined in path
	//image = imread(path);
	t = (double) cvGetTickCount();
	FLANDMARK_Model * model = flandmark_init("flandmark_model.dat");
	if (model == 0) {
		printf(
				"Structure model was not created. Corrupted file flandmark_model.dat?\n");
		exit(1);
	}
	t = (double) cvGetTickCount() - t;
	ms = cvRound(t / ((double) cvGetTickFrequency() * 1000.0));
	//printf("Structure model loaded in %d ms.\n", ms);
	// ------------- end flandmark load model
	int *bbox = (int*) malloc(4 * sizeof(int));
	double *landmarks = (double*) malloc(
				2 * model->data.options.M * sizeof(double));

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);
	Mat frame_gray;
	cvtColor(image,frame_gray,CV_BGR2GRAY);
	IplImage copy1 = frame_gray;
	//IplImage* newCopy = &copy;
	IplImage* frame =  &copy1;
	cv ::Mat leftEye,rightEye;
	detectFaceInImage(frame, frame, faceCascade, model, bbox, landmarks,
				path,leftEye,rightEye);
		//calculate FPS
	/*imshow("ROI",leftEye);
	cvWaitKey(0);
	imshow("ROI",rightEye);
	cvWaitKey(0);*/
	cv::Mat eyes;
	if(!leftEye.empty() && !rightEye.empty()){
	cv::resize(leftEye, leftEye, Size(112,48));
	cv::resize(rightEye, rightEye, Size(112,48));
	vconcat(leftEye,rightEye, eyes);
	}
	/*cvShowImage(flandmark_window, frame);
	cvWaitKey(0);
	*/	// Free the camera.
	free(landmarks);
	free(bbox);
	cvReleaseHaarClassifierCascade(&faceCascade);
	cvDestroyWindow(flandmark_window);
	flandmark_free(model);
	//return eyes;
	return pyopencv_from(eyes);
}

BOOST_PYTHON_MODULE(brivasmodule)
{
	import_array();
	def("detect",detect);
}

int main(int, char **) {

  Py_Initialize();

  try {
    initbrivasmodule(); // initialize Pointless

    PyRun_SimpleString("import brivasmodule");
  } catch (error_already_set) {
    PyErr_Print();
  }

  Py_Finalize();
  return 0;
}
/* --------------------------- detects pupil --------------------------------------------------
 inputs are:
 1- cropped image,2  Vector of rectangles where each rectangle contains the detected object,
 3- landmark locations, 4 width of cropped region, 5 height of cropped region
 ----------------------------------------------------------------------------------------------*/
void findEyes(cv::Mat frame_gray, cv::Rect face, double *landmarks, int dx,
		int dy, std::string path,cv::Mat &leftEye,cv::Mat& rightEye) {
//	cout << "eyes" << line << endl;
	cv::Mat faceROI = frame_gray(face);
	cv::Mat debugFace = faceROI;
	// std::string path = line;
	if (kSmoothFaceImage) {
		double sigma = kSmoothFaceFactor * face.width;
		GaussianBlur(faceROI, faceROI, cv::Size(0, 0), sigma);
	}
	//-- Find eye regions and draw them
	int eye_region_width = face.width * (kEyePercentWidth / 100.0);
	int eye_region_height = face.width * (kEyePercentHeight / 100.0);
	int eye_region_top = face.height * (kEyePercentTop / 100.0);
	cv::Rect leftEyeRegion(face.width * (kEyePercentSide / 100.0),
			eye_region_top, eye_region_width, eye_region_height);
	cv::Rect rightEyeRegion(
			face.width - eye_region_width
					- face.width * (kEyePercentSide / 100.0), eye_region_top,
			eye_region_width, eye_region_height);

	//-- Find Eye Centers
	cv::Point leftPupil = findEyeCenter(faceROI, leftEyeRegion, "Left Eye",
			path, "left");
	cv::Point rightPupil = findEyeCenter(faceROI, rightEyeRegion, "Right Eye",
			path, "right");
	// clone image
	Mat img = debugFace.clone();
	// get corner regions
	cv::Rect leftRightCornerRegion(leftEyeRegion);

	leftRightCornerRegion.width -= leftPupil.x;
	leftRightCornerRegion.x += leftPupil.x;
	leftRightCornerRegion.height /= 2;
	leftRightCornerRegion.y += leftRightCornerRegion.height / 2;

	cv::Rect leftLeftCornerRegion(leftEyeRegion);
	leftLeftCornerRegion.width = leftPupil.x;
	leftLeftCornerRegion.height /= 2;
	leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;


	cv::Rect rightLeftCornerRegion(rightEyeRegion);
	rightLeftCornerRegion.width = rightPupil.x;
	rightLeftCornerRegion.height /= 2;
	rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
	cv::Rect rightRightCornerRegion(rightEyeRegion);
	rightRightCornerRegion.width -= rightPupil.x;
	rightRightCornerRegion.x += rightPupil.x;
	rightRightCornerRegion.height /= 2;
	rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
/*

	cout << "left right" << leftRightCornerRegion.x << leftRightCornerRegion.y<<leftRightCornerRegion.height<< leftRightCornerRegion.width << endl;
	cout << "Left right " << leftRightCornerRegion << endl;
	cout << "left left" << leftLeftCornerRegion << endl;
	cout << "left left" << leftLeftCornerRegion.x << leftLeftCornerRegion.y << leftLeftCornerRegion.height << leftLeftCornerRegion.width << endl;
*/

	int leftWidth = leftLeftCornerRegion.width + leftRightCornerRegion.width;
	int leftHeight = leftLeftCornerRegion.height;
	int rightWidth = leftLeftCornerRegion.width + leftRightCornerRegion.width;
	int rightHeight = leftLeftCornerRegion.height;

	cv::Mat leftImageROI;
	cv::Mat rightImageROI;
	leftImageROI = debugFace(
			cv::Rect(leftLeftCornerRegion.x, leftLeftCornerRegion.y, leftWidth,
					leftHeight));
	rightImageROI = debugFace(
				cv::Rect(rightLeftCornerRegion.x, rightRightCornerRegion.y, rightWidth,
						rightHeight));
	//imwrite(path, leftImageROI); // eye patch extracted and saved in path defined.
	//imwrite(path, rightImageROI); // eye patch extracted and saved in path defined.
	/*
	imshow("ROI",leftImageROI);
	cvWaitKey(0);
	imshow("ROI",rightImageROI);
	cvWaitKey(0);*/
	leftEye = leftImageROI.clone();
	rightEye = rightImageROI.clone();
	rectangle(debugFace, leftRightCornerRegion, 200);
	rectangle(debugFace, leftLeftCornerRegion, 200);
	rectangle(debugFace, rightLeftCornerRegion, 200);
	rectangle(debugFace, rightRightCornerRegion, 200);
	// change eye centers to face coordinates
	rightPupil.x += rightEyeRegion.x;
	rightPupil.y += rightEyeRegion.y;
	leftPupil.x += leftEyeRegion.x;
	leftPupil.y += leftEyeRegion.y;
	// draw eye centers

	circle(debugFace, rightPupil, 3, 1234);
	circle(debugFace, leftPupil, 3, 1234);
	// relocate pupil corners to the uncropped image.
	//float RP_x = rightPupil.x + dx;
	//float RP_y = 0; // rightPupil.y+dy;
	//float LP_x = leftPupil.x + dx;
	//float LP_y = 0; //leftPupil.y+dy;
}
