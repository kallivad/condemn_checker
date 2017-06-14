#include <iostream>
#include <cv.h>
#include <core.hpp>
#include <highgui.h>

#include "package_bgs/PBAS/PixelBasedAdaptiveSegmenter.h"
#include "package_tracking/BlobTracking.h"
#include "package_analysis/VehicleCouting.h"


int main(int argc, char **argv)
{
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	/* Open video file */
	CvCapture *capture = 0;
	//capture = cvCaptureFromAVI("dataset/video.avi");
	capture = cvCaptureFromAVI(argv[1]);
	int width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	int height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
	int fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	cv::VideoWriter output;
	output.open(argv[2], CV_FOURCC('H', '2', '6', '4'), fps, cv::Size(640, 480), true);
	//output.open(argv[2], -1, fps, cv::Size(width, height), false);

	if (!capture) {
		std::cerr << "Cannot open video!" << std::endl;
		return 1;
	}

	/* Background Subtraction Algorithm */
	IBGS *bgs;
	bgs = new PixelBasedAdaptiveSegmenter;

	/* Blob Tracking Algorithm */
	cv::Mat img_blob;
	BlobTracking* blobTracking;
	blobTracking = new BlobTracking;

	/* Vehicle Counting Algorithm */
	VehicleCouting* vehicleCouting;
	vehicleCouting = new VehicleCouting;

	std::cout << "Press 'q' to quit..." << std::endl;
	int key = 0;
	IplImage *inframe, *frame;
	const float kRescaleFactor = 640.0/(float)width;


	while (key != 'q')
	{
		inframe = cvQueryFrame(capture);
		if (!inframe) break;
		const int new_width = (int)((float)inframe->width * kRescaleFactor);
		const int new_height = (int)((float)inframe->height * kRescaleFactor);
		frame = cvCreateImage(cvSize(new_width, new_height), inframe->depth, inframe->nChannels);
		cvResize(inframe, frame);

		cv::Mat img_input(frame);
		//cv::imshow("Input", img_input);

		// bgs->process(...) internally process and show the foreground mask image
		cv::Mat img_mask;
		bgs->process(img_input, img_mask);

		output.write(img_mask);

		if (!img_mask.empty())
		{
			// Perform blob tracking
			blobTracking->process(img_input, img_mask, img_blob);

			// Perform vehicle counting
			vehicleCouting->setInput(img_blob);
			vehicleCouting->setTracks(blobTracking->getTracks());
			vehicleCouting->process();
		}

		//Constructing foregound Subject from BGS maskﬂ Ë

		key = cvWaitKey(1);
	}

	delete vehicleCouting;
	delete blobTracking;
	delete bgs;

	output.release();
	cvReleaseImage(&frame);
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);

	return 0;
}
