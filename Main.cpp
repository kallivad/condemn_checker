#include <iostream>
#include <cv.h>
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
	const float kRescaleFactor = 0.5;


	while (key != 'q')
	{
		inframe = cvQueryFrame(capture);
		if (!frame) break;
		const int new_width = (int)((float)inframe->width * kRescaleFactor);
		const int new_height = (int)((float)inframe->height * kRescaleFactor);
		frame = cvCreateImage(cvSize(new_width, new_height), inframe->depth, inframe->nChannels);
		cvResize(inframe, frame);

		cv::Mat img_input(frame);
		//cv::imshow("Input", img_input);

		// bgs->process(...) internally process and show the foreground mask image
		cv::Mat img_mask;
		bgs->process(img_input, img_mask);

		if (!img_mask.empty())
		{
			// Perform blob tracking
			blobTracking->process(img_input, img_mask, img_blob);

			// Perform vehicle counting
			vehicleCouting->setInput(img_blob);
			vehicleCouting->setTracks(blobTracking->getTracks());
			vehicleCouting->process();
		}

		//Constructing foregound Subject from BGS mask

		key = cvWaitKey(1);
	}

	delete vehicleCouting;
	delete blobTracking;
	delete bgs;

	cvReleaseImage(&frame);
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);

	return 0;
}
