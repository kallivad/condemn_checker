/*
��������� ��������� ����������� ������� �� ����� ��������
�� ����� ��� � ������� ������� �����
������������� �� ��������� � ������������ ���� (ROI)
� ��������� �� � ���� ����� ����������� � ����������� �����
����� ������� ����� ������ ������������� ������� ( �� ������� )
*/

#include <iostream>
#include <cv.h>
#include <core.hpp>
#include <highgui.h>

#include "package_bgs/PBAS/PixelBasedAdaptiveSegmenter.h"
#include "package_tracking/BlobTracking.h"
#include "package_analysis/VehicleCouting.h"

using namespace cv;

int main(int argc, char **argv)
{
	//���������� �� ������������� ���������� OpenCV
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	/* �������� ����� ���������*/
	CvCapture *capture = 0;
	//capture = cvCaptureFromAVI("dataset/video.avi");
	capture = cvCaptureFromAVI(argv[1]);
	//capture = cvCaptureFromCAM(0);
	int width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
	int height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
	//int fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	
	// �������� ����� ������ ����������� - ������ ��� �������
	//cv::VideoWriter output;
	//output.open(argv[2], CV_FOURCC('H', '2', '6', '4'), fps, cv::Size(640, 480), true);
	//output.open(argv[2], CV_FOURCC('F', 'M', 'P', '4'), fps, cv::Size(width, height), false);

	if (!capture) {
		std::cerr << "Cannot open video!" << std::endl;
		return 1;
	}

	/* �������� ��������� ��������� ���� */
	//�� ���������� PBAS: https://sites.google.com/site/pbassegmenter/home
	IBGS *bgs;
	bgs = new PixelBasedAdaptiveSegmenter;

	/* �������� �������� ������� ��������� Blob */
	cv::Mat img_blob;
	BlobTracking* blobTracking;
	blobTracking = new BlobTracking;

	/* �������� �������� ��������� */
	VehicleCouting* issueCounting;
	issueCounting = new VehicleCouting;

	std::cout << "Press 'q' to quit..." << std::endl;
	int key = 0;
	IplImage *inframe, *frame;
	//const float kRescaleFactor = 640.0/(float)width;
	const float kRescaleFactor = 0.5;

	while (key != 'q')
	{
		inframe = cvQueryFrame(capture);
		if (!inframe) break;
		const int new_width = (int)((float)inframe->width * kRescaleFactor);
		const int new_height = (int)((float)inframe->height * kRescaleFactor);
		frame = cvCreateImage(cvSize(new_width, new_height), inframe->depth, inframe->nChannels);
		cvResize(inframe, frame);

		cv::Mat img_input(frame);		
		//������ ������� ����������� ��� ���������� �������
		cv::Mat img_inclear;
		img_input.copyTo(img_inclear);

		//cv::imshow("Input", img_input);

		// bgs->process(...) ������������ ����������� ������� ����� ����. ��������
		cv::Mat img_mask;
		bgs->process(img_input, img_mask);

		//��������������� ��������� ����� ��� ���������� ��������
		//����� �������� ������������
		int rect_size = 20;
		cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(rect_size, rect_size));
		morphologyEx(img_mask, img_mask, MORPH_CLOSE, kernel);
		cv::imshow("Processed Foreground", img_mask);

		//output.write(img_mask);

		if (!img_mask.empty())
		{
			// ������������ ������� ������� ��������� Blob ��������� ����
			blobTracking->process(img_input, img_mask, img_blob);

			// ���������� �������� ���� ������ � ������� ROI
			cvb::CvTracks tracks = blobTracking->getTracks();
			cvb::CvBlobs blobs = blobTracking->getBlobs();
			if (!tracks.empty())
			{
				std::map<cvb::CvID, cvb::CvTrack*>::iterator it = tracks.begin();
				//cvb::CvID id = (*it).first;
				cvb::CvTrack* track = (*it).second;

				CvPoint2D64f centroid = track->centroid; //���������� ������ ������� ���������
				double rx0 = issueCounting->r_x0;
				double ry0 = issueCounting->r_y0;
				double rx1 = issueCounting->r_x1;
				double ry1 = issueCounting->r_y1;

				if (centroid.x > rx0 && centroid.x < rx1 && centroid.y > ry0 && centroid.y < ry1)
				{
					std::cout << "Object in zone \n";
					
					if (!blobs.empty())
					{
						std::map<cvb::CvID, cvb::CvBlob*>::iterator iter = blobs.begin();
						//cvb::CvID id = (*iter).first;				
						cvb::CvBlob* blob = (*iter).second;

						int roi_x0 = blob->minx;
						int roi_y0 = blob->miny;
						int roi_x1 = blob->maxx;
						int roi_y1 = blob->maxy;
						cv::Rect roi_rect = cv::Rect(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);

						IplImage* input_image = cvCloneImage(&(IplImage)img_inclear);

						// ��������� ���
						cvSetImageROI(input_image, roi_rect);
						// �������� ������ ( image ) ��� ���������� ����������� �����������
						// ������� cvGetSize ���������� ������ ( width ) � ������ ( height ) ���
						IplImage *roi_image = cvCreateImage(cvGetSize(input_image), input_image->depth, input_image->nChannels);
						cvCopy(input_image, roi_image, NULL);
						cvResetImageROI(input_image);
						cv::Mat roi_img = cvarrToMat(roi_image);
						cv::imshow("ROI issue", roi_img);

						cvReleaseImage(&input_image);
						cvReleaseImage(&roi_image);
						//roi_img.release();

						//������������ ������ �� ������� �����������
						cv::rectangle(img_input, roi_rect, cv::Scalar(0, 250, 0), 2);
					}
				}

			}
			
			// ������������ ����� ��������
			issueCounting->setInput(img_blob);
			issueCounting->setTracks(blobTracking->getTracks());
			issueCounting->process();
		}
		

		img_inclear.release();
		img_blob.release();
		img_mask.release();
		img_input.release();

		key = cvWaitKey(1);
	}

	delete issueCounting;
	delete blobTracking;
	delete bgs;

	//output.release();	
	cvReleaseImage(&frame);
	cvDestroyAllWindows();
	cvReleaseCapture(&capture);

	return 0;
}
