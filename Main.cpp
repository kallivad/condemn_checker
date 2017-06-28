/*
Программа позволяет отслеживать объекты на ленте конвеера
по видео или с помощью системы камер
детектировать их появление в определенной зоне (ROI)
и сохранять их в виде серии изображений в определённой папке
номер которой равен номеру обнаруженного объекта ( по порядку )
*/

#include <iostream>
#include <cv.h>
#include <core.hpp>
#include <highgui.h>

#include <stdio.h>
#include <direct.h>

#include "package_bgs/PBAS/PixelBasedAdaptiveSegmenter.h"
#include "package_tracking/BlobTracking.h"
#include "package_analysis/VehicleCouting.h"

using namespace cv;

/*
cv::Rect recalculate_roi(cv::Rect roi, float scale) 
{
	
}*/

int main(int argc, char **argv)
{
	//Информация об использовании фреймворка OpenCV
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	/* Открытие видео источника*/
	CvCapture* capture[7];
	char video_file[50];
	//sprintf(video_file, "%s%d.avi", argv[1], 1);
	//sprintf(video_file, "%s", argv[1]);
	std::cout << "Opening " << video_file << "\n";
	for (int i = 0; i < 7; i++)
	{
		capture[i] = 0;
	}

	capture[0] = cvCaptureFromAVI("../video_new/out1.avi");
	capture[1] = cvCaptureFromAVI("../video_new/out2.avi");
	capture[2] = cvCaptureFromAVI("../video_new/out3.avi");


	//capture = cvCaptureFromCAM(0);
	int width = cvGetCaptureProperty(capture[1], CV_CAP_PROP_FRAME_WIDTH);
	int height = cvGetCaptureProperty(capture[1], CV_CAP_PROP_FRAME_HEIGHT);
	//int fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	
	int camera_num = 1;

	// Создание файла записи процессинга - только для отладки
	//cv::VideoWriter output;
	//output.open(argv[2], CV_FOURCC('F', 'M', 'P', '4'), fps, cv::Size(width, height), false);

	if (!capture) {
		std::cerr << "Cannot open video!" << std::endl;
		return 1;
	}

	/* Алгоритм выделения переднего фона */
	//Мы используем PBAS: https://sites.google.com/site/pbassegmenter/home
	IBGS *bgs[7];
	bgs[0] = new PixelBasedAdaptiveSegmenter;

	/* Алгоритм трекинга связных компонент Blob */
	cv::Mat img_blob[7];
	BlobTracking* blobTracking[7];
	blobTracking[0] = new BlobTracking;

	/* Алгоритм подсчёта предметов */
	VehicleCouting* issueCounting;
	issueCounting = new VehicleCouting;

	std::cout << "Press 'q' to quit..." << std::endl;
	int key = 0;
	IplImage *inframe[7], *frame[7];
	const float kRescaleFactor = 0.3;

	int issue_count = 0;
	int issue_in_roi_count = 0;
	bool first_time = true;

	while (key != 'q')
	{
		
		inframe[0] = cvQueryFrame(capture[0]);
		cv::Mat img_origin[7];
		img_origin[0] = cv::Mat(inframe[0]);
		if (!inframe[0]) break;
		const int new_width = (int)((float)inframe[0]->width * kRescaleFactor);
		const int new_height = (int)((float)inframe[0]->height * kRescaleFactor);
		frame[0] = cvCreateImage(cvSize(new_width, new_height), inframe[0]->depth, inframe[0]->nChannels);
		cvResize(inframe[0], frame[0]);

		cv::Mat img_input[7];
		img_input[0]=cv::Mat(frame[0]);

		// bgs->process(...) обрабатывает изображение выделяя маску движ. объектов
		cv::Mat img_mask[7];
		bgs[0]->process(img_input[0], img_mask[0]);

		//Морфологическая обработка маски для устранения разрывов
		//между близкими компонентами
		int rect_size = 20;
		cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(rect_size, rect_size));
		morphologyEx(img_mask[0], img_mask[0], MORPH_CLOSE, kernel);
		cv::imshow("Processed Foreground", img_mask[0]);
		kernel.release();

		//output.write(img_mask);

		if (!img_mask[0].empty())
		{
			// Осуществляем трекинг связных компонент Blob переднего фона
			blobTracking[0]->process(img_input[0], img_mask[0], img_blob[0]);

			// Записываем картинку если попали в область ROI первой ind[0] камеры
			cvb::CvTracks tracks = blobTracking[0]->getTracks();
			cvb::CvBlobs blobs[7];
			blobs[0] = blobTracking[0]->getBlobs();

			if (!tracks.empty())
			{
				std::map<cvb::CvID, cvb::CvTrack*>::iterator it = tracks.begin();
				//cvb::CvID id = (*it).first;
				cvb::CvTrack* track = (*it).second;

				CvPoint2D64f centroid = track->centroid; //координаты центра области связности
				double rx0 = issueCounting->r_x0;
				double ry0 = issueCounting->r_y0;
				double rx1 = issueCounting->r_x1;
				double ry1 = issueCounting->r_y1;

				if (centroid.x > rx0 && centroid.x < rx1 && centroid.y > ry0 && centroid.y < ry1)
				{
					issue_in_roi_count++;
					if (first_time) 
					{
						cv::destroyWindow("ROI_issue");
						issue_count++; 
						first_time = false;
					}

					std::cout << "Object in zone \n";
					
					if (!blobs[0].empty())
					{
						std::map<cvb::CvID, cvb::CvBlob*>::iterator iter = blobs[0].begin();
						//cvb::CvID id = (*iter).first;
						cvb::CvBlob* blob = (*iter).second;

						int roi_x0 = blob->minx / kRescaleFactor;
						int roi_y0 = blob->miny / kRescaleFactor;
						int roi_x1 = blob->maxx / kRescaleFactor;
						int roi_y1 = blob->maxy / kRescaleFactor;
						cv::Rect roi_rect = cv::Rect(roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0);


						IplImage* input_image[7];
						input_image[0] = cvCloneImage(&(IplImage)img_origin[0]);

						// Установка ИОР
						cvSetImageROI(input_image[0], roi_rect);
						// Создание образа ( image ) для сохранения вырезаниого изображения
						// Функция cvGetSize возвращает ширину ( width ) и высоту ( height ) ИОР
						IplImage *roi_image[7];
						roi_image[0] = cvCreateImage(cvGetSize(input_image[0]), input_image[0]->depth, input_image[0]->nChannels);
						cvCopy(input_image[0], roi_image[0], NULL);
						cvResetImageROI(input_image[0]);
						cv::Mat roi_img[7];
						roi_img[0] = cvarrToMat(roi_image[0]);
						
						//формирование строки директории, файла и запись в файл
						char roi_file_dir[50];
						char roi_file_string[50];
						//sprintf(roi_file_string, "%s/%d/%d_issue.jpg", argv[2], issue_count, camera_num);
						sprintf(roi_file_dir, "%s\\%d\\",argv[2],issue_count);
						int res = mkdir(roi_file_dir);
						sprintf(roi_file_string, "%s\%d\\cam%d_num%d_issue.jpg",argv[2], issue_count, camera_num, issue_in_roi_count);
						std::cout << "Writing in " << roi_file_string << "\n";
						vector<int> compression_params;
						compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
						compression_params.push_back(100);
						imwrite(roi_file_string, roi_img[0], compression_params);

						//показываем в отдельном окне
						cv::imshow("ROI_issue", roi_img[0]);

						cvReleaseImage(&input_image[0]);
						cvReleaseImage(&roi_image[0]);
						//roi_img.release();

						//подсвечиваем объект на входном изображении
						//cv::rectangle(img_input, roi_rect, cv::Scalar(0, 250, 0), 2);
					}
				}
				else 
				{ 
					first_time = true; 
					issue_in_roi_count = 0; 
				}

			}
			
			// Подсчитываем число объектов
			issueCounting->setInput(img_blob[0]);
			issueCounting->setTracks(tracks);
			issueCounting->process();
		}
		

		img_origin[0].release();
		img_blob[0].release();
		img_mask[0].release();
		//img_input.release();
		cvReleaseImage(&frame[0]);

		key = cvWaitKey(1);
	}

	delete issueCounting;
	delete blobTracking[0];
	delete bgs;

	//output.release();	
	cvReleaseImage(&inframe[0]);
	cvDestroyAllWindows();
	cvReleaseCapture(capture);

	return 0;
}
