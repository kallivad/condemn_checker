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

int main(int argc, char **argv)
{
	//Информация об использовании фреймворка OpenCV
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	int source_num = atoi(argv[2]);
	int control_num = atoi(argv[8]);

	/* Открытие видео источника*/
	CvCapture* capture[7];
	char video_file[50];
	//sprintf(video_file, "%s//out%d.avi", argv[3], 1);
	//sprintf(video_file, "%s", argv[3]);
	std::cout << "Opening " << video_file << "\n";
	for (int i = 0; i < source_num; i++)
	{
		capture[i] = 0;
	}

	std::cout << argv[1] << std::endl;

	if (atoi(argv[1]) == 0) 
	{
		std::cout << "Opening from " << argv[3] << std::endl;
		
		for (int i = 0; i < source_num; i++) {
			sprintf(video_file, "%s//out%d.avi", argv[3], i+1);
			capture[i] = cvCaptureFromAVI(video_file);
		}
	
	}
	else if (atoi(argv[1]) == 1) 
	{
		std::cout << "Opening from" << source_num << "cameras" << std::endl;
		for (int i = 0; i < source_num; i++) {
			capture[i] = cvCaptureFromCAM(i);
		}
	}


	//capture = cvCaptureFromCAM(0);
	int width = cvGetCaptureProperty(capture[control_num], CV_CAP_PROP_FRAME_WIDTH);
	int height = cvGetCaptureProperty(capture[control_num], CV_CAP_PROP_FRAME_HEIGHT);
	int fps = cvGetCaptureProperty(capture[control_num], CV_CAP_PROP_FPS);

	std::cout << "Capture created: " << width << "X" << height << "  FPS:" << fps << std::endl;
	
	int set_width = atoi(argv[6]);
	int set_height = atoi(argv[7]);

	std::cout << "Trying set propierty width: " << set_width << " height:" << set_height << std::endl;

	for (int i = 0; i < source_num; i++)
	{
		//обработать исключение
		cvSetCaptureProperty(capture[i], CV_CAP_PROP_FRAME_WIDTH, set_width);
		cvSetCaptureProperty(capture[i], CV_CAP_PROP_FRAME_HEIGHT, set_height);
	}

	int camera_num = 1;

	// Создание файла записи процессинга - только для отладки
	//cv::VideoWriter output;
	//output.open(argv[2], CV_FOURCC('F', 'M', 'P', '4'), fps, cv::Size(width, height), false);

	if (!capture[control_num]) {
		std::cerr << "Cannot open video!" << std::endl;
		return 1;
	}

	/* Алгоритм выделения переднего фона */
	//Мы используем PBAS: https://sites.google.com/site/pbassegmenter/home
	IBGS *bgs[7];
	

	/* Алгоритм трекинга связных компонент Blob */
	cv::Mat img_blob[7];
	BlobTracking* blobTracking[7];
	
	for (int i = 0; i < source_num; i++) {
		bgs[i] = new PixelBasedAdaptiveSegmenter;
		blobTracking[i] = new BlobTracking;
	}

	/* Алгоритм подсчёта предметов */
	VehicleCouting* issueCounting;
	issueCounting = new VehicleCouting;

	std::cout << "Press 'q' to quit..." << std::endl;
	int key = 0;
	IplImage *inframe[7], *frame[7];
	const float kRescaleFactor = atof(argv[5]);

	int issue_count = 0;
	int issue_in_roi_count = 0;
	bool first_time = true;


	while (key != 'q')
	{
		
		cv::Mat img_origin[7];
		cv::Mat img_input[7];
		cv::Mat img_mask[7];


		for (int i = 0; i < source_num; i++) 
		{
			inframe[i] = cvQueryFrame(capture[i]);
			img_origin[i] = cv::Mat(inframe[i]);
			if (!inframe[i]) break;
			const int new_width = (int)((float)inframe[i]->width * kRescaleFactor);
			const int new_height = (int)((float)inframe[i]->height * kRescaleFactor);
			frame[i] = cvCreateImage(cvSize(new_width, new_height), inframe[i]->depth, inframe[i]->nChannels);
			cvResize(inframe[i], frame[i]);

			img_input[i] = cv::Mat(frame[i]);

			// bgs->process(...) обрабатывает изображение выделяя маску движ. объектов		
			bgs[i]->process(img_input[i], img_mask[i]);

			//Визуализируем маску
			char win_name[10];
			sprintf(win_name, "FG%d", i + 1);
			if(!img_mask[i].empty())
				cv::imshow(win_name, img_mask[i]);

			//Морфологическая обработка маски для устранения разрывов
			//между близкими компонентами
			int rect_size = 20;
			cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(rect_size, rect_size));
			morphologyEx(img_mask[i], img_mask[i], MORPH_CLOSE, kernel);			
			kernel.release();
		}

		//output.write(img_mask[0]);

		if (!img_mask[control_num].empty())
		{
			cvb::CvBlobs blobs[7];
			cvb::CvTracks tracks;
			
			tracks = blobTracking[control_num]->getTracks();

		
			for (int i = 0; i < source_num; i++)
			{
				// Осуществляем трекинг связных компонент Blob переднего фона
				blobTracking[i]->process(img_input[i], img_mask[i], img_blob[i]);
			}
			
						
			if (!tracks.empty())
			{
				for (int i = 0; i < source_num; i++)
					blobs[i] = blobTracking[i]->getBlobs();

				std::map<cvb::CvID, cvb::CvTrack*>::iterator it = tracks.begin();
				//cvb::CvID id = (*it).first;
				cvb::CvTrack* track = (*it).second;

				CvPoint2D64f centroid = track->centroid; //координаты центра области связности
				double rx0 = issueCounting->r_x0;
				double ry0 = issueCounting->r_y0;
				double rx1 = issueCounting->r_x1;
				double ry1 = issueCounting->r_y1;

				// Записываем картинку если попали в область ROI контрольной [control_num] камеры
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
					IplImage* input_image[7];
					IplImage *roi_image[7];
					cv::Mat roi_img[7];
					
					//проходим по всем blobs от всех источников и выделяем на них текущий единственный blob
					for (int i = 0; i < source_num; i++)
					{

						if (!blobs[i].empty())
						{
							std::map<cvb::CvID, cvb::CvBlob*>::iterator iter = blobs[i].begin();
							//cvb::CvID id = (*iter).first;
							cvb::CvBlob* blob = (*iter).second;

							int roi_x0 = blob->minx / kRescaleFactor;
							int roi_y0 = blob->miny / kRescaleFactor;
							int roi_x1 = blob->maxx / kRescaleFactor;
							int roi_y1 = blob->maxy / kRescaleFactor;
							int width = roi_x1 - roi_x0;
							int height = roi_y1 - roi_y0;
							int side = width;
							if (side < height) side = height;
							cv::Rect roi_rect = cv::Rect(roi_x0, roi_y0, side, side);
							
							input_image[i] = cvCloneImage(&(IplImage)img_origin[i]);

							// Установка ИОР
							cvSetImageROI(input_image[i], roi_rect);
							// Создание образа ( roi_image ) для сохранения вырезанного изображения
							roi_image[i] = cvCreateImage(cvGetSize(input_image[i]), input_image[i]->depth, input_image[i]->nChannels);
							cvCopy(input_image[i], roi_image[i], NULL);
							cvResetImageROI(input_image[i]);							
							roi_img[i] = cvarrToMat(roi_image[i]);

							//формирование строки директории, файла и запись в файл
							char roi_file_dir[50];
							char roi_file_string[50];
							//sprintf(roi_file_string, "%s/%d/%d_issue.jpg", argv[2], issue_count, camera_num);
							sprintf(roi_file_dir, "%s\\%d\\", argv[4], issue_count);
							int res = mkdir(roi_file_dir);
							camera_num = i + 1;
							sprintf(roi_file_string, "%s\%d\\cam%d_num%d_issue.jpg", argv[4], issue_count, camera_num, issue_in_roi_count);
							std::cout << "Writing in " << roi_file_string << "\n";
							vector<int> compression_params;
							compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
							compression_params.push_back(100);
							imwrite(roi_file_string, roi_img[i], compression_params);

							//показываем в отдельном окне
							cv::imshow("ROI_issue", roi_img[i]);

							cvReleaseImage(&input_image[i]);
							cvReleaseImage(&roi_image[i]);							
						} //end of internal if (blobs)

					} //end of for
				
				} //end of middle if(roi)
				else
				{
					first_time = true;
					issue_in_roi_count = 0;
				} 

			} // end of external if(tracks)
			
			// Подсчитываем число объектов
			issueCounting->setInput(img_blob[control_num]);
			issueCounting->setTracks(tracks);
			issueCounting->process();
		} //end of external if (mask)
		
		for (int i = 0; i < source_num; i++)
		{
			img_origin[i].release();
			img_blob[i].release();
			img_mask[i].release();
			cvReleaseImage(&frame[i]);
			//img_input.release();
		}

		key = cvWaitKey(1);
	}

	delete issueCounting;
	
	for (int i = 0; i < source_num; i++) 
	{
		delete blobTracking[i];
		delete bgs[i];
		cvReleaseImage(&inframe[i]);
		cvReleaseCapture(&capture[i]);
	}

	//output.release();		
	cvDestroyAllWindows();

	return 0;
}
