#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/photo/cuda.hpp>
#include <opencv2/photo.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

void conv2(Mat src, int kernel_size)
{

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			Vec3b bgrPixel = src.at<Vec3b>(i, j);

			// do something with BGR values...
		}
	}

	////////////////////////////////////////////////////////////// MATCH FILTER PART //////////////////////////////////////////////////////////////////////
	Mat dst, kernel, kernel_pro, src_gray, greyMat, blur_image, adp_image, ad_thr_med, dst2, profiltered, dst3, dest_gabor, dst4;


	kernel_pro = (Mat_<float>(21, 21) <<
		0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   7.3633207e-06,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,
		0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   9.9022245e-06, - 9.9925971e-07, - 1.0537604e-05, - 1.3228926e-05, - 1.2906245e-05, - 1.3228926e-05, - 1.0537604e-05, - 9.9925971e-07,   9.9022245e-06,   0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,
		- 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   2.4728735e-05,   8.3992305e-06, - 1.9064050e-05, - 4.5946268e-05, - 5.3760363e-05, - 4.4442205e-05, - 3.7524185e-05, - 4.4442205e-05, - 5.3760363e-05, - 4.5946268e-05, - 1.9064050e-05,   8.3992305e-06,   2.4728735e-05,   0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,
		- 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   3.6472655e-05,   1.7199104e-05, - 2.8858374e-05, - 7.4575483e-05, - 8.1185581e-05, - 3.2812420e-05,   3.1048220e-05,   5.9422589e-05,   3.1048220e-05, - 3.2812420e-05, - 8.1185581e-05, - 7.4575483e-05, - 2.8858374e-05,   1.7199104e-05,   3.6472655e-05,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,
		- 0.0000000e+00,   0.0000000e+00,   2.4728735e-05,   1.7199104e-05, - 2.9197428e-05, - 8.5319764e-05, - 6.9304815e-05,   6.1141404e-05,   2.3491006e-04,   3.5269368e-04,   3.8883844e-04,   3.5269368e-04,   2.3491006e-04,   6.1141404e-05, - 6.9304815e-05, - 8.5319764e-05, - 2.9197428e-05,   1.7199104e-05,   2.4728735e-05,   0.0000000e+00, - 0.0000000e+00,
		0.0000000e+00,   0.0000000e+00,   8.3992305e-06, - 2.8858374e-05, - 8.5319764e-05, - 5.4036416e-05,   1.6293519e-04,   4.3670950e-04,   5.0257520e-04,   3.2855556e-04,   2.0579024e-04,   3.2855556e-04,   5.0257520e-04,   4.3670950e-04,   1.6293519e-04, - 5.4036416e-05, - 8.5319764e-05, - 2.8858374e-05,   8.3992305e-06,   0.0000000e+00,   0.0000000e+00,
		0.0000000e+00,   9.9022245e-06, - 1.9064050e-05, - 7.4575483e-05, - 6.9304815e-05,   1.6293519e-04,   5.1298516e-04,   3.9682933e-04, - 6.6764063e-04, - 2.1999850e-03, - 2.9433000e-03, - 2.1999850e-03, - 6.6764063e-04,   3.9682933e-04,   5.1298516e-04,   1.6293519e-04, - 6.9304815e-05, - 7.4575483e-05, - 1.9064050e-05,   9.9022245e-06,   0.0000000e+00,
		0.0000000e+00, - 9.9925971e-07, - 4.5946268e-05, - 8.1185581e-05,   6.1141404e-05,   4.3670950e-04,   3.9682933e-04, - 1.3679917e-03, - 5.3771477e-03, - 9.9552993e-03, - 1.2025360e-02, - 9.9552993e-03, - 5.3771477e-03, - 1.3679917e-03,   3.9682933e-04,   4.3670950e-04,   6.1141404e-05, - 8.1185581e-05, - 4.5946268e-05, - 9.9925971e-07,   0.0000000e+00,
		0.0000000e+00, - 1.0537604e-05, - 5.3760363e-05, - 3.2812420e-05,   2.3491006e-04,   5.0257520e-04, - 6.6764063e-04, - 5.3771477e-03, - 1.3948939e-02, - 2.2880908e-02, - 2.6844249e-02, - 2.2880908e-02, - 1.3948939e-02, - 5.3771477e-03, - 6.6764063e-04,   5.0257520e-04,   2.3491006e-04, - 3.2812420e-05, - 5.3760363e-05, - 1.0537604e-05,   0.0000000e+00,
		0.0000000e+00, - 1.3228926e-05, - 4.4442205e-05,   3.1048220e-05,   3.5269368e-04,   3.2855556e-04, - 2.1999850e-03, - 9.9552993e-03, - 2.2880908e-02, - 3.5826761e-02, - 4.1580881e-02, - 3.5826761e-02, - 2.2880908e-02, - 9.9552993e-03, - 2.1999850e-03,   3.2855556e-04,   3.5269368e-04,   3.1048220e-05, - 4.4442205e-05, - 1.3228926e-05,   0.0000000e+00,
		7.3633207e-06, - 1.2906245e-05, - 3.7524185e-05,   5.9422589e-05,   3.8883844e-04,   2.0579024e-04, - 2.9433000e-03, - 1.2025360e-02, - 2.6844249e-02, - 4.1580881e-02,   9.5216230e-01, - 4.1580881e-02, - 2.6844249e-02, - 1.2025360e-02, - 2.9433000e-03,   2.0579024e-04,   3.8883844e-04,   5.9422589e-05, - 3.7524185e-05, - 1.2906245e-05,   7.3633207e-06,
		0.0000000e+00, - 1.3228926e-05, - 4.4442205e-05,   3.1048220e-05,   3.5269368e-04,   3.2855556e-04, - 2.1999850e-03, - 9.9552993e-03, - 2.2880908e-02, - 3.5826761e-02, - 4.1580881e-02, - 3.5826761e-02, - 2.2880908e-02, - 9.9552993e-03, - 2.1999850e-03,   3.2855556e-04,   3.5269368e-04,   3.1048220e-05, - 4.4442205e-05, - 1.3228926e-05,   0.0000000e+00,
		0.0000000e+00, - 1.0537604e-05, - 5.3760363e-05, - 3.2812420e-05,   2.3491006e-04,   5.0257520e-04, - 6.6764063e-04, - 5.3771477e-03, - 1.3948939e-02, - 2.2880908e-02, - 2.6844249e-02, - 2.2880908e-02, - 1.3948939e-02, - 5.3771477e-03, - 6.6764063e-04,   5.0257520e-04,   2.3491006e-04, - 3.2812420e-05, - 5.3760363e-05, - 1.0537604e-05,   0.0000000e+00,
		0.0000000e+00, - 9.9925971e-07, - 4.5946268e-05, - 8.1185581e-05,   6.1141404e-05,   4.3670950e-04,   3.9682933e-04, - 1.3679917e-03, - 5.3771477e-03, - 9.9552993e-03, - 1.2025360e-02, - 9.9552993e-03, - 5.3771477e-03, - 1.3679917e-03,   3.9682933e-04,   4.3670950e-04,   6.1141404e-05, - 8.1185581e-05, - 4.5946268e-05, - 9.9925971e-07,   0.0000000e+00,
		0.0000000e+00,   9.9022245e-06, - 1.9064050e-05, - 7.4575483e-05, - 6.9304815e-05,   1.6293519e-04,   5.1298516e-04,   3.9682933e-04, - 6.6764063e-04, - 2.1999850e-03, - 2.9433000e-03, - 2.1999850e-03, - 6.6764063e-04,   3.9682933e-04,   5.1298516e-04,   1.6293519e-04, - 6.9304815e-05, - 7.4575483e-05, - 1.9064050e-05,   9.9022245e-06,   0.0000000e+00,
		0.0000000e+00,   0.0000000e+00,   8.3992305e-06, - 2.8858374e-05, - 8.5319764e-05, - 5.4036416e-05,   1.6293519e-04,   4.3670950e-04,   5.0257520e-04,   3.2855556e-04,   2.0579024e-04,   3.2855556e-04,   5.0257520e-04,   4.3670950e-04,   1.6293519e-04, - 5.4036416e-05, - 8.5319764e-05, - 2.8858374e-05,   8.3992305e-06,   0.0000000e+00,   0.0000000e+00,
		- 0.0000000e+00,   0.0000000e+00,   2.4728735e-05,   1.7199104e-05, - 2.9197428e-05, - 8.5319764e-05, - 6.9304815e-05,   6.1141404e-05,   2.3491006e-04,   3.5269368e-04,   3.8883844e-04,   3.5269368e-04,   2.3491006e-04,   6.1141404e-05, - 6.9304815e-05, - 8.5319764e-05, - 2.9197428e-05,   1.7199104e-05,   2.4728735e-05,   0.0000000e+00, - 0.0000000e+00,
		- 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   3.6472655e-05,   1.7199104e-05, - 2.8858374e-05, - 7.4575483e-05, - 8.1185581e-05, - 3.2812420e-05,   3.1048220e-05,   5.9422589e-05,   3.1048220e-05, - 3.2812420e-05, - 8.1185581e-05, - 7.4575483e-05, - 2.8858374e-05,   1.7199104e-05,   3.6472655e-05,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,
		- 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   2.4728735e-05,   8.3992305e-06, - 1.9064050e-05, - 4.5946268e-05, - 5.3760363e-05, - 4.4442205e-05, - 3.7524185e-05, - 4.4442205e-05, - 5.3760363e-05, - 4.5946268e-05, - 1.9064050e-05,   8.3992305e-06,   2.4728735e-05,   0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,
		0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   9.9022245e-06, - 9.9925971e-07, - 1.0537604e-05, - 1.3228926e-05, - 1.2906245e-05, - 1.3228926e-05, - 1.0537604e-05, - 9.9925971e-07,   9.9022245e-06,   0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,
		0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   7.3633207e-06,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00,   0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00, - 0.0000000e+00,   0.0000000e+00,   0.0000000e+00
	);

	kernel = (Mat_<float>(17, 17) <<
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1 ,1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1,
		-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ,-1 ,-1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	/*kernel = (Mat_<float>(11, 11) <<
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1,
	-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1,
	-1, -1, 0, 0, 1, 1, 1, 0, 0, -1, -1,
	-1, -1, 0, 0, 1, 1, 1, 0, 0, -1, -1,
	-1, -1, 0, 0, 1, 1, 1, 0, 0, -1, -1,
	-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1,
	-1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);*/
	//kernel = (Mat_<float>(9, 9) <<
	//	 -1, -1, -1, -1, -1, -1, -1, -1, -1,
	//	-1, 0, 0, 0, 0, 0, 0, 0, -1,
	//	-1, 0, 0, 0, 0, 0, 0, 0, -1,
	//	-1, 0, 0, 1, 1, 1, 0, 0, -1,
	//	-1, 0, 0, 1, 2, 1, 0, 0, -1,
	//	-1, 0, 0, 1, 1, 1, 0, 0, -1,
	//	-1, 0, 0, 0, 0, 0, 0, 0, -1,
	//	-1, 0, 0, 0, 0, 0, 0, 0, -1,
	//	 -1, -1, -1, -1, -1, -1, -1, -1, -1);
	/*kernel = (Mat_<float>(7, 7) <<
		-1, -1, -1, -1, -1, -1, -1,
		-1,  1,  1,  1,  1,  1, -1,
		-1,  1,  1,  1,  1,  1, -1,
		-1,  1,  1,  1,  1,  1, -1,
		-1,  1,  1,  1,  1,  1, -1,
		-1,  1,  1,  1,  1,  1, -1,
		-1, -1, -1, -1, -1, -1, -1);*/
		/*kernel = (Mat_<float>(7, 7) <<
		-1, -1, -1, -1, -1, -1, -1,
		-1,  -1,  -1,  -1,  -1,  -1, -1,
		-1,  -1,  1,  1,  1,  -1, -1,
		-1,  -1,  1,  1,  1,  -1, -1,
		-1,  -1,  1,  1,  1,  -1, -1,
		-1,  -1,  -1,  -1,  -1,  -1, -1,
		-1, -1, -1, -1, -1, -1, -1);*/
	/*kernel = (Mat_<float>(7, 7) <<
		-1, -1, -1, -1, -1, -1, -1,
		-1, 0, 0, 0, 0, 0, -1,
		-1, 0, 1, 1, 1, 0, -1,
		-1, 0, 1, 1, 1, 0, -1,
		-1, 0, 1, 1, 1, 0, -1,
		-1, 0, 0, 0, 0, 0, -1,
		-1, -1, -1, -1, -1, -1, -1);*/

	kernel = (1.0)*kernel;
	Point anchor = (-1, -1);
	double delta = 0;
	int ddepth = CV_64F; //CV_8UC3
	double BORDER_CONSTANT = 0;

	int kernel_size_lap = 1;
	int scale = 1;

	int kernel_size2 = 45;
	double sig = 5, th = (CV_PI/180)*102 , lm = 10; //Wavelength of the sinusoidal factor
	double gm = 0.5, ps = 0;
	Mat kernel_gabor = cv::getGaborKernel(cv::Size(kernel_size2, kernel_size2), sig, th, lm, gm, ps);

	cout << "aaa" << kernel_gabor;
	cout << "sum" << sum(kernel_gabor);
	cout << "abs sum" << sum(abs(kernel_gabor));
	////kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);
	////flip(kernel, kernel, -1); // if kernel is not symmetric
	//// anchor Point
	//Point anchor(kernel.cols - kernel.cols / 2 - 1, kernel.rows - kernel.rows / 2 - 1);
	//// flip the kernel
	//Mat fl_kernel;
	//int flip_code = -1; // -1-both axis, 0-x-axis, 1-y-axis
	//cv::flip(kernel, fl_kernel, flip_code);
	//int borderMode = BORDER_CONSTANT;
	//filter2D(src, dst, src.depth(), fl_kernel, anchor, borderMode);

	/// Apply filter
	//GaussianBlur(src, src, Size(15, 15), 0, 0, 4);
	//cv::medianBlur(src, src, 7);

	cvtColor(src, src_gray, CV_BGR2GRAY);
	cout << "original image channels: " << src.channels() << "gray image channels: " << src_gray.channels() << endl;
	// //medianBlur(src_gray, blur_image, 7);
	// //adaptiveThreshold(blur_image, adp_image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 7, -7);
	// //medianBlur(adp_image, ad_thr_med, 7); // For global threshold algorithm after blurring

	src_gray.convertTo(src_gray, CV_64F);
	//filter2D(src_gray, profiltered, ddepth, kernel_pro, anchor, delta, BORDER_CONSTANT); // high-pass'i geçirmeden daha doğru sonuç veriyor gibi
	//Laplacian(src_gray, src_gray, ddepth, kernel_size_lap, scale, delta, BORDER_CONSTANT);
	//filter2D(src_gray, dst, ddepth, kernel, anchor, delta, BORDER_REPLICATE); //
	filter2D(src_gray, dest_gabor, ddepth, kernel_gabor);
	imshow("Gabor Pre",dest_gabor);
	// //filter2D(ad_thr_med, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT); //
	// //imshow("Adaptive Threshold", adp_image);
	// //imshow("Blurred Adaptive Threshold", ad_thr_med);

	//normalize(profiltered, dst3, 0, 255, cv::NORM_MINMAX, CV_64F);
	//dst3.convertTo(dst3, CV_8UC1);
	normalize(dest_gabor, dst4, 0, 255, cv::NORM_MINMAX, CV_64F);
	dst4.convertTo(dst4, CV_8UC1);
	//normalize(dst, dst2, 0, 255, cv::NORM_MINMAX, CV_64F);
	//dst2.convertTo(dst2, CV_8UC1);
	//dst = 10 * dst;

	//imshow("High Pass Filtered", profiltered);
	imshow("Gabor Filtered", dst4);
	//namedWindow("filter2D Demo", CV_WINDOW_AUTOSIZE); imshow("filter2D Demo", dst2);
	/*
	for (int i = 0; i < dst2.rows; i++)
	{
		for (int j = 0; j < dst2.cols; j++)
		{
			double Pixel = dst2.at<uchar>(i, j);

			cout << " " << Pixel;
		}
	}
	*/
	float max=-100000;
	float maxi = 0;
	float maxj = 0;
	for (int i = dst4.rows/4; i < dst4.rows- dst4.rows/4; i++)
	{
		for (int j = dst4.cols / 4; j < dst4.cols - dst4.cols / 4; j++)
		{
			double Peak = dst4.at<uchar>(i, j)+ dst4.at<uchar>(i+1, j)+ dst4.at<uchar>(i-1, j)+ dst4.at<uchar>(i, j+1) + dst4.at<uchar>(i, j-1) + dst4.at<uchar>(i+1, j+1)+ dst4.at<uchar>(i-1, j-1) + dst4.at<uchar>(i-1, j+1) + dst4.at<uchar>(i+1, j-1);
			//cout << " " << Peak;
			if (Peak > max)
			{
				max = Peak;
				maxi = i;
				maxj = j;
			}
		}
	}

	float max2 = -100000;
	float maxi2 = 0;
	float maxj2 = 0;
	for (int i = dst4.rows / 4; i < dst4.rows - dst4.rows / 4; i++)
	{
		for (int j = dst4.cols / 4; j < dst4.cols - dst4.cols / 4; j++)
		{
			double Peak2 = dst4.at<uchar>(i, j) + dst4.at<uchar>(i + 1, j) + dst4.at<uchar>(i - 1, j) + dst4.at<uchar>(i, j + 1) + dst4.at<uchar>(i, j - 1) + dst4.at<uchar>(i + 1, j + 1) + dst4.at<uchar>(i - 1, j - 1) + dst4.at<uchar>(i - 1, j + 1) + dst4.at<uchar>(i + 1, j - 1);
			//cout << " " << Peak;
			if (Peak2 > max2 && max > Peak2)
			{
				max2 = Peak2;
				maxi2 = i;
				maxj2 = j;
			}
		}
	}

	float max3 = -100000;
	float maxi3 = 0;
	float maxj3 = 0;
	for (int i = dst4.rows / 4; i < dst4.rows - dst4.rows / 4; i++)
	{
		for (int j = dst4.cols / 4; j < dst4.cols - dst4.cols / 4; j++)
		{
			double Peak3 = dst4.at<uchar>(i, j) + dst4.at<uchar>(i + 1, j) + dst4.at<uchar>(i - 1, j) + dst4.at<uchar>(i, j + 1) + dst4.at<uchar>(i, j - 1) + dst4.at<uchar>(i + 1, j + 1) + dst4.at<uchar>(i - 1, j - 1) + dst4.at<uchar>(i - 1, j + 1) + dst4.at<uchar>(i + 1, j - 1);
			//cout << " " << Peak;
			if (Peak3 > max3 && max2 > Peak3)
			{
				max3 = Peak3;
				maxi3 = i;
				maxj3 = j;
			}
		}
	}

	cout << "max val: " << max << "max loc (x,y):" << maxj << ","<< maxi << endl;
	cout << "max val2: " << max2 << "max loc2 (x,y):" << maxj2 << "," << maxi2 << endl;
	cout << "max val3: " << max3 << "max loc3 (x,y):" << maxj3 << "," << maxi3 << endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	////////////////////////////////////////////////////////////// minmaxLoc //////////////////////////////////////////////////////////////////////
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;

	minMaxLoc(dst2, &minVal, &maxVal, &minLoc, &maxLoc);

	//cout << "min val : " << minVal << endl;
	cout << "max val: " << maxVal << "max loc" << maxLoc<< endl;
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	Mat src_gray_2;
	cvtColor(src, src_gray_2, CV_BGR2GRAY);

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;

	// Filter by Area.
	params.filterByArea = true;
	params.minArea = 5;

	// Filter by Circularity
	params.filterByCircularity = true;
	params.minCircularity = 0.1;

	// Filter by Convexity
	params.filterByConvexity = true;
	params.minConvexity = 0.87;

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

#if CV_MAJOR_VERSION < 3   // If you are using OpenCV 2

	// Set up detector with params
	SimpleBlobDetector detector(params);

	// You can use the detector this way
	// detector.detect( im, keypoints);

#else

	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// SimpleBlobDetector::create creates a smart pointer.
	// So you need to use arrow ( ->) instead of dot ( . )
	// detector->detect( im, keypoints);

#endif
	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector->detect(src_gray_2, keypoints);

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;
	drawKeypoints(src_gray_2, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	imshow("keypoints", im_with_keypoints);

	*/

	////////////////////////////////////////////////////////////// HISTOGRAM //////////////////////////////////////////////////////////////////////
	// Initialize parameters
	// int histSize = 510;    // bin size
	// float range[] = { -255, 255 };
	// const float *ranges[] = { range };

	// Calculate histogram
	/*
	MatND hist;
//	calcHist(&dst, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	// Show the calculated histogram in command window
	double total;
	total = dst.rows * dst.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = hist.at<float>(h);
		cout << " " << binVal;
	}

	// Plot the histogram
	int hist_w = 502; int hist_h = 400000;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	// namedWindow("Result", 1);    imshow("Result", histImage);
	*/
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////// PIXEL MODIFICATIONS //////////////////////////////////////////////////////////////
	// ******************* READ the Pixel intensity *********************
	// single channel grey scale image (type 8UC1) and pixel coordinates x=5 and y=2
	// by convention, {row number = y} and {column number = x}
	// intensity.val[0] contains a value from 0 to 255
/*
	Scalar intensity1 = src_gray.at<uchar>(2, 5);
	cout << "Intensity = " << endl << " " << intensity1.val[0] << endl << endl;

	// 3 channel image with BGR color (type 8UC3)
	// the values can be stored in "int" or in "uchar". Here int is used.
	Vec3b intensity2 = src.at<Vec3b>(10, 15);
	int blue = intensity2.val[0];
	int green = intensity2.val[1];
	int red = intensity2.val[2];
	cout << "Intensity = " << endl << " " << blue << " " << green << " " << red << endl << endl;

	// ******************* WRITE to Pixel intensity **********************
	// This is an example in OpenCV 2.4.6.0 documentation
	Mat H(10, 10, CV_64F);
	for (int i = 0; i < H.rows; i++)
		for (int j = 0; j < H.cols; j++)
			H.at<double>(i, j) = 1. / (i + j + 1);
	cout << H << endl << endl;

	// Modify the pixels of the BGR image
	for (int i = 100; i<src.rows; i++)
	{
		for (int j = 100; j<src.cols; j++)
		{
			src.at<Vec3b>(i, j)[0] = 0;
			src.at<Vec3b>(i, j)[1] = 200;
			src.at<Vec3b>(i, j)[2] = 0;
		}
	}
	// namedWindow("Modify pixel", CV_WINDOW_AUTOSIZE);
	// imshow("Modify pixel", src);
	*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////// FIND MIN OR MAX LINE DISTANCE //////////////////////////////////////////////////////
/*	Mat srcblur2, cdst2, cdst3;
	int kernel_size_2 = 1;
	int scale_2 = 1;
	char* window_name = "Laplace Demo";
	GaussianBlur(src, srcblur2, Size(7, 7), 0, 0, 4);
	cvtColor(srcblur2, src_gray, CV_BGR2GRAY);
	Laplacian(src_gray, dst2, ddepth, kernel_size_2, scale_2, delta, BORDER_DEFAULT);
	convertScaleAbs(dst2, cdst2);
	cdst2 = cdst2 > 8;
	cvtColor(cdst2, cdst3, CV_GRAY2BGR);
	imshow(window_name, cdst3);
	cout << cdst2.type() << endl;

	vector<Vec4i> lines;
	HoughLinesP(cdst2, lines, 1, CV_PI / 180, 60, 50, 10);
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		if (theta>CV_PI / 180 * 80 && theta<CV_PI / 180 * 100) // horizontal
			// theta>CV_PI/180*170 || theta<CV_PI/180*10
		{
			Point pt1, pt2;
			Vec4i l = lines[i];
			line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, CV_AA);
			std::cout << "Laplacian Lines:" << lines[i] << ' ' << '\n';
		}
	}
	imshow("detected lines 2", src);
	imshow("Gaussian Blurred Demo", srcblur2);
	imshow("Laplacian Demo", cdst2);*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////// FIND DIRECTION OF IMAGE //////////////////////////////////////////////////////
/*
double height = src.rows;
double width = src.cols;
if (width > height)
{
	double buf = (width - height) / 2;
	src = src[1:height,(buf+1):(width-buf)];
}
else
{
	double buf2 = (height - width) / 2;
	src = src[(buf2 + 1) : (height - buf2), 1 : width];
}

double height_new = src.rows;
double width_new = src.cols;
src = src - mean(src);
// hp.txt dosyası ile oluşturulan görüntü bu kodda "profiltered" olarak tanımlanmıştır ve filter2D kullanılmıştır
// matlabdaki "fim" burada "profiltered" olarak yer almaktadır


// abs(fftshift(ifft2(fft2(H).*conj(fft2(H)))))./(n*m); kısmını fonksiyon olarak değil direk uygulamak istedim ancak c++'da "fft" yok
// onun yerine "dft" kullanırım dedim ancak bu sefer de "fftshift" yerine nasıl bir şey uygulayacağımı bulamadım. "conj" var c++'da sanırım o
// Matlab'daki ile aynıdır.
//...

*/
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}

int main(int argc, char** argv)
{
	Mat src;

	/// Load an image
	src = imread("./TDRS_4.jpg"); //CV_LOAD_IMAGE_GRAYSCALE otomatik olarak grayscale alıyor
	if (!src.data) { return -1; }

	conv2(src, 3);

	waitKey(0);
	return 0;
}
