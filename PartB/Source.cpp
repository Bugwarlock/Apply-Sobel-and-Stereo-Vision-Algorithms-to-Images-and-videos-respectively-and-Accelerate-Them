#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include "ipp.h"
#include <emmintrin.h>
#include <xmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include "mmintrin.h"
#include <intrin.h>

using namespace cv;
using namespace std;

int main()
{
	IplImage *in_img;
	IplImage *out_img_s;
	IplImage *out_img_p;

	unsigned char *in_image;
	unsigned char *out_image_s;
	unsigned char *out_image_p;

	Ipp64u start, end;
	Ipp64u time1, time2;
	// LOAD image
	//in_img = cvLoadImage("Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);	// Read image
	in_img = cvLoadImage("girlface.bmp", CV_LOAD_IMAGE_GRAYSCALE);	// Read image

	Mat image = cvarrToMat(in_img);

	cout << "first input image width: " << image.rows << endl;
	cout << "first input image height: " << image.cols << endl;

	out_img_s = cvCreateImage(CvSize(image.rows, image.cols), IPL_DEPTH_8U, 1);
	out_img_p = cvCreateImage(CvSize(image.rows, image.cols), IPL_DEPTH_8U, 1);

	in_image = (unsigned char *)in_img->imageData;
	out_image_s = (unsigned char *)out_img_s->imageData;
	out_image_p = (unsigned char *)out_img_p->imageData;


	start = ippGetCpuClocks();

	int x = 0, y = 0;
	int xG = 0, yG = 0;
	for (unsigned int i = 0; i < image.rows * image.cols; i++) {
		x = i % image.cols;			// x = cols
		if (i != 0 && x == 0) {		// y = rows
			y++;
		}

		if (x < (image.cols - 1) && y < (image.rows - 1) && (y > 0) && (x > 0)) {
			//Finds the horizontal gradient
			xG = (in_image[(x + 1) + ((y - 1) * image.cols)] +
				(2 * in_image[(x + 1) + (y *    image.cols)]) +
				in_image[(x + 1) + ((y + 1) *   image.cols)] -
				in_image[(x - 1) + ((y - 1) *   image.cols)] -
				(2 * in_image[(x - 1) + (y *    image.cols)]) -
				in_image[(x - 1) + ((y + 1) *   image.cols)]);
			//Finds the vertical gradient
			yG = (in_image[(x - 1) + ((y + 1) * image.cols)] +
				(2 * in_image[(x)+((y + 1) *    image.cols)]) +
				in_image[(x + 1) + ((y + 1) *   image.cols)] -
				in_image[(x - 1) + ((y - 1) *   image.cols)] -
				(2 * in_image[(x)+((y - 1) *    image.cols)]) -
				in_image[(x + 1) + ((y - 1) *   image.cols)]);

			*(out_image_s + i) = sqrt((xG * xG) + (yG * yG));
		}
		else {
			//Pads out of bound in_image with 0
			*(out_image_s + i) = 0;
		}
	}

	end = ippGetCpuClocks();

	time1 = (end - start) / 1000;
	cout << "Execution time of Serial: " << (Ipp32s)time1 << " s" << endl;

	__m128i *pSrc;
	__m128i x_gradient = _mm_setzero_si128();
	__m128i y_gradient = _mm_setzero_si128();
	__m128i m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11 = _mm_setzero_si128();

	start = ippGetCpuClocks();

	for (int i = 1; i < image.rows - 1; i++)		// = y
		for (int j = 1; j < image.cols - 1; j += 16) {		// = x

			//Gx
			m1 = _mm_loadu_si128((__m128i *)(in_image + (i - 1) * image.cols + (j + 1)));
			m2 = _mm_loadu_si128((__m128i *)(in_image + (i)* image.cols + (j + 1)));
			m3 = _mm_loadu_si128((__m128i *)(in_image + (i + 1) * image.cols + (j + 1)));
			m4 = _mm_loadu_si128((__m128i *)(in_image + (i - 1) * image.cols + (j - 1)));
			m5 = _mm_loadu_si128((__m128i *)(in_image + (i)* image.cols + (j - 1)));
			m6 = _mm_loadu_si128((__m128i *)(in_image + (i + 1) * image.cols + (j - 1)));
			m7 = _mm_add_epi8(_mm_add_epi8(_mm_add_epi8(m2, m2), m1), m3);
			x_gradient = _mm_sub_epi8(_mm_sub_epi8(_mm_sub_epi8(_mm_sub_epi8(m7, m4), m5), m5), m6);

			//Gy
			m1 = _mm_loadu_si128((__m128i *)(in_image + (i + 1) * image.cols + (j - 1)));
			m2 = _mm_loadu_si128((__m128i *)(in_image + (i + 1) * image.cols + (j)));
			m3 = _mm_loadu_si128((__m128i *)(in_image + (i + 1) * image.cols + (j + 1)));
			m4 = _mm_loadu_si128((__m128i *)(in_image + (i - 1) * image.cols + (j - 1)));
			m5 = _mm_loadu_si128((__m128i *)(in_image + (i - 1) * image.cols + (j)));
			m6 = _mm_loadu_si128((__m128i *)(in_image + (i - 1) * image.cols + (j + 1)));
			m7 = _mm_add_epi8(_mm_add_epi8(_mm_add_epi8(m2, m2), m1), m3);
			y_gradient = _mm_sub_epi8(_mm_sub_epi8(_mm_sub_epi8(_mm_sub_epi8(m7, m4), m5), m5), m6);

			m8 = _mm_adds_epu8(_mm_abs_epi8(x_gradient), _mm_abs_epi8(y_gradient));
			_mm_storeu_si128((__m128i *)(out_image_p + i * image.cols + j), m8);
		}

	end = ippGetCpuClocks();


	time2 = (end - start) / 1000;
	cout << "Execution time of Parallel: " << (Ipp32s)time2 << " s" << endl;

	//DISPLAY image
	IplImage * in_show = cvCreateImage(cvSize((int)(in_img->width / 2),
		(int)(in_img->height / 2)),
		in_img->depth,
		in_img->nChannels);
	IplImage * out_show_s = cvCreateImage(cvSize((int)(out_img_s->width / 2),
		(int)(out_img_s->height / 2)),
		out_img_s->depth,
		out_img_s->nChannels);
	IplImage * out_show_p = cvCreateImage(cvSize((int)(out_img_p->width / 2),
		(int)(out_img_p->height / 2)),
		out_img_p->depth,
		out_img_p->nChannels);
	cvResize(in_img, in_show);
	cvResize(out_img_s, out_show_s);
	cvResize(out_img_p, out_show_p);
	namedWindow("input1", CV_WINDOW_AUTOSIZE);  // Create a window for display.
	cvShowImage("input1", in_show); 			// Show input image inside it.
	namedWindow("output of Serial Code", CV_WINDOW_AUTOSIZE);
	cvShowImage("output of Serial Code", out_show_s);
	namedWindow("output of Parallel Code", CV_WINDOW_AUTOSIZE);
	cvShowImage("output of Parallel Code", out_show_p);

	cout << "Speedup = " << (Ipp32s)(time1) / (Ipp32s)(time2) << endl;

	waitKey(0);
	return 0;
}
