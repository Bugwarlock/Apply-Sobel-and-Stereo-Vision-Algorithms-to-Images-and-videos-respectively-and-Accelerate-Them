#include <iostream>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <intrin.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <ipp.h>
#include "stdio.h"
using namespace std;
using namespace cv;
#define BETA 45
#define W 3
int main()
{
	IplImage *in_imgR;
	IplImage *in_imgL;
	IplImage *out_img;
	IplImage *out_img2;
	unsigned char *in_imageR;
	unsigned char *in_imageL;
	unsigned char *out_image;
	unsigned char *pSrcR;
	unsigned char *pSrcL;
	unsigned char *pRes;
	unsigned int NROWS, NCOLS;
	Ipp64u start, end;
	Ipp64u time1, time2;
	Mat inr = imread("./im1.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat inl = imread("./im0.png", CV_LOAD_IMAGE_GRAYSCALE);
	in_imgR = cvLoadImage("./im1.png", CV_LOAD_IMAGE_GRAYSCALE);
	in_imgL = cvLoadImage("./im0.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img = cvarrToMat(in_imgR);
	NROWS = img.rows;
	NCOLS = img.cols;
	out_img2 = cvCreateImage(CvSize(NCOLS, NROWS), IPL_DEPTH_8U, 1);
	if (!in_imgR | !in_imgL)  // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	Mat out;
	equalizeHist(inr, inr);
	equalizeHist(inl, inl);
	out = Mat::zeros(inr.rows, inr.cols, CV_8U);
	long int sum[BETA] = { 0 };
	long int min = 2300;
	int d;
	int re = BETA;
	start = ippGetCpuClocks();		//start time for serial
	for (int i = 1; i < inr.rows; i++) {
		re = BETA;
		for (int j = 1; j < inr.cols; j++) {
			min = 2300;
			d = 0;
			if ((inr.cols - j) < BETA)
				re--;
			else
				re = BETA;
			for (int k = 0; k < re; k++) {
				sum[k] = 0;
				for (int l = (i - ((W - 1) / 2)); l < (1 + i + ((W - 1) / 2)); l++) {
					for (int m = (j - ((W - 1) / 2)); m < (1 + j + ((W - 1) / 2)); m++) {
						sum[k] += abs(inr.at<uchar>(l, m) - inl.at<uchar>(l, m + k));
					}
				}
				if (sum[k] <= min) {
					min = sum[k];
					d = k;
				}
			}
			out.at<uchar>(i, j) = (255 - (d * 255 / BETA));
		}
	}
	end = ippGetCpuClocks();		// end time for serial
	time1 = (end - start) / 1000;
	cout << "Execution time of Serial: " << (Ipp32s)time1 << " s" << endl;
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	imshow("output", out);
	namedWindow("in", CV_WINDOW_AUTOSIZE);
	imshow("in", inr);
	pSrcL = (unsigned char *)in_imgL->imageData;
	pSrcR = (unsigned char *)in_imgR->imageData;
	pRes = (unsigned char *)out_img2->imageData;
	__m128i rs2;
	__m128i rs3;
	__m128 in1, in2, in3, in4, in5, in6, rs1;
	const size_t wsize = 4;
	int step = NROWS;
	const int off[wsize] = { -1, +1, 0, +step };
	start = ippGetCpuClocks();		//start time for parallel
	__m128i minsum = _mm_set1_epi8(0xff);
	__m128i setone = _mm_set1_epi8(1);
	__m128i ks = _mm_set1_epi8(0);
	__m128i setff = _mm_set1_epi8(220);
	__m128i kb = _mm_set1_epi8(0);
	for (int i = 1; i < NROWS - 1; i++) {
		for (int j = 0; j < NCOLS - 16; j += 16) {
			minsum = _mm_set1_epi8(0xff);
			ks = _mm_set1_epi8(0);
			if ((NCOLS / 16 - j) < BETA)
				re = BETA - NCOLS - j * 16;
			else
				re = BETA;
			for (int k = 0; k < re; k++) {
				__m128i mr1 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j - 1));
				__m128i mr2 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + 1));
				__m128i mr3 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j));
				__m128i mr4 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + NCOLS - 1));
				__m128i mr5 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + NCOLS + 1));
				__m128i mr6 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + NCOLS));
				__m128i mr7 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + 2 * (NCOLS - 1)));
				__m128i mr8 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + 2 * (NCOLS + 1)));
				__m128i mr9 = _mm_loadu_si128((const __m128i*)(pSrcR + (i - 1)*NCOLS + j + 2 * NCOLS));
				__m128i ml1 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j - 1));
				__m128i ml2 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + 1));
				__m128i ml3 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j));
				__m128i ml4 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + NCOLS - 1));
				__m128i ml5 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + NCOLS + 1));
				__m128i ml6 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + NCOLS));
				__m128i ml7 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + 2 * (NCOLS - 1)));
				__m128i ml8 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + 2 * (NCOLS + 1)));
				__m128i ml9 = _mm_loadu_si128((const __m128i*)(pSrcL + (i - 1)*NCOLS + k + j + 2 * NCOLS));
				__m128i sum = _mm_adds_epi8(_mm_add_epi8(_mm_add_epi8(_mm_add_epi8(_mm_sub_epi8(mr1, ml1), _mm_sub_epi8(mr2, ml2)), _mm_add_epi8(_mm_sub_epi8(mr3, ml3), _mm_sub_epi8(mr4, ml4))),
					_mm_add_epi8(_mm_add_epi8(_mm_sub_epi8(mr5, ml5), _mm_sub_epi8(mr6, ml6)), _mm_add_epi8(_mm_sub_epi8(mr7, ml7), _mm_sub_epi8(mr8, ml8)))), _mm_sub_epi8(mr9, ml9));
				__m128i cmp = _mm_cmplt_epi8(minsum, sum);
				kb = _mm_add_epi8(cmp, setone);
				ks = _mm_add_epi8(kb, ks);
				minsum = _mm_min_epi8(sum, minsum);
			}
			ks = _mm_adds_epi8(_mm_add_epi8(_mm_add_epi8(_mm_add_epi8(_mm_add_epi8(_mm_add_epi8(ks, _mm_add_epi8(_mm_add_epi8(ks, ks), _mm_add_epi8(ks, ks))),
				_mm_add_epi8(ks, ks)), _mm_add_epi8(ks, _mm_add_epi8(ks, ks))), _mm_add_epi8(ks, ks)), _mm_adds_epi8(ks, ks)), _mm_adds_epi8(_mm_adds_epi8(ks, ks), _mm_adds_epi8(ks, ks)));
			_mm_store_si128((__m128i*)(pRes + (i - 1)*NCOLS + j), ks);
		}
	}
	end = ippGetCpuClocks();
	time2 = (end - start) / 1000;
	cout << "Execution time of Serial: " << (Ipp32s)time2 << " s" << endl;
	cout << "Speedup = " << (Ipp32s)(time1) / (Ipp32s)(time2) << endl;
	IplImage * out_show_p = cvCreateImage(cvSize((int)(out_img2->width),
		(int)(out_img2->height)),
		out_img2->depth,
		out_img2->nChannels);
	cvResize(out_img2, out_show_p);
	namedWindow("output of Parallel Code", CV_WINDOW_AUTOSIZE); // Create a window for display.
	cvShowImage("output of Parallel Code", out_show_p);
	waitKey(0);
	return (0);
}
