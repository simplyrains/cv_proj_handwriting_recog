//
//  main.cpp
//  OpenCV
//
//  Created by Sarin Achawaranont on 11/20/2557 BE.
//  Copyright (c) 2557 Sarin Achawaranont. All rights reserved.
//


#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <stdio.h>
#include <time.h>
#include <vector>
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core.hpp"
#include <fstream>
#include <queue>
#define M 16
#define N 16
#define K 4
#define TRAINING_SAMPLES 5400       //Number of samples in training dataset
#define ATTRIBUTES 128  //Number of pixels per sample.16X4X2
#define TEST_SAMPLES 1890       //Number of samples in test dataset
#define CLASSES 36                  //Number of distinct labels.
#define XML_PATH "param2.xml" //Path of ANN XML
#define STATE_UNFILL 2
#define STATE_UNFILL_D 5
#define STATE_ENABLE 1
#define STATE_DISABLE 0
#define STATE_INITIAL 4

using namespace cv;
using namespace std;

typedef struct{
	Rect area;
	string val;
	int score;
	int x;
	int y;
	int type;
}Answer;


int totalscore = 0;
vector<Answer> answer, answer2;
int e_dist = 5;
Mat display, src, display2;

class comparator{
public:
	bool operator()(vector<Point> c1, vector<Point>c2){

		return boundingRect(Mat(c1)).x<boundingRect(Mat(c2)).x;

	}

};

class comparator2{
public:
	bool operator()(vector<vector<Point>> c1, vector<vector<Point>>c2){

		return boundingRect(Mat(c1[0])).x<boundingRect(Mat(c2[0])).x;

	}

};

bool isR1overlapR2(Rect r1, Rect r2, double overlapPercent){
	//r1 comes before r2
	double overlapLength;
	if (r1.x < r2.x){
		if (r1.x + r1.width < r2.x) return false; //r2 does not overlap r1
		overlapLength = r1.x + r1.width - r2.x;
		if (overlapLength / (r1.width)>overlapPercent || overlapLength / (r2.width)>overlapPercent) return true;
		return false;
	}
	else if (r2.x<r1.x){  //r2 comes before r1
		if (r2.x + r2.width < r1.x) return false; //r1 does not overlap r2
		overlapLength = r2.x + r2.width - r1.x;
		if (overlapLength / (r1.width)>overlapPercent || overlapLength / (r2.width)>overlapPercent) return true;
		return false;
	}
	else{//r1.x =r2.x
		return true;
	}
}
///////////////////////////////////////////////////////////////// PRAGMA: OCR+SPLIT WORD TO CHAR

void findConnectedComponent(Mat &binaryImg, vector<Mat> &output){

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	//preprocessing
	Mat bitwiseNotImg;
	cv::bitwise_not(binaryImg, bitwiseNotImg);

	/// Find contours
	findContours(bitwiseNotImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Approximate contours to polygons + get bounding rects
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
	}

	// Get only important contours, merge contours that are within another or cover 80% of another contour in x
	vector<vector<bool>> overlapVector;
	for (int i = 0; i<contours_poly.size(); i++){
		vector<bool> overlap;
		Rect r = boundingRect(Mat(contours_poly[i]));
		//check if any contour i is overlap with any contour j, if it is, we push back 1, 0 otherwise
		for (int j = 0; j < contours_poly.size(); j++){
			if (j == i){
				overlap.push_back(false); continue;
			}
			Rect r2 = boundingRect(Mat(contours_poly[j]));
			if (isR1overlapR2(r, r2, 0.8)){
				overlap.push_back(true);
			}
			else{
				overlap.push_back(false);
			}
		}
		overlapVector.push_back(overlap);
	}

	//Suggest that we don't merge contour and draw, but just draw it multiple times
	//Find appropriate set of contours
	vector<vector<vector<Point>>> diffContourSet;
	vector<bool> alreadyMerge(contours.size()); //initialize vector of bool with false
	for (int i = 0; i < contours.size(); i++){
		if (alreadyMerge[i] == true)continue;
		vector<vector<Point>> contourSet;
		int thisContour;
		std::queue<int> myQueue;
		myQueue.push(i);
		while (myQueue.size()>0){
			thisContour = myQueue.front();
			myQueue.pop();
			if (alreadyMerge[thisContour] == true) continue;
			contourSet.push_back(contours[thisContour]);
			for (int j = 0; j < contours.size(); j++){
				if (overlapVector[thisContour][j] == true) myQueue.push(j);
			}
			alreadyMerge[thisContour] = true;
		}
		diffContourSet.push_back(contourSet);
	}

	//Sort contours inside each contour set for x
	for (int i = 0; i < diffContourSet.size(); i++){
		sort(diffContourSet[i].begin(), diffContourSet[i].end(), comparator());
	}
	//Sort each contour set for x
	sort(diffContourSet.begin(), diffContourSet.end(), comparator2());

	//Making merge set of contours to estimate the bounding rectagle
	vector<vector<Point>> mergedContours;
	for (int i = 0; i < diffContourSet.size(); i++){
		vector<Point> points;
		for (int j = 0; j < diffContourSet[i].size(); j++){
			for (int k = 0; k < diffContourSet[i][j].size(); k++){
				points.push_back(diffContourSet[i][j][k]);
			}
		}
		mergedContours.push_back(points);
	}


	//Loop through all contours to extract
	for (int i = 0; i < diffContourSet.size(); i++){

		//Find boundingRect for the mask
		Rect r = boundingRect(Mat(mergedContours[i]));
		Mat mask = Mat::zeros(binaryImg.size(), CV_8UC1);

		//Draw mask onto image
		for (int j = 0; j < diffContourSet[i].size(); j++){
			drawContours(mask, diffContourSet[i], j, Scalar(255), CV_FILLED);
		}

		Mat extractPic(binaryImg.size(), binaryImg.type(), cv::Scalar(255));
		//Extract the character using the mask
		binaryImg.copyTo(extractPic, mask);
		//resize with fitting regtagle and put into output
		Mat resizedPic = extractPic(r);
		output.push_back(resizedPic);
	}
}

void scaleDownImage(cv::Mat &originalImg, cv::Mat &scaledDownImage)
{
	for (int x = 0; x < M; x++)
	{
		for (int y = 0; y < N; y++)
		{
			int yd = ceil((float)(y*originalImg.cols / M));
			int xd = ceil((float)(x*originalImg.rows / N));
			scaledDownImage.at<uchar>(x, y) = originalImg.at<uchar>(xd, yd);
		}
	}
}

void cropImage(cv::Mat &originalImage, cv::Mat &croppedImage)
{
	int row = originalImage.rows;
	int col = originalImage.cols;
	int tlx, tly, bry, brx;//t=top r=right b=bottom l=left
	tlx = tly = bry = brx = 0;
	float suml = 0;
	float sumr = 0;
	int flag = 0;

	/**************************top edge***********************/
	for (int x = 1; x < row; x++)
	{
		for (int y = 0; y < col; y++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				tly = x;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}
	/*******************bottom edge***********************************/
	for (int x = row - 1; x > 0; x--)
	{
		for (int y = 0; y < col; y++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				bry = x;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}

	}
	/*************************left edge*******************************/

	for (int y = 0; y < col; y++)
	{
		for (int x = 0; x < row; x++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				tlx = y;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}

	/**********************right edge***********************************/

	for (int y = col - 1; y > 0; y--)
	{
		for (int x = 0; x < row; x++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{
				flag = 1;
				brx = y;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}
	int width = brx - tlx;
	int height = bry - tly;
	cv::Mat crop(originalImage, cv::Rect(tlx, tly, brx - tlx, bry - tly));
	croppedImage = crop.clone();
}

void convertToPixelValueArray(cv::Mat &img, int pixelarray[])
{
	int i = 0;
	for (int x = 0; x < M; x++)
	{
		for (int y = 0; y < N; y++)
		{
			pixelarray[i] = (img.at<uchar>(x, y) == 255) ? 1 : 0;
			i++;
		}
	}
}
string convertInt(int number)
{
	stringstream ss;//create a stringstream
	ss << number;//add number to the stream
	string s = ss.str();
	return s;
}

int predict(cv::Mat img)
{
	//read the model from the XML file and create the neural network.
	CvANN_MLP nnetwork;
	CvFileStorage* storage = cvOpenFileStorage(XML_PATH, 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "OCR");
	nnetwork.read(storage, n);
	cvReleaseFileStorage(&storage);

	cv::Mat output;
	//Applying gaussian blur to remove any noise
	cv::GaussianBlur(img, output, cv::Size(5, 5), 0);
	//thresholding to get a binary image
	cv::threshold(output, output, 200, 255, 0);
	//declaring mat to hold the scaled down image
	cv::Mat scaledDownImage(M, N, CV_8U, cv::Scalar(0));

	//cropping the image.
	cropImage(output, output);

	//reducing the image dimension to MXN
	scaleDownImage(output, scaledDownImage);

	//Horizontal
	int pixelValueArray[N*K] = { 0 };
	int q = N / K;
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			if (scaledDownImage.at<uchar>(i, j) == 255){
				pixelValueArray[i + M*((j - 1) / q)] = 1;
				j = q*ceil(1.0 * j / q);
			}
		}
	}
	//Vertical
	int pixelValueArrayV[N*K] = { 0 };
	for (int i = 0; i < M; i++){
		for (int j = 0; j < N; j++){
			if (scaledDownImage.at<uchar>(j, i) == 255){
				pixelValueArrayV[i + M*((j - 1) / q)] = 1;
				j = q*ceil(1.0 * j / q);
			}
		}
	}
	//writing pixel data to file
	cv::Mat data(1, ATTRIBUTES, CV_32F);
	for (int i = 0; i < ATTRIBUTES / 2; i++){
		data.at<float>(0, i) = pixelValueArray[i];
		data.at<float>(0, i + ATTRIBUTES / 2) = pixelValueArrayV[i];
	}
	int maxIndex = 0;
	cv::Mat classOut(1, CLASSES, CV_32F);
	//prediction
	nnetwork.predict(data, classOut);
	float value;
	float maxValue = classOut.at<float>(0, 0);
	for (int index = 0; index<CLASSES; index++)
	{
		value = classOut.at<float>(0, index);
		if (value>maxValue)
		{
			maxValue = value;
			maxIndex = index;
		}
		//printf("%d %f\n", index, value);
	}
	//printf("MAX: %d %f\n", maxIndex, maxValue);
	//maxIndex is the predicted class.
	return maxIndex;
}

string findWord(Mat src)
{
	Mat src_gray;
	/// Convert image to gray and blur it
	src_gray = src.clone();
	cv::Size size(3, 3);
	cv::GaussianBlur(src_gray, src_gray, size, 0);

	Mat binaryImg;
	vector<Mat> characters;
	//change to binary image
	threshold(src_gray, binaryImg, 128, 255, THRESH_BINARY);
	//get output as set of sorted and segmented image
	findConnectedComponent(binaryImg, characters);
	string word = "";
	for (int j = 0; j< characters.size(); j++)
	{
		Mat character = characters[j];
		int result = predict(character);
		if (result <= 9)
			word.push_back(result + '0');
		else
			word.push_back((result - 10) + 'A');
	}
	return word;
}
///////////////////////////////////////////////////////////////// PRAGMA: HELPER FUNC

// HELPER FUNC
int convert_to_range(int val, int max){
	int range = 155;
	int start = 100;
	int result = (int)((double)val*range / (double)max) + start;
	return result;
}

// HELPER FUNC
void hitmiss(cv::Mat& src, cv::Mat& dst, cv::Mat& kernel)
{
	CV_Assert(src.type() == CV_8U && src.channels() == 1);

	cv::Mat k1 = (kernel == 1) / 255;
	cv::Mat k2 = (kernel == -1) / 255;

	cv::normalize(src, src, 0, 1, cv::NORM_MINMAX);

	cv::Mat e1, e2;
	cv::erode(src, e1, k1, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	cv::erode(1 - src, e2, k2, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
	dst = e1 & e2;
}
//
//void draw_circle(cv::Mat& dst, int column, int row, int type){
//    int thickness = -1;
//    int lineType = 8;
//    Scalar color;
//
//    switch (type) {
//        case 0:
//            color =Scalar(0,0,255);
//            break;
//        case 1:
//            color =Scalar(0,255,255);
//            break;
//        case 2:
//            color =Scalar(255,0,255);
//            break;
//        case 3:
//            color =Scalar(255,0,0);
//            break;
//        default:
//            color =Scalar(255,255,255);
//            break;
//    }
//    cv::circle( dst,
//               *new cv::Point(row, column),
//               5,
//               color,
//               thickness,
//               lineType );
//}

// HELPER FUNC
Rect fill(cv::Mat& dst, int column, int row, int type, int val, int max){
	Scalar color;
	Rect com;
	int v = convert_to_range(val, max);

	int ffillMode = 1;
	int loDiff = 20, upDiff = 20;
	int connectivity = 4;
	int lo = ffillMode == 0 ? 0 : loDiff;
	int up = ffillMode == 0 ? 0 : upDiff;
	int newMaskVal = 255;
	int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);

	switch (type) {
	case 0:
		color = Scalar(v, 0, 0);
		break;
	case 5:
		color = Scalar(v, 0, 0);
		break;
	case 1:
		color = Scalar(0, v, 0);
		break;
	case 2:
		color = Scalar(0, 0, v);
		break;
	case 3:
		color = Scalar(v, v, 0);
		break;
	case 4:
		color = Scalar(v, 0, 0);
		break;
	default:
		color = Scalar(255, 255, 255);
		break;
	}
	cv::floodFill(dst, *new cv::Point(row, column), color, &com, Scalar(lo, lo, lo), Scalar(up, up, up), flags);
	return com;
}

// TEST PURPOSE
//void save_to_box(string char_id, string img_name, Mat& src, Rect& boundary, int y, int x){
//    if(boundary.width<70||boundary.height<70) return;
//    Scalar color=Scalar(255);
//    Rect com;
//    
//    int ffillMode = 1;
//    int loDiff = 40, upDiff = 40;
//    int connectivity = 8;
//    int lo = ffillMode == 0 ? 0 : loDiff;
//    int up = ffillMode == 0 ? 0 : upDiff;
//    int newMaskVal = 255;
//    int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);
//    
//    
//    Mat new_img = src(boundary).clone();
//    string name = to_string((int)(y*1000.0/src.rows))+","+to_string((int)(x*1000.0/src.cols));
//    cvtColor(new_img, new_img, CV_BGR2GRAY);
//    
//    cv::threshold(new_img, new_img, 127, 255, CV_THRESH_BINARY);
//    for(int i=0;i<new_img.cols&&i<new_img.rows;i++){
//        if(new_img.at<uchar>(i,i) == 0){
//            cv::floodFill(new_img, *new cv::Point(i,i), color, &com, Scalar(lo, lo, lo),Scalar(up, up, up), flags);
//            break;
//        }
//    }
//    
//    int erosion_type = MORPH_ELLIPSE;
//    int erosion_size = 1;
//    Mat element = getStructuringElement( erosion_type,
//                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
//                                        Point( erosion_size, erosion_size ) );
//    /// Apply the erosion operation
//    erode( new_img, new_img, element );
//    imwrite( char_id+"_"+name+"_"+img_name+".jpg", new_img);
//    
//    //cv::threshold(new_img, new_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
//    
//}
//

// HELPER: String SImilarity computation
// From http://rosettacode.org/wiki/Levenshtein_distance
// Compute Levenshtein Distance
// Martin Ettl, 2012-10-05

bool compareChar(char a, char b){
	if (a == 'I') a = '1';
	if (b == 'I') b = '1';
	if (a == 'O') a = '0';
	if (b == 'O') b = '0';

	return a == b;
}

int edit_distance(string s1, string s2)
{
	unsigned int len1 = s1.size(), len2 = s2.size();
	vector<vector<unsigned int> > d(len1 + 1, vector<unsigned int>(len2 + 1));

	d[0][0] = 0;
	for (unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
	for (unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

	for (unsigned int i = 1; i <= len1; ++i)
	for (unsigned int j = 1; j <= len2; ++j)

		d[i][j] = std::min(std::min(d[i - 1][j] + 1, d[i][j - 1] + 1),
		d[i - 1][j - 1] + (compareChar((s1.at(i-1)), (s2.at(j-1))) ? 0 : 1));
	return d[len1][len2];
}

///////////////////////////////////////////////////////////////// PRAGMA: GRADING MODE
//GRADING MDOE
void render(Mat &display){
	display = (255 - src.clone()) / 2;

	for (int i = 0; i<answer.size(); i++){
		fill(display, answer[i].y, answer[i].x, answer[i].type, answer[i].x, display.cols);

		if (answer[i].type != STATE_DISABLE && answer[i].type != STATE_UNFILL_D)
			putText(display, answer[i].val + " / " + to_string(answer[i].score), answer[i].area.tl() + Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 200), 4);
	}

}

// GRADING MODE
void render2(Mat &display2){
	display2 = (255 - src.clone()) / 2;

	for (int i = 0; i<answer2.size(); i++){
		fill(display2, answer2[i].y, answer2[i].x, answer2[i].type, answer2[i].x, display2.cols);

		if (answer2[i].type == STATE_ENABLE)
			putText(display2, answer2[i].val + " / " + to_string(answer2[i].score), answer2[i].area.tl() + Point(20, 40), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 200), 4);
	}

}


// GRADING MODE - FIND BOUNDARY FOR Grading Image and Predict the word
void check_boundary(Mat& src, Rect& boundary, int y, int x){
	if (boundary.width<70 || boundary.height<70) return;
	Scalar color = Scalar(255);
	Rect com;

	int ffillMode = 1;
	int loDiff = 40, upDiff = 40;
	int connectivity = 8;
	int lo = ffillMode == 0 ? 0 : loDiff;
	int up = ffillMode == 0 ? 0 : upDiff;
	int newMaskVal = 255;
	int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1 ? CV_FLOODFILL_FIXED_RANGE : 0);


	Mat new_img = src(boundary).clone();
	cvtColor(new_img, new_img, CV_BGR2GRAY);

	cv::threshold(new_img, new_img, 127, 255, CV_THRESH_BINARY);
	for (int i = 0; i<new_img.cols&&i<new_img.rows; i++){
		if (new_img.at<uchar>(i, i) == 0){
			cv::floodFill(new_img, *new cv::Point(i, i), color, &com, Scalar(lo, lo, lo), Scalar(up, up, up), flags);
			break;
		}
	}

	int erosion_type = MORPH_ELLIPSE;
	int erosion_size = 1;
	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	/// Apply the erosion operation
	erode(new_img, new_img, element);

	string ans = findWord(new_img);
	//cout << ans << endl;
	int matched_ans = -1;
	int formerdistance = -1;
	int e_distance;
	for (int i = 0; i < answer.size(); i++){
		if (answer[i].type != STATE_ENABLE) continue;
		int distance = abs(answer[i].y - y) + abs(answer[i].x - x);
		//cout << answer[i].val << "/" << ans << "(" << answer[i].x << "," << answer[i].y << ") / (" << x << "," << y << ") - " << distance << endl;
		if (distance > 200) continue;
		if ( distance < formerdistance || formerdistance == -1 ){
			e_distance = edit_distance(answer[i].val, ans);
			formerdistance = distance;
			matched_ans = i;
		}
	}
	Answer a;
	a.area = boundary;
	a.score = 0;
	a.val = ans;
	a.x = x;
	a.y = y;
	a.type = STATE_UNFILL;

	if (formerdistance >= 0){
		a.score = answer[matched_ans].score;
		if (e_distance < e_dist){
			a.type = STATE_ENABLE;
		}
		else{
			a.type = STATE_DISABLE;
		}
		a.val = answer[matched_ans].val + " - " + a.val;
		totalscore += a.score;
		cout << "- MATCH: " + a.val + " | distance = ";
		cout << formerdistance;
		cout << " | score = " + a.score << endl;
	}
	answer2.push_back(a);

	//cv::threshold(new_img, new_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

}
//GRADING MODE
void CallBackFunc2(int event, int x, int y, int flags, void* userdata)
{

	if (event == EVENT_RBUTTONDOWN)
	{
		for (int i = 0; i<answer2.size(); i++){
			Rect rect = answer2[i].area;
			if (rect.contains(Point(x, y))){
				if (answer2[i].type == STATE_ENABLE) answer2[i].type = STATE_DISABLE;
				else if (answer2[i].type == STATE_DISABLE) answer2[i].type = STATE_ENABLE;
			}
		}
		render2(display2);
		imshow("Point", display2);
	}

}
//GRADING MODE
void check_answer(string &img_name)
{

	/// Declare variables
	Mat dst, bw;

	Mat kernel[4], result[5];
	Point anchor;

	src = imread("OCR_Data\\" + img_name);
	if (src.empty()) return;

	// Resize source
	Size s(2240, src.size().height * 2240 / src.size().width);
	resize(src, src, s);

	result[0] = (255 - src.clone()) / 2;
	for (int i = 1; i<5; i++){
		result[i] = result[0].clone();
	}

	cvtColor(src, bw, CV_BGR2GRAY);
	bw = 255 - bw;
	cv::threshold(bw, bw, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);

	/// Initialize arguments for the filter
	anchor = Point(-1, -1);

	/// Create kernel
	/// < ^ (TL: Top Left)
	kernel[0] = (cv::Mat_<char>(11, 11) <<
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1);
	for (int i = 1; i<4; i++){
		kernel[i] = kernel[0].clone();
	}
	// v > (BR: Bottom Right)
	flip(kernel[0], kernel[1], -1);
	// < v (BL: Bottom Left)
	flip(kernel[0], kernel[2], 0);
	// ^ > (TR: Top Right)
	flip(kernel[0], kernel[3], 1);

	/// Apply filter: < ^ (TL)
	hitmiss(bw, dst, kernel[0]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i - 5;
			int y = j - 5;
			if (dst.at<uchar>(j, i) == 1 &&
				/*not filled before*/
				result[0].at<cv::Vec3b>(y, x)[1] != 0){
				fill(result[0], y, x, 0, y, dst.rows);
			}
		}
	}
	/// Apply filter: < v (BL, with respect to TL)
	hitmiss(bw, dst, kernel[2]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i - 5;
			int y = j + 5;
			if (dst.at<uchar>(j, i) == 1 &&
				/*not filled before*/
				result[2].at<cv::Vec3b>(y, x)[0] != 0){

				int b = result[0].at<cv::Vec3b>(y, x)[0];
				int g = result[0].at<cv::Vec3b>(y, x)[1];
				int r = result[0].at<cv::Vec3b>(y, x)[2];

				if (g == 0 &&
					r == 0 &&
					b < convert_to_range(y, dst.rows)){
					fill(result[2], y, x, 2, x, dst.cols);
				}
			}
		}
	}

	/// Apply filter: ^ > (TR)
	hitmiss(bw, dst, kernel[3]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i + 5;
			int y = j - 5;
			if (dst.at<uchar>(j, i) == 1 &&
				/*not filled before*/
				result[3].at<cv::Vec3b>(y, x)[1] != 0){
				fill(result[3], y, x, 3, y, dst.rows);
			}
		}
	}

	/// Apply filter: v > (BR, with respect to TL+BL+TR)
	hitmiss(bw, dst, kernel[1]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i + 5;
			int y = j + 5;
			if (dst.at<uchar>(j, i) == 1 &&
				//not filled before
				result[4].at<cv::Vec3b>(y, x)[0] != 0){

				int b3 = result[3].at<cv::Vec3b>(y, x)[0];
				int g3 = result[3].at<cv::Vec3b>(y, x)[1];
				int r3 = result[3].at<cv::Vec3b>(y, x)[2];

				int b2 = result[2].at<cv::Vec3b>(y, x)[0];
				int g2 = result[2].at<cv::Vec3b>(y, x)[1];
				int r2 = result[2].at<cv::Vec3b>(y, x)[2];

				if (b3 == g3 &&
					r3 == 0 &&
					g3 < convert_to_range(y, dst.rows) &&
					b2 == 0 &&
					g2 == 0 &&
					r2 < convert_to_range(x, dst.cols)){
					Rect boundary = fill(result[4], y, x, 1, x, dst.cols);
					//string char_id = to_string(calc_alp(y, x, threshx, threshy));

					check_boundary(src, boundary, y, x);

					//                    string char_id = to_string(x)+"-"+to_string(y);
					//                    save_to_box(char_id,img_name,src,boundary,y,x);
					//                    Answer a;
					//                    a.area = boundary;
					//                    a.score = 0;
					//                    a.val = "?";
					//                    a.x = x;
					//                    a.y = y;
					//                    a.type = STATE_UNFILL;
					//                    answer.push_back(a);
				}
			}
		}
	}
	render2(display2);
	//Create a window
	namedWindow("Point", CV_WINDOW_KEEPRATIO);
	setMouseCallback("Point", CallBackFunc2, NULL);

	//set the callback function for any mouse event

	imshow("Point", display2);
	waitKey(0);
}

///////////////////////////////////////////////////////////////// PRAGMA: GRADING MODE


//EDIT MODE
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		for (int i = 0; i<answer.size(); i++){
			Rect rect = answer[i].area;
			if (rect.contains(Point(x, y))){
				cout << "<<" << i << ">>" << endl;
				//if(answer[i].type==4){
				cout << "Enter Answer: " << endl;
				string val;
				cin >> val;
				cout << "Enter Score: " << endl;
				int score;
				cin >> score;
				answer[i].score = score;
				answer[i].val = val;
				answer[i].type = STATE_ENABLE;

				//}
				//else if(answer[i].type==2){
				//    answer[i].type=4;
				//}

				//                cout<<"Enter Answer: "<<endl;
				//                string val;
				//                cin >> val;
				//
				//                cout<<"Enter Score: "<<endl;
				//                int score;
				//                cin >> score;
			}
		}
		render(display);
		imshow("Answer", display);

	}
	if (event == EVENT_RBUTTONDOWN)
	{
		for (int i = 0; i<answer.size(); i++){
			Rect rect = answer[i].area;
			if (rect.contains(Point(x, y))){
				if (answer[i].type == STATE_ENABLE) answer[i].type = STATE_DISABLE;
				else if (answer[i].type == STATE_DISABLE) answer[i].type = STATE_ENABLE;
				else if (answer[i].type == STATE_UNFILL) answer[i].type = STATE_UNFILL_D;
				else if (answer[i].type == STATE_UNFILL_D) answer[i].type = STATE_UNFILL;

			}
		}
		render(display);
		imshow("Answer", display);
	}


}

//EDIT MODE
void calc(string img_name)
{

	/// Declare variables
	Mat dst, bw;

	Mat kernel[4], result[5];
	Point anchor;

	src = imread("OCR_Data\\" + img_name);
	if (src.empty()) return;

	// Resize source
	Size s(2240, src.size().height * 2240 / src.size().width);
	resize(src, src, s);

	result[0] = (255 - src.clone()) / 2;
	for (int i = 1; i<5; i++){
		result[i] = result[0].clone();
	}

	cvtColor(src, bw, CV_BGR2GRAY);
	bw = 255 - bw;
	cv::threshold(bw, bw, 0, 1, CV_THRESH_BINARY | CV_THRESH_OTSU);

	/// Initialize arguments for the filter
	anchor = Point(-1, -1);

	/// Create kernel
	/// < ^ (TL: Top Left)
	kernel[0] = (cv::Mat_<char>(11, 11) <<
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1,
		1, 1, 0, 0, 0, -1, -1, -1, -1, -1, -1);
	for (int i = 1; i<4; i++){
		kernel[i] = kernel[0].clone();
	}
	// v > (BR: Bottom Right)
	flip(kernel[0], kernel[1], -1);
	// < v (BL: Bottom Left)
	flip(kernel[0], kernel[2], 0);
	// ^ > (TR: Top Right)
	flip(kernel[0], kernel[3], 1);




	/// Apply filter: < ^ (TL)
	hitmiss(bw, dst, kernel[0]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i - 5;
			int y = j - 5;
			if (dst.at<uchar>(j, i) == 1 &&
				/*not filled before*/
				result[0].at<cv::Vec3b>(y, x)[1] != 0){
				fill(result[0], y, x, 0, y, dst.rows);
			}
		}
	}
	/// Apply filter: < v (BL, with respect to TL)
	hitmiss(bw, dst, kernel[2]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i - 5;
			int y = j + 5;
			if (dst.at<uchar>(j, i) == 1 &&
				/*not filled before*/
				result[2].at<cv::Vec3b>(y, x)[0] != 0){

				int b = result[0].at<cv::Vec3b>(y, x)[0];
				int g = result[0].at<cv::Vec3b>(y, x)[1];
				int r = result[0].at<cv::Vec3b>(y, x)[2];

				if (g == 0 &&
					r == 0 &&
					b < convert_to_range(y, dst.rows)){
					fill(result[2], y, x, 2, x, dst.cols);
				}
			}
		}
	}

	/// Apply filter: ^ > (TR)
	hitmiss(bw, dst, kernel[3]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i + 5;
			int y = j - 5;
			if (dst.at<uchar>(j, i) == 1 &&
				/*not filled before*/
				result[3].at<cv::Vec3b>(y, x)[1] != 0){
				fill(result[3], y, x, 3, y, dst.rows);
			}
		}
	}

	/// Apply filter: v > (BR, with respect to TL+BL+TR)
	hitmiss(bw, dst, kernel[1]);
	for (int j = 0; j<dst.rows; j++)
	{
		for (int i = 0; i<dst.cols; i++)
		{
			int x = i + 5;
			int y = j + 5;
			if (dst.at<uchar>(j, i) == 1 &&
				//not filled before
				result[4].at<cv::Vec3b>(y, x)[0] != 0){

				int b3 = result[3].at<cv::Vec3b>(y, x)[0];
				int g3 = result[3].at<cv::Vec3b>(y, x)[1];
				int r3 = result[3].at<cv::Vec3b>(y, x)[2];

				int b2 = result[2].at<cv::Vec3b>(y, x)[0];
				int g2 = result[2].at<cv::Vec3b>(y, x)[1];
				int r2 = result[2].at<cv::Vec3b>(y, x)[2];

				if (b3 == g3 &&
					r3 == 0 &&
					g3 < convert_to_range(y, dst.rows) &&
					b2 == 0 &&
					g2 == 0 &&
					r2 < convert_to_range(x, dst.cols)){
					Rect boundary = fill(result[4], y, x, 1, x, dst.cols);
					//string char_id = to_string(calc_alp(y, x, threshx, threshy));
					string char_id = to_string(x) + "-" + to_string(y);
					//save_to_box(char_id,img_name,src,boundary,y,x);
					Answer a;
					a.area = boundary;
					a.score = 0;
					a.val = "?";
					a.x = x;
					a.y = y;
					a.type = STATE_UNFILL;
					answer.push_back(a);
				}
			}
		}
	}

	render(display);

	//Create a window
	namedWindow("Answer", CV_WINDOW_KEEPRATIO);
	//set the callback function for any mouse event
	setMouseCallback("Answer", CallBackFunc, NULL);
	imshow("Answer", display);
	// Wait until user press some key
	waitKey(0);
	//imwrite("result_"+img_name+".jpg", result[4]);
}

int main(){
	string img_name = "Z";
	cout << "::Handwriting Recognition for Grading Short Answer Exams::" << endl << endl;
	cout << "Specify EDIT_DISTANCE (0~10): ";
	cin >> e_dist;

	cout << "Enter input image for ANSWER KEY (i.e. Z.PNG): ";
	cin >> img_name;
	cout << "<< Edit Mode >>" << endl << 
		"- Press <LCLICK> inside the selected box to edit answer/score" << endl <<
		"- Press <RCLICK> to enable/disable answer" << endl <<
		"- Close image window [X] when finish editing" << endl << endl;

	calc(img_name);
	cout << "Enter input image for GRADING (i.e. Z.PNG): ";
	cin >> img_name;
	check_answer(img_name);

	cout << "GRADE COMPLETE" << endl;
	int score = 0;
	for (int i = 0; i < answer2.size(); i++) if (answer2[i].type == STATE_ENABLE) score += answer2[i].score;
	cout << "Score: " << score;
	cin >> img_name;
	cv::waitKey(0);
	return 0;
}