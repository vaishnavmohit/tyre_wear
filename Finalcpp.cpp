#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>     /* atoi */
#include <vector>
#include <cctype>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>



// g++ Finalcpp.cpp -o Finalcpp `pkg-config --cflags --libs opencv`
// ./Finalcpp 2340 image_1 image_2 8.33

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

float roundtoPrecision(float val,int precision)
{
     // Do initial checks
     float output = roundf(val * pow(10,precision))/pow(10,precision);
     return output;
}

int main(int argc, char** argv)
{
    Mat image1, image2;
    image1 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);   // Read the files
    image2 = imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
    
    cout << "Rows is " << image1.rows << " and Columns of an image is " << image1.cols <<"\n" ;

    if(! image1.data || ! image2.data )                              // Check for invalid input
    {
        cout << "Could not open or find the image" << endl ;
        return -1;
    }

    // Histogram equalization:
    Mat img_1;
    equalizeHist( image1, img_1 );

    Mat img_2;
    equalizeHist( image2, img_2 );
    
    // SIFT
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();

	//-- Step 1: Detect the keypoints:
	vector<KeyPoint> keypoints_1, keypoints_2;    
	f2d->detect( img_1, keypoints_1 );
	f2d->detect( img_2, keypoints_2 );

	//-- Step 2: Calculate descriptors (feature vectors)    
	Mat descriptors_1, descriptors_2;    
	f2d->compute( img_1, keypoints_1, descriptors_1 );
	f2d->compute( img_2, keypoints_2, descriptors_2 );
	
	vector< vector< DMatch >  > matches;
    vector< DMatch > good_matches;
    
    FlannBasedMatcher flannMatcher (new flann::KDTreeIndexParams(5), new flann::SearchParams(50)); // 4,64
    flannMatcher.knnMatch(descriptors_1, descriptors_2, matches, 2);
    
    //---New Addition---//
    for (int i = 0; i < keypoints_1.size(); ++i)
    {
        if (matches[i].size() < 2)
            continue;

        const DMatch &m1 = matches[i][0];
        const DMatch &m2 = matches[i][1];

        if (m1.distance <= 0.8 * m2.distance)
            good_matches.push_back(m1);
    }
    
	

	//-- Localize the object
	vector<Point2f> pts1;
	vector<Point2f> pts2;
	
	for( size_t i = 0; i < good_matches.size(); i++ )
	{
	//-- Get the keypoints from the good matches
		pts1.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
		pts2.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
	}
	
	if(pts1.size() != pts2.size())
    {
    cerr << "There must be the same number of points in both files (since they are correspondences!). File1 has " << pts1.size() << " while file2 has " << pts2.size() << std::endl;
    return -1;
    }
   
	vector<uchar> inliers(pts1.size(),0);
	vector<unsigned char> match_mask;
	Mat fundamental_matrix = findFundamentalMat(pts1, pts2, FM_RANSAC, 1, 0.9, match_mask);
	// Mat fundamental_matrix = findFundamentalMat(pts1, pts2, FM_LMEDS, match_mask); //CV_FM_8POINT,FM_LMEDS
	
	
	//------using mask to get the new points:--------//
	
	vector<Point2f> pts1_n;
	vector<Point2f> pts2_n;
	
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		if (match_mask[i])
		{
			pts1_n.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
			pts2_n.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
		}
	}
    
    Mat H1(4,4, CV_32F);
	Mat H2(4,4, CV_32F);
	stereoRectifyUncalibrated(pts1_n, pts2_n, fundamental_matrix, img_1.size(), H1, H2);
	
	// Creating folder:
	
	if(mkdir("rectified",0777)==-1)//creating a directory
	{
		cerr<<"Error :  "<<strerror(errno)<<endl;
		//exit(1);
	}
	
	if(mkdir("depth",0777)==-1)//creating a directory
	{
		cerr<<"Error :  "<<strerror(errno)<<endl;
		//exit(1);
	}
	
	string name1( getenv("PWD") );
	name1.append( "/rectified/" );
	name1.append( argv[2] );
	string name2( getenv("PWD") );
	name2.append( "/rectified/" );
	name2.append( argv[3] );
	string depth_stereo( getenv("PWD") );
	depth_stereo.append( "/depth/" );
	depth_stereo.append( argv[2] );
	
	Mat rectified1(img_1.size(), CV_8U);
	warpPerspective(img_1, rectified1, H1, img_1.size());
	imwrite(name1, rectified1);

	Mat rectified2(img_2.size(), CV_8U);
	warpPerspective(img_2, rectified2, H2, img_2.size());
	imwrite(name2, rectified2);
	
	// StereoSBGM
	Ptr<StereoSGBM> stereo = StereoSGBM::create	(1, 256, 19, 0, 0, -1, 0, 12, 7, 2, StereoSGBM::MODE_SGBM);	


	Mat disp(rectified1.size(), CV_8U);
	Mat vdisp(rectified1.size(), CV_8U);
	
	stereo->compute( rectified1, rectified2, disp);
	normalize(disp, vdisp, 0, 255, CV_MINMAX, CV_8U);

	imwrite(depth_stereo, vdisp);
	

    int intensity;	

	for (int i = 0; i < vdisp.rows; i++)
	{
		for (int j = 0; j < vdisp.cols; j++)
		{
			intensity = (int)vdisp.at<uchar>((i,j));
		}
	}
	
	
	double dep_val;
	
	Mat dep_n(vdisp.size(), CV_32FC1);
	Mat dep(vdisp.size(), CV_32FC1);
	
	////////// Reciprocal in a matrix ////////////////
	for (int i = 0; i < dep.rows; i++)
	{
		for (int j = 0; j < dep.cols; j++)
		{
			dep.at<float>((i,j)) = 1.0f / ((int)vdisp.at<uchar>((i,j))+.0001) ;
		}
	}
	
	dep_n = dep * vdisp.cols * 1.2 * 100 / atoi(argv[1]);
	
	float maxval = 0;
	
	for (int i = 0; i < dep.rows; i++)
	{
		for (int j = 0; j < dep.cols; j++)
		{
			dep_val = (float)dep_n.at<float>((i,j));
			float val = roundtoPrecision(dep_val, 3);
			if(val >= atof(argv[4]))
			{
				dep_n.at<float>((i,j)) = 0.0;
			}
			else
			{
				dep_n.at<float>((i,j)) = val;
				if (val > maxval)
				{
					maxval = val;
				}
			}
		}
	}

	int counter = 0;
	float depth = 0.0;
	
	for (int j = 0; j < dep.rows; j++)
	{
	for (int k = 0; k < dep.cols; k++)
		{
			dep_val = (float) dep_n.at<float>((j,k));
			if(dep_val > 0.0)
			{
				depth = depth + dep_val;
				counter = counter+1;
			}
		}
	}

	float depth_n = depth/counter; 
	depth_n = maxval - depth_n;
	cout << "Depth is: " << depth_n << " and Max value is: " << maxval <<" and counter is: " << counter << endl;
	
    return 0;
}
