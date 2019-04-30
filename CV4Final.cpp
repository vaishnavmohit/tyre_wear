#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/xfeatures2d.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <string>
#include <stdlib.h>     /* atoi */
#include <vector>
#include <cctype>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>


// g++ testFinal.cpp -std=c++11 -o testFinal `pkg-config --cflags --libs opencv4`

// ./testFinal 2340 t22_c_1.jpg t22_c_2.jpg 8.33

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

int main(int argc, char** argv)
{

    Mat image1, image2;
    image1 = imread(argv[2],  IMREAD_GRAYSCALE);   // Read the files
    image2 = imread(argv[3],  IMREAD_GRAYSCALE);
    
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
    // https://stackoverflow.com/questions/27533203/how-do-i-use-sift-in-opencv-3-0-with-c
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
    
	

// https://docs.opencv.org/3.1.0/d7/dff/tutorial_feature_homography.html
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
   
// https://github.com/daviddoria/Examples/blob/master/c%2B%2B/OpenCV/StereoRectifyUncalibrated/StereoRectifyUncalibrated.cxx
	vector<uchar> inliers(pts1.size(),0);
	vector<unsigned char> match_mask;

	Mat fundamental_matrix = findFundamentalMat(pts1, pts2, FM_LMEDS, 3, 0.7, match_mask); //CV_FM_8POINT,FM_LMEDS
	
	//------using mask to get the new points:--------//
	
	vector<Point2f> pts1_n;
	vector<Point2f> pts2_n;
	
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		// match_mask.empty() || match_mask[i]
		if (match_mask[i])
		{
			pts1_n.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
			pts2_n.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
		}
	}
    
	//-- For Fundamental Matrix --//
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

	//-- Name Convention --//
	
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
	Ptr<StereoSGBM> stereo = StereoSGBM::create();

	stereo->setMinDisparity(3);
	stereo->setNumDisparities(256);
	stereo->setBlockSize(19);
	stereo->setSpeckleRange(12);
	stereo->setSpeckleWindowSize(7);

	Mat disp(rectified1.size(), CV_16UC1);
	Mat vdisp(rectified1.size(), CV_8U);
	
	// https://stackoverflow.com/questions/23643813/opencv-stereo-vision-depth-map-code-does-not-work
	stereo->compute( rectified1, rectified2, disp);
	normalize(disp, vdisp, 0, 255, NORM_MINMAX,  CV_8U); 
	
	
	imshow("disp", vdisp);
	waitKey(0);

	imwrite(depth_stereo, vdisp);
	

	//-- Converting Disparity to Real Depth--//
    	int intensity =0;	

	for (int i = 0; i < vdisp.rows; i++)
	{
		for (int j = 0; j < vdisp.cols; j++)
		{
			Scalar intensity1 = vdisp.at<uchar>(i, j);
			if(intensity1.val[0]>150)
			{
				vdisp.at<uchar>((i,j)) = 255;
			}
		}
	}
	
	
	double dep_val;
	
	Mat dep_n(vdisp.size(), CV_32F);
	Mat dep(vdisp.size(), CV_32F);
	
	////////// Reciprocal in a matrix ////////////////
	for (int i = 0; i < dep.rows; i++)
	{
		for (int j = 0; j < dep.cols; j++)
		{	
			Scalar intensity1 = vdisp.at<uchar>(i, j);
			dep.at<float>((i,j)) = 1.0f / (intensity1.val[0]+.001) ;
			//-- As per the approximation --//
			dep.at<float>((i,j)) = dep.at<float>((i,j))* vdisp.cols * 1.2 * 100 / atoi(argv[1]);
		}
	}
	
	cout <<"Columns are: " << vdisp.cols << " and argument is: " << atoi(argv[1]) << endl;
	
	float maxval = 0;
	
	for (int i = 0; i < dep.rows; i++)
	{
		for (int j = 0; j < dep.cols; j++)
		{
			dep_val = (float)dep.at<float>((i,j));
			if(dep_val >= atof(argv[4]))
			{
				dep.at<float>((i,j)) = 0.0;
			}
			else
			{
				dep.at<float>((i,j)) = dep_val;
				if (dep_val > maxval)
				{
					maxval = dep_val;
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
			dep_val = (float) dep.at<float>((j,k));
			if (dep_val > .1)
			{
				depth = depth + dep_val;
				counter = counter+1;
			}
		}
	}

	cout << "Counter: " << counter << endl;

	float depth_n = depth/counter; 
	float depth_nn = maxval - depth_n;
	cout << "Depth is: " << depth_nn << " and Max value is: " << maxval << endl;
	
	ofstream myfile ("results.txt", fstream::app);
	if (myfile.is_open())
	{
		myfile << "For image " << argv[2] << " and " << argv[3] << " depth found is: " << depth_n << " and Max value is: " << maxval << " depth nn is " << depth_nn << endl;
		myfile.close();
	}
	else cout << "Unable to open file";

	
    return 0;
}
