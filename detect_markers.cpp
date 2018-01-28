
// compile:  g++ -ggdb `pkg-config --cflags --libs opencv3` detect_markers.cpp -o dm.o 
// run:  ./dm.o -d=0 -c=cal.yml

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <numeric>

using namespace std;
using namespace cv;

namespace {
const char* about = "Basic marker detection";
const char* keys  =
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }";
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}

int row(int id) {
    return id/10;
}

int col(int id) {
    return id - (row(id) * 10);
}

void transform(Vec3d &point, Mat rot_mat, double xyz[3]);
void get_mean(vector<Vec3d> &points, Vec3d &mean);
// void rotate(Vec3d xyz, Mat rot_mat_t, Vec3d &rvec);
void rotate(double x, double y, double z, Mat &rot_mat, Vec3d &rvec);

/**
 */
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }

    VideoCapture inputVideo;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(0);
        waitTime = 0;
    }

    double x = 0;
    double y = -0.096;
    double z = -0.126;
    double c = -0.184;

    // process video frames
    while(inputVideo.grab()) {
        Mat image, imageCopy;
        inputVideo.retrieve(image);

        vector< int > ids; // array of marks ids detected
        vector< vector< Point2f > > corners, rejected; // corners of each marker
        vector< Vec3d > rvecs, tvecs; // resolved 3D rotation/pose and postion

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);

        // convert 2d postion to 3d position and pose using camera lens information and known marker length
        if(estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

        // double total = 0;
        // double total_count = 0;
        // // estimate distance bettween markers:
        // for(int i=0; i<ids.size(); i++) {
        //     for(int j=0; j<i; j++) {
        //         Vec3d diff = tvecs[i] - tvecs[j];
        //         double distance_between_points = norm(diff);

        //         if (col(ids[i]) == col(ids[j])) { // 1.27
        //             total += distance_between_points / abs(row(ids[i]) - row(ids[j])); // devide by
        //             total_count++;
        //         } else if (row(ids[i]) == row(ids[j])) { // 1.33
        //             total += sqrt(distance_between_points * distance_between_points / 2) * 2;
        //             total_count++;
        //         }

        //     }
        // }

        // double side_length = total_count ? total/total_count : 0;
        //cout << "side_length: " << side_length << endl;
        double side_length = 0.127;

        // find center of the cube
        vector<Vec3d> cube_center_points;
        vector<Vec3d> glasses_center_points;
        Vec3d glasses_r(NAN, NAN, NAN);

        if (ids.size() > 0 && side_length != 0) {
            aruco::drawDetectedMarkers(image, corners, ids);

            for(unsigned int i = 0; i < ids.size(); i++) {
                cv::Mat rot_mat;
                Rodrigues(rvecs[i], rot_mat);
                Mat rot_mat_t = rot_mat.t(); // transpose of rot_mat for easy columns extraction
                Vec3d center_point = tvecs[i];

                // glasses
                if (row(ids[i]) == 0) {
                    // rotate and transform to center point
                    switch (col(ids[i])) {
                        // ear right
                        case 1: {
                            transform(center_point, rot_mat_t, (double []){0, 0, c-0.06});
                            rotate(0, CV_PI/2, CV_PI, rot_mat, rvecs[i]);
                            break;
                        }
                        // ear left
                        case 2:
                            transform(center_point, rot_mat_t, (double []){0, 0, c});
                            rotate(0, -CV_PI/2, CV_PI, rot_mat, rvecs[i]);
                            break;
                        // eye right
                        case 3:
                            transform(center_point, rot_mat_t, (double []){side_length/2, y, z});
                            rotate(0.35, 0, 0, rot_mat, rvecs[i]);
                            break;
                        // eye left
                        case 4:
                            transform(center_point, rot_mat_t, (double []){-side_length/2, y, z});
                            rotate(0.35, 0, 0, rot_mat, rvecs[i]);
                            break;
                    }
                    glasses_center_points.push_back(center_point);
                    glasses_r = rvecs[i];
                } 
                // cube
                else {
                    transform(center_point, rot_mat_t, (double []){0, side_length*(row(ids[i])-0.5), -side_length/2});
                    cube_center_points.push_back(center_point);
                }
                // draw results
                aruco::drawAxis(image, camMatrix, distCoeffs, rvecs[i], center_point, markerLength * 0.5f);
            }


            Vec3d cube_mean(0.0, 0.0, 0.0);
            get_mean(cube_center_points, cube_mean);
            // aruco::drawAxis(image, camMatrix, distCoeffs, Vec3d(0.0, 0.0, 0.0), cube_mean, markerLength * 2);

            Vec3d glasses_mean(0.0, 0.0, 0.0);
            get_mean(glasses_center_points, glasses_mean);
            // aruco::drawAxis(image, camMatrix, distCoeffs, Vec3d(0.0, 0.0, 0.0), glasses_mean, markerLength * 2);

            cout << "{ "
            "\"c\": " << cube_mean - glasses_mean << ","
            // "\"g\": [" << glasses_mean[0] << ", " << glasses_mean[1] << ", " << glasses_mean[2] << "],"
            "\"r\": " << glasses_r
            << "}" << endl;

        }

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(image, rejected, noArray(), Scalar(100, 0, 255));

        cv::flip (image, image, 1);
        imshow("out", image);

        int key = cvWaitKey(1);
        // switch(key) {
        //     case 'x':
        //         x += 0.002;
        //         break;
        //     case 's':
        //         x -= 0.002;
        //         break;
        //     case 'y':
        //         y += 0.002;
        //         break;
        //     case '6':
        //         y -= 0.002;
        //         break;
        //     case 'z':
        //         z += 0.002;
        //         break;
        //     case 'a':
        //         z -= 0.002;
        //         break;
        //     case 'c':
        //         c += 0.002;
        //         break;
        //     case 'd':
        //         c -= 0.002;
        //         break;
        // }
        // cout << " c: " << c << " y: " << y << " z: " << z << endl;
    }

    return 0;
}

void rotate(double x, double y, double z, Mat &rot_mat, Vec3d &rvec) {
    cv::Mat matX, matY, matZ;
    Rodrigues(Vec3d(x,0,0), matX);
    Rodrigues(Vec3d(0,y,0), matY);
    Rodrigues(Vec3d(0,0,z), matZ);
    rot_mat = rot_mat * matZ * matY * matX;
    Rodrigues(rot_mat, rvec);
}

void get_mean(vector<Vec3d> &points, Vec3d &mean) {
    Vec3d sum = std::accumulate(
        points.begin(), points.end(), // Run from begin to end
        Vec3d(0.0,0.0,0.0),       // Initialize with a zero point
        std::plus<cv::Vec3d>()      // Use addition for each point (default)
    );
    mean = Vec3d(sum[0]/points.size(), sum[1]/points.size(), sum[2]/points.size());
}

void transform(Vec3d &point, Mat rot_mat, double xyz[3]) {
    for(int i=0; i<3; i++) {
        double * r = rot_mat.ptr<double>(i); // x=0, y=1, z=2
        for(int j=0; j<3; j++) {
            point[j] +=  r[j]*xyz[i];
        }
    }
}



