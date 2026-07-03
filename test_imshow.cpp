#include <opencv4/opencv2/opencv.hpp>
int main() {
    cv::Mat img = cv::Mat::zeros(200, 200, CV_8UC3);
    cv::imshow("Test", img);
    cv::waitKey(0);
    return 0;
}
