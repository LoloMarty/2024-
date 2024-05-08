#inlcude <opencv2/opencv.hpp>
#include <sys/time.h>

#define FILTER_RADIUS 2

//for simplicity, we use the constant average filter only in this assignment
const float F_h[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
}

__constant__ float F_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

// check CUDA error if exists
#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}

// check if the difference of two cv::Mat images is small
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsinged int nCols) {
    const float relativeTolerance = 1e-2;
    for(int i=0; i<nRows; i++){
        for(int j=0; j<nCols; j++){
            float relativeError = ((float)answer1.at<unsigned char>(i, j) - (float)answer2.at<unsigned char>(i,j)) / 255;
            if(relativeError > relativeTolerance || relativeError < -relativeTolerance) {
                printf("TEST FAILED at (%d, %d) with relativeError: %f\n", i, j, relativeError);
                printf("    answer1.at<unsigned char>(%d, %d): %u\n", i, j, answer1.at<unsigned char>(i,j));
                printf("    answer2.at<unsigned char>(%d, %d): %u\n\n", i, j, answer2.at<unsigned char>(i,j));
                return false;
            }
        }
    }
    printf("TEST PASSED\n\n");
    return true;
}

//CPU timer
double myCPUTimer(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ( (double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

// A CPU-implementation of image blur using the average box filter
void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

}

// A CUDA kernel of image blur using the average box filter
__global__ void blurImage_Kernel (unsigned char * Pout, unsigned char * pin, unsigned int width, unsigned int height)
{

}

// A GPU-implementation of image blur using the average box filter
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

}

// An optimized CUDA kernel of image blur using the average box filter from constant memory 
__global__ void blurImage_tiled_Kernel (unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height)
{

}

// A GPU-implementation of image blur, where the kernel performs shared memory tiled convolution using the average box filter from constant memory 
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

}

int main(int argc, char** argv){
    cudaDeviceSynchronize();

    double startTime, endTime;

    // use openCV to load a grayscale image. 
    cv::Mat grayImg = cv::imread("santa-grayscale.jpg", cv::IMREAD_GRAYSCALE);
    if(grayImg.empty()) return -1;

    //obtain image's height, width, and number of channels
    unsigned int nRows = grayImg.rows, nCols = grayImg.cols, nChannels = grayImg.channels();

    // for comparison purpose, here uses OpenCV's blur function which uses average filter in convolution
    cv::Mat blurredImg_opencv(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    cv::blur( grayImg, blurredImg_opencv, cv::Size( 2*FILTER_RADIUS+1, 2*FILTER_RADIUS+1 ), cv::Point(-1, -1), cv::BORDER_CONSTANT);
    endTime = myCPUTimer();
    printf("openCV's blur (CPU):                            %f s\n\n", endTime - startTime); fflush(stdout);

    // for comparison purpose, implement a cpu version
    cv::Mat blurredImg_cpu(nRows, nCols, CV_8UC1, cv::Scalar(0));   // cv:Mat constructor to create and initialize an cv:Mat object; note that CV_8UC1 implies 8-bit unsigned, single channel
    startTime = myCPUTimer();
    blurImage_h(blurredImg_cpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on CPU:                            %f s\n\n", endTime - startTime); fflush(stdout);

    // implement a gpu version that calls a CUDA kernel
    cv::Mat blurredImg_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_d(blurredImg_gpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf("blurImage on GPU:                            %f s\n\n", endTime - startTime); fflush(stdout);

    // implement a gpu verions that calls a CUDA kernel which performs a shared-memory tiled comvolution, and filter elements are loaded from constant memory 
    cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_d(blurredImg_tiled_gpu, grayImg, nRows, nCols);
    endTime = myCPUTimer();
    printf(" (tiled)blurImage on GPU:                            %f s\n\n", endTime - startTime); fflush(stdout);

    // save the result blurred images to disk
    bool check = cv::imwrite("./blurredImg_opencv.jpg", blurredImg_opencv);
    if(check == false) { printf("error!\n"); return -1; }

    check = cv::imwrite("./blurredImg_cpu.jpg", blurredImg_cpu);
    if(check == false) { printf("error!\n"); return -1; }

    check = cv::imwrite("./blurredImg_gpu.jpg", blurredImg_gpu);
    if(check == false) { printf("error!\n"); return -1; }

    check = cv::imwrite("./blurredImg_tiled_gpu.jpg", blurredImg_tiled_gpu);
    if(check == false) { printf("error!\n"); return -1; }

    // check if the result blurred images are similar to that of OpenCV's 
    verify(blurredImg_opencv, blurredImg_cpu, nRows, nCols);
    verify(blurredImg_opencv, blurredImg_gpu, nRows, nCols);
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);.

    return 0;
}