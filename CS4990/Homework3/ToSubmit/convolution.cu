#include <opencv2/opencv.hpp>
#include <sys/time.h>

#define FILTER_RADIUS 2

//for simplicity, we use the constant average filter only in this assignment
fquitloat F_h[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};

__constant__ float F_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}, 
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}
};

//__constant__ float F_d[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

//cudaMemcpyToSymbol(F_d, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));

// check CUDA error if exists
#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
        exit(-1); \
    } \
}

const float relativeTolerance = 1e-2;

// check if the difference of two cv::Mat images is small
bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols) { 
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
__host__ void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{
    for(int i = 0; i < nRows; ++i) {
        for(int j = 0; j < nCols; ++j) {
            float sum = 0.0f;
            for(int fi = -FILTER_RADIUS; fi <= FILTER_RADIUS; ++fi) {
                for(int fj = -FILTER_RADIUS; fj <= FILTER_RADIUS; ++fj) {
                    int ii = i + fi;
                    int jj = j + fj;
                    if(ii >= 0 && ii < nRows && jj >= 0 && jj < nCols) {
                        sum += F_h[fi + FILTER_RADIUS][fj + FILTER_RADIUS] * static_cast<float>(Pin_Mat_h.at<unsigned char>(ii, jj));
                    }
                }
            }
            Pout_Mat_h.at<unsigned char>(i, j) = static_cast<unsigned char>(sum);
        }
    }
}

// A CUDA kernel of image blur using the average box filter
__global__ void blurImage_Kernel (unsigned char * Pout, unsigned char * pin, unsigned int width, unsigned int height)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < height && j < width) {
        float sum = 0.0f;
        for(int fi = -FILTER_RADIUS; fi <= FILTER_RADIUS; ++fi) {
            for(int fj = -FILTER_RADIUS; fj <= FILTER_RADIUS; ++fj) {
                int ii = i + fi;
                int jj = j + fj;
                if(ii >= 0 && ii < height && jj >= 0 && jj < width) {
                    sum += F_d[fi + FILTER_RADIUS][fj + FILTER_RADIUS] * pin[ii * width + jj];
                }
            }
        }
        Pout[i * width + j] = static_cast<unsigned char>(sum);
    }
}

// A GPU-implementation of image blur using the average box filter
void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{
    unsigned char *d_Pout, *d_Pin;
    size_t size = nRows * nCols * sizeof(unsigned char);

    CHECK(cudaMalloc(&d_Pout, size));
    CHECK(cudaMalloc(&d_Pin, size));

    CHECK(cudaMemcpy(d_Pin, Pin_Mat_h.data, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (nRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //cudaMemcpyToSymbol(F_d, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    blurImage_Kernel<<<numBlocks, threadsPerBlock>>>(d_Pout, d_Pin, nCols, nRows);

    CHECK(cudaMemcpy(Pout_Mat_h.data, d_Pout, size, cudaMemcpyDeviceToHost));

    cudaFree(d_Pout);
    cudaFree(d_Pin);
}

// An optimized CUDA kernel of image blur using the average box filter from constant memory 
__global__ void blurImage_tiled_Kernel (unsigned char * Pout, unsigned char * Pin, unsigned int width, unsigned int height)
{
    __shared__ float shared[FILTER_RADIUS * 2 + 1][FILTER_RADIUS * 2 + 1];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y * blockDim.y + ty;
    int col_o = blockIdx.x * blockDim.x + tx;
    if(row_o < height && col_o < width) {
        float sum = 0.0f;
        for(int i = 0; i < FILTER_RADIUS * 2 + 1; ++i) {
            for(int j = 0; j < FILTER_RADIUS * 2 + 1; ++j) {
                int row_i = row_o + i - FILTER_RADIUS;
                int col_i = col_o + j - FILTER_RADIUS;
                if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
                    shared[i][j] = Pin[row_i * width + col_i];
                } else {
                    shared[i][j] = 0.0f;
                }
            }
        }
        for(int i = 0; i < FILTER_RADIUS * 2 + 1; ++i) {
            for(int j = 0; j < FILTER_RADIUS * 2 + 1; ++j) {
                sum += shared[i][j] * F_d[i][j];
            }
        }
        Pout[row_o * width + col_o] = static_cast<unsigned char>(sum);
    }

}

// A GPU-implementation of image blur, where the kernel performs shared memory tiled convolution using the average box filter from constant memory 
void blurImage_tiled_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{
    unsigned char *d_Pout, *d_Pin;
    size_t size = nRows * nCols * sizeof(unsigned char);

    CHECK(cudaMalloc(&d_Pout, size));
    CHECK(cudaMalloc(&d_Pin, size));

    CHECK(cudaMemcpy(d_Pin, Pin_Mat_h.data, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nCols + threadsPerBlock.x - 1) / threadsPerBlock.x, (nRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    blurImage_tiled_Kernel<<<numBlocks, threadsPerBlock>>>(d_Pout, d_Pin, nCols, nRows);

    CHECK(cudaMemcpy(Pout_Mat_h.data, d_Pout, size, cudaMemcpyDeviceToHost));

    cudaFree(d_Pout);
    cudaFree(d_Pin);
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
    blurImage_d(blurredImg_gpu, grayImg, nCols, nRows); // Change the order of arguments
    endTime = myCPUTimer();
    printf("blurImage on GPU:                            %f s\n\n", endTime - startTime); fflush(stdout);

    // implement a gpu verions that calls a CUDA kernel which performs a shared-memory tiled comvolution, and filter elements are loaded from constant memory 
    cv::Mat blurredImg_tiled_gpu(nRows, nCols, CV_8UC1, cv::Scalar(0));
    startTime = myCPUTimer();
    blurImage_tiled_d(blurredImg_tiled_gpu, grayImg, nCols, nRows); // Change the order of arguments
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
    verify(blurredImg_opencv, blurredImg_tiled_gpu, nRows, nCols);

    return 0;
}