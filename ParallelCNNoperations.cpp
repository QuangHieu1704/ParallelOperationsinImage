#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <winapifamily.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void RandomMatrix(double*& imageArray, int& width, int& height, int& depth);
void LoadImg(string path, int &width, int &height, int &depth, double*& imageArray);
void WriteImg(int width, int height, int depth, double* imageArray);
void MaxPooling_Serial(double* imageArray, int width, int height, int depth, int kernelSize, int stride);
void MaxPooling_Parallel(double* imageArray, int width, int height, int depth, int kernelSize, int stride);
void AvgPooling_Serial(double* imageArray, int width, int height, int depth, int kernelSize, int stride);
void AvgPooling_Parallel(double* imageArray, int width, int height, int depth, int kernelSize, int stride);
void Conv_Serial(double* imageArray, int width, int height, int depth, double(&Kernel)[3][3], int stride, int padding);
void Conv_Parallel(double* imageArray, int width, int height, int depth, double(&Kernel)[3][3], int stride, int padding);
double LiToDouble(LARGE_INTEGER x);
double GetTime();

string type_to_str(int type);

void main()
{
    int width;
    int height;
    int depth;
    int padding, stride, kernelSize;
    double* imageArray;

    double Kernel_Identity[3][3] = { {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0} };
    double Kernel_Blur[3][3] = { {0.0625, 0.125, 0.0625}, {0.125, 0.25, 0.125}, {0.0625, 0.125, 0.0625} };
    double Kernel_MoreBlur[3][3] = { {0.011, 0.011, 0.011}, {0.011, 0.011, 0.011}, {0.011, 0.011, 0.011} };
    double Kernel_Emboss[3][3] = { {-2.0, -1.0, 0.0}, {-1.0, 1.0, 1.0}, {0.0, 1.0, 2.0} };
    double Kernel_Sharpen[3][3] = { {0.0, -1.0, 0.0}, {-1.0, 5.0, -1.0}, {0.0, -1.0, 0.0} };
    double Kernel_leftSobel[3][3] = { {1.0, 0.0, -1.0}, {2.0, 0.0, -2.0}, {1.0, 0.0, -1.0} };
    double Kernel_rightSobel[3][3] = { {-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0} };
    double Kernel_topSobel[3][3] = { {1.0, 2.0, 1.0}, {0.0, 0.0, 0.0}, {-1.0, -2.0, -3.0} };
    double Kernel_outline[3][3] = { {-1.0, -1.0, -1.0}, {-1.0, 8.0, -1.0}, {-1.0, -1.0, -1.0} };

    kernelSize = 3;
    stride = 1;
    padding = 1;

    //string input_path = "D:/Sources/MultiThread/ParallelCNNoperations/Image/img6.png";
    //LoadImg(input_path, width, height, depth, imageArray);
    //printf("\n\n\n\n\n");
    //printf("Loading Pixel value to imageArray sucessfully! \n");

    //WriteImg(width, height, depth, imageArray);
    //MaxPooling_Serial(imageArray, width, height, depth, kernelSize, stride);
    //AvgPooling_Serial(imageArray, width, height, depth, kernelSize, stride);
    //Conv_Serial(imageArray, width, height, depth, Kernel_topSobel, stride, padding);

    RandomMatrix(imageArray, width, height, depth);

    // serial
    double start_time = GetTime();
    printf("\nStarted Serial at %.20f \n", start_time);
    Conv_Serial(imageArray, width, height, depth, Kernel_outline, stride, padding);
    // AvgPooling_Serial(imageArray, width, height, depth, kernelSize, stride);
    double finish_time = GetTime();
    printf("Finished Serial at %.20f \n", finish_time);
    double time = finish_time - start_time;
    printf("Time of Serial:  %.20f\n\n\n", time);

    // parallel 
    start_time = GetTime();
    printf("Started Parallel at %.20f \n", start_time);
    Conv_Parallel(imageArray, width, height, depth, Kernel_outline, stride, padding);
    // AvgPooling_Parallel(imageArray, width, height, depth, kernelSize, stride);
    finish_time = GetTime();
    printf("Finished Parallel at %.20f \n", finish_time);
    time = finish_time - start_time;
    printf("Time of Parallel:  %.20f", time);

}

void RandomMatrix(double*& imageArray, int& width, int& height, int& depth)
{
    printf("Width = ");
    scanf_s("%d", &width);
    printf("Height = ");
    scanf_s("%d", &height);
    printf("Depth = ");
    scanf_s("%d", &depth);
    imageArray = new double[height * width * depth];
    for (int channel = 0; channel < depth; channel++)
    {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                imageArray[(channel * height + row) * width + col] = rand() / double(256);
            }
        }
    }
}

string type_to_str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

void LoadImg(string path, int& width, int& height, int& depth, double*& imageArray)
{
    Mat img = imread(path, IMREAD_COLOR);
    imshow("img", img);
    waitKey(0);
    width = img.cols;
    height = img.rows;
    depth = img.channels();
    string ty = type_to_str(img.type());
    printf("Matrix: %s %dx%d \n", ty.c_str(), img.cols, img.rows);
    printf("Width of input image is %d \n", width);
    printf("Height of input image is %d \n", height);
    printf("Depth of input image is %d \n", depth);
    imageArray = new double[height * width * depth];
    for (int channel = 0; channel < depth; channel++)
    {
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                imageArray[(channel * height + row) * width + col] = (double)img.at<Vec3b>(row, col)[channel];
            }
        }
    }
}

void WriteImg(int width, int height, int depth, double* imageArray)
{
    if(depth == 3)
    {
        Mat img(height, width, CV_8UC3);
        for (int k = 0; k < depth; k++)
        {
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    img.at<Vec3b>(i, j)[k] = (unsigned char)(imageArray[(k * height + i) * width + j]);
                }
            }
        }
        imshow("output imgage", img);
        waitKey(0);
    }
    if (depth == 1)
    {
        Mat img(height, width, CV_8UC1);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                img.at<uchar>(i, j) = (unsigned char)(imageArray[i * width + j]);
            }
        }
        imshow("output imgage", img);
        waitKey(0);
    }
}

void MaxPooling_Serial(double* imageArray, int width, int height, int depth, int kernelSize, int stride)
{
    int width_out = ceil((width - kernelSize) / stride) + 1;
    int height_out = ceil((height - kernelSize) / stride) + 1;
    int depth_out = depth;
    printf("Max Pooling Operator: \n");
    printf("width output = %d\n", width_out);
    printf("height output = %d\n", height_out);
    double* outputArray = new double[width_out * height_out * depth];
    for (int channel = 0; channel < depth; channel++)
    {
        for (int row = 0; row < height - kernelSize; row += stride)
        {
            for (int col = 0; col < width - kernelSize; col += stride)
            {
                double max = imageArray[(channel * height + row) * width + col];
                for (int sRow = row; sRow < row + kernelSize; sRow++)
                {
                    for (int sCol = col; sCol < col + kernelSize; sCol++)
                    {
                        if (max < imageArray[(channel * height + sRow) * width + sCol])
                        {
                            max = imageArray[(channel * height + sRow) * width + sCol];
                        }
                    }
                }
                outputArray[(channel * height_out + row / stride) * width_out + col / stride] = max;
            }
        }
    }
    // WriteImg(width_out, height_out, depth_out, outputArray);
}

void MaxPooling_Parallel(double* imageArray, int width, int height, int depth, int kernelSize, int stride)
{
    int width_out = ceil((width - kernelSize) / stride) + 1;
    int height_out = ceil((height - kernelSize) / stride) + 1;
    int depth_out = depth;
    printf("Max Pooling Operator: \n");
    printf("width output = %d\n", width_out);
    printf("height output = %d\n", height_out);
    double* outputArrayMax = new double[width_out * height_out * depth];
    for (int channel = 0; channel < depth; channel++)
    {
        #pragma omp parallel for schedule(static) 
        for (int row = 0; row < height - kernelSize; row += stride)
        {
            // printf("thread %d executing row = %d \n", omp_get_thread_num(), row);
            for (int col = 0; col < width - kernelSize; col += stride)
            {
                double max = imageArray[(channel * height + row) * width + col];
                for (int sRow = row; sRow < row + kernelSize; sRow++)
                {
                    for (int sCol = col; sCol < col + kernelSize; sCol++)
                    {
                        if (max < imageArray[(channel * height + sRow) * width + sCol])
                        {
                            max = imageArray[(channel * height + sRow) * width + sCol];
                        }
                    }
                }
                outputArrayMax[(channel * height_out + row / stride) * width_out + col / stride] = max;
            }
        }
    }
    // WriteImg(width_out, height_out, depth_out, outputArrayMax);
}

void AvgPooling_Serial(double* imageArray, int width, int height, int depth, int kernelSize, int stride)
{
    int width_out = ceil((width - kernelSize) / stride) + 1;
    int height_out = ceil((height - kernelSize) / stride) + 1;
    int depth_out = depth;
    printf("Average Pooling Operator: \n");
    printf("width output = %d\n", width_out);
    printf("height output = %d\n", height_out);
    double* outputArray = new double[width_out * height_out * depth_out];
    for (int channel = 0; channel < depth; channel++)
    {
        for (int row = 0; row < height - kernelSize; row += stride)
        {
            for (int col = 0; col < width - kernelSize; col += stride)
            {
                double avg = 0;
                for (int sRow = row; sRow < row + kernelSize; sRow++)
                {
                    for (int sCol = col; sCol < col + kernelSize; sCol++)
                    {
                        avg += imageArray[(channel * height + sRow) * width + sCol];
                    }
                }
                outputArray[(channel * height_out + row / stride) * width_out + col / stride] = avg / (kernelSize * kernelSize);
            }
        }
    }
    // WriteImg(width_out, height_out, depth_out, outputArray);
}

void AvgPooling_Parallel(double* imageArray, int width, int height, int depth, int kernelSize, int stride)
{ 
    int width_out = ceil((width - kernelSize) / stride) + 1;
    int height_out = ceil((height - kernelSize) / stride) + 1;
    int depth_out = depth;
    printf("Average Pooling Operator: \n");
    printf("width output = %d\n", width_out);
    printf("height output = %d\n", height_out);
    double* outputArrayAvg = new double[width_out * height_out * depth_out];
    for (int channel = 0; channel < depth; channel++)
    {
        #pragma omp parallel for schedule(static) 
        for (int row = 0; row < height - kernelSize; row += stride)
        {
            // printf("thread %d executing row = %d \n", omp_get_thread_num(), row);
            for (int col = 0; col < width - kernelSize; col += stride)
            {
                double avg = 0;
                for (int sRow = row; sRow < row + kernelSize; sRow++)
                {
                    for (int sCol = col; sCol < col + kernelSize; sCol++)
                    {
                        avg += imageArray[(channel * height + sRow) * width + sCol];
                    }
                }
                outputArrayAvg[(channel * height_out + row / stride) * width_out + col / stride] = avg / (kernelSize * kernelSize);
            }
        }
    }
}

void Conv_Serial(double* imageArray, int width, int height, int depth, double(&Kernel)[3][3], int stride, int padding)
{
    int kernelSize = sizeof(Kernel) / sizeof(Kernel[0]);
    int width_out = (width + 2 * padding - kernelSize) / stride + 1;
    int height_out = (height + 2 * padding - kernelSize) / stride + 1;
    int depth_out = depth;
    printf("Convolutional Operator: \n");
    printf("Kernel size = %d \n", kernelSize);
    printf("width output = %d \n", width_out);
    printf("height output = %d \n", height_out);
    double* outputArray = new double[width_out * height_out * depth_out];
    for (int channel = 0; channel < depth; channel++)
    {
        for (int row = 0; row < height_out; row++)
        {
            for (int col = 0; col < width_out; col++)
            {
                outputArray[(channel * height_out + row) * width_out + col] = 0;
            }
        }
    }
    int Center = (kernelSize - 1)/ 2;
    for (int channel = 0; channel < depth; channel++)
    {
        for (int row = 0; row < height; row = row + stride)
        {
            for (int col = 0; col < width; col = col + stride)
            {
                for (int kRow = 0; kRow < kernelSize; kRow++)
                {
                    for (int kCol = 0; kCol < kernelSize; kCol++)
                    {
                        int sRow = row + (kRow - Center);
                        int sCol = col + (kCol - Center);
                        if (sRow >= 0 && sRow < height && sCol >= 0 && sCol < width)
                        {
                            outputArray[(channel * height_out + row) * width_out + col] += imageArray[(channel * height_out + sRow) * width_out + sCol] * Kernel[kRow][kCol];
                        }
                    }
                }
                outputArray[(channel * height_out + row) * width_out + col] = abs(outputArray[(channel * height_out + row) * width_out + col]);
            }
        }
    }
    WriteImg(width_out, height_out, depth_out, outputArray);
}

void Conv_Parallel(double* imageArray, int width, int height, int depth, double(&Kernel)[3][3], int stride, int padding)
{
    int kernelSize = sizeof(Kernel) / sizeof(Kernel[0]);
    int width_out = (width + 2 * padding - kernelSize) / stride + 1;
    int height_out = (height + 2 * padding - kernelSize) / stride + 1;
    int depth_out = depth;
    printf("Convolutional Operator: \n");
    printf("Kernel size = %d \n", kernelSize);
    printf("width output = %d \n", width_out);
    printf("height output = %d \n", height_out);
    double* outputArray = new double[width_out * height_out * depth_out];
    for (int channel = 0; channel < depth; channel++)
    {
        for (int row = 0; row < height_out; row++)
        {
            for (int col = 0; col < width_out; col++)
            {
                outputArray[(channel * height_out + row) * width_out + col] = 0;
            }
        }
    }
    int Center = (kernelSize - 1) / 2;
    for (int channel = 0; channel < depth; channel++)
    {   
        #pragma omp parallel for schedule(static) 
        for (int row = 0; row < height; row = row + stride)
        {
            // printf("thread %d executing row = %d \n", omp_get_thread_num(), row);
            for (int col = 0; col < width; col = col + stride)
            {
                for (int kRow = 0; kRow < kernelSize; kRow++)
                {
                    for (int kCol = 0; kCol < kernelSize; kCol++)
                    {
                        int sRow = row + (kRow - Center);
                        int sCol = col + (kCol - Center);
                        if (sRow >= 0 && sRow < height && sCol >= 0 && sCol < width)
                        {
                            outputArray[(channel * height_out + row) * width_out + col] += imageArray[(channel * height_out + sRow) * width_out + sCol] * Kernel[kRow][kCol];
                        }
                    }
                }
                outputArray[(channel * height_out + row) * width_out + col] = abs(outputArray[(channel * height_out + row) * width_out + col]);
                outputArray[(channel * height_out + row) * width_out + col] = fmod(outputArray[(channel * height_out + row) * width_out + col], 255);
            }
        }
    }
}

double LiToDouble(LARGE_INTEGER x)
{
    double result =
        ((double)x.HighPart) * 4.294967296E9 + (double)((x).LowPart);
    return result;
}
// Function that gets the timestamp in seconds
double GetTime()
{
    LARGE_INTEGER lpFrequency, lpPerfomanceCount;
    QueryPerformanceFrequency(&lpFrequency);QueryPerformanceCounter(&lpPerfomanceCount);
    return LiToDouble(lpPerfomanceCount) / LiToDouble(lpFrequency);
}
