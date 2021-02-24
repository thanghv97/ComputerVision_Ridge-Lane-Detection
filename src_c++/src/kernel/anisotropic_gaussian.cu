#include "anisotropic_gaussian.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.hpp"

extern rld::Gpu gpu;

#define BLOCK_SIZE 32
__global__ void applyGaussianKernelX(uchar *input, uchar *output,
                                     uchar **kernelX_, uchar *kernelX_size, int rows, int cols, size_t step) {
    /* Get index row, col of thread process */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int curr_kernelX_size_pad = 0;
    float value = 0;
    float sum = 0;
    if (row >= rows || col >= cols)
        return;
    int i = row;
    int j = col;
    float *kernel_ptr;

    curr_kernelX_size_pad = (kernelX_size[i] - 1) / 2;
    float *row_input_ptr = (float *)(input + i * step);
    float *row_output_ptr = (float *)(output + i * step);

    kernel_ptr = (float *)kernelX_[i];

    value = 0;
    sum = 0;
    for (int k = 0; k < kernelX_size[i]; k++)
        sum = sum + kernel_ptr[k];
    for (int k = j - curr_kernelX_size_pad; k <= j + curr_kernelX_size_pad; k++) {
        if (k < 0 || k >= cols) {
            value += row_input_ptr[2 * j - k] * kernel_ptr[k - j + curr_kernelX_size_pad];
        } else if (k >= 0 && k < cols) {
            value += row_input_ptr[k] * kernel_ptr[k - j + curr_kernelX_size_pad];
        }
    }
    row_output_ptr[j] = value;
}

__global__ void applyGaussianKernelY(uchar *input, uchar *output,
                                     uchar **kernelY_, uchar *kernelY_size, int rows, int cols, size_t step) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_kernelY_size_pad;
    float value = 0;
    float *row_input_ptr = 0;
    float *kernel_ptr;
    if (row >= rows || col >= cols)
        return;
    int i = row;
    int j = col;

    curr_kernelY_size_pad = (kernelY_size[i] - 1) / 2;
    kernel_ptr = (float *)kernelY_[i];
    float *row_output_ptr = (float *)(output + i * step);

    value = 0;
    for (int k = i - curr_kernelY_size_pad; k <= i + curr_kernelY_size_pad; k++) {
        if (k < 0 || k >= rows) {
            row_input_ptr = (float *)(input + (2 * i - k) * step);
        } else if (k >= 0 && k < rows) {
            row_input_ptr = (float *)(input + k * step);
        }
        value += row_input_ptr[j] * kernel_ptr[k - i + curr_kernelY_size_pad];
    }
    row_output_ptr[j] = value;
}

void rld::kernel::cudaAnisotropicGaussian(cv::cuda::GpuMat srcImg, cv::cuda::GpuMat &dstImg,
                                          std::vector<cv::cuda::GpuMat> kernelX, std::vector<cv::cuda::GpuMat> kernelY,
                                          cv::cuda::GpuMat kernelXSize, cv::cuda::GpuMat kernelYSize) {
    int rows = srcImg.rows;
    int cols = srcImg.cols;
    uchar *kernelPtrX[rows];
    uchar *kernelPtrY[rows];
    uchar **kernelCudaX;
    uchar **kernelCudaY;

    /* Initialize host arrays */
    for (int i = 0; i < rows; i++) {
        kernelPtrX[i] = kernelX[i].data;
        kernelPtrY[i] = kernelY[i].data;
    }

    /* Allocate device memory */
    cudaMalloc(&kernelCudaX, rows * sizeof(uchar *));
    cudaMalloc(&kernelCudaY, rows * sizeof(uchar *));

    /* Transfer data from host to device memory */
    cudaMemcpy(kernelCudaX, kernelPtrX, rows * sizeof(uchar *), cudaMemcpyHostToDevice);
    cudaMemcpy(kernelCudaY, kernelPtrY, rows * sizeof(uchar *), cudaMemcpyHostToDevice);

    /* Get number block and thread */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cols / dimBlock.x + 1, rows / dimBlock.y + 1);

    /* Apply kernel */
    applyGaussianKernelY<<<dimGrid, dimBlock>>>(srcImg.data, gpu.tmpImg[0].data, kernelCudaY, kernelYSize.data, rows, cols, srcImg.step);
    applyGaussianKernelX<<<dimGrid, dimBlock>>>(gpu.tmpImg[0].data, srcImg.data, kernelCudaX, kernelXSize.data, rows, cols, gpu.tmpImg[0].step);

    cudaDeviceSynchronize();
    cudaFree(kernelCudaX);
    cudaFree(kernelCudaY);

    srcImg.copyTo(dstImg);
}