#include "dominant_vector.h"

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "include.hpp"

extern rld::Gpu gpu;

#define BLOCK_SIZE 32
#define N BLOCK_SIZE *BLOCK_SIZE

__global__ void getDominantVector(uchar *imgXX, uchar *imgXY, uchar *imgYY, uchar **grad,
                                  uchar **domintVtr, uchar **eigenVals, int step, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) {
        return;
    }

    /* Get row index poiter */
    int rowIdx = row * step;
    float* domintVtrUPtr = (float*)(domintVtr[0] + rowIdx);
    float* domintVtrVPtr = (float*)(domintVtr[1] + rowIdx);
    float* eigenValsUPtr = (float*)(eigenVals[0] + rowIdx);
    float* eigenValsVPtr = (float*)(eigenVals[1] + rowIdx);
    float* gradUPtr = (float*)(grad[0] + rowIdx);
    float* gradVPtr = (float*)(grad[1] + rowIdx);

    /* Solve equation: X^2 - (imgXX + imgYY) * X + (imgXX * imgYY - imgXY^2) */
    float c = ((float *)(imgXX + rowIdx))[col] * ((float *)(imgYY + rowIdx))[col] - ((float *)(imgXY + rowIdx))[col] * ((float *)(imgXY + rowIdx))[col];
    float b = ((float *)(imgXX + rowIdx))[col] + ((float *)(imgYY + rowIdx))[col];
    float delta = sqrt(b * b - 4 * c);
    if (delta < 0) {
        delta = 0;
    }

    eigenValsUPtr[col] = (b + delta) / 2;
    eigenValsVPtr[col] = (b - delta) / 2;

    domintVtrUPtr[col] = eigenValsUPtr[col] - ((float *)(imgYY + rowIdx))[col];
    domintVtrVPtr[col] = ((float *)(imgXY + rowIdx))[col];

    // /* normalize dominant vector */
    float sample = sqrt(domintVtrUPtr[col] * domintVtrUPtr[col] + domintVtrVPtr[col] * domintVtrVPtr[col]);
    sample = (sample > 1e-9) ? sample : 1e-9;
    domintVtrUPtr[col] /= sample;
    domintVtrVPtr[col] /= sample;

    // /* get sign dominant vector */
    float sign = domintVtrUPtr[col] * gradUPtr[col] + domintVtrVPtr[col] * gradVPtr[col];
    if (fabs(sign) < 1e-9) {
        domintVtrUPtr[col] *= 0;
        domintVtrVPtr[col] *= 0;
    } else if (sign < -1e-9) {
        domintVtrUPtr[col] *= -1;
        domintVtrVPtr[col] *= -1;
    }
}

void rld::kernel::cudaDominantVector() {
    int rows = gpu.imgXX.rows;
    int cols = gpu.imgXX.cols;

    uchar *gradHostPtr[2];
    uchar *domintVtrHostPtr[2];
    uchar *eigenValsHostPtr[2];
    uchar **gradDevicePtr;
    uchar **domintVtrDevicePtr;
    uchar **eigenValsDevicePtr;

    /* Initialize host arrays */
    for (int i = 0; i < 2; i++) {
        gradHostPtr[i] = gpu.grad[i].data;
        domintVtrHostPtr[i] = gpu.domintVtr[i].data;
        eigenValsHostPtr[i] = gpu.eigenVals[i].data;
    }

    /* Allocate device memory */
    cudaMalloc(&gradDevicePtr, 2 * sizeof(uchar *));
    cudaMalloc(&domintVtrDevicePtr, 2 * sizeof(uchar *));
    cudaMalloc(&eigenValsDevicePtr, 2 * sizeof(uchar *));

    /* Transfer data from host to device memory */
    cudaMemcpy(gradDevicePtr, gradHostPtr, 2 * sizeof(uchar *), cudaMemcpyHostToDevice);
    cudaMemcpy(domintVtrDevicePtr, domintVtrHostPtr, 2 * sizeof(uchar *), cudaMemcpyHostToDevice);
    cudaMemcpy(eigenValsDevicePtr, eigenValsHostPtr, 2 * sizeof(uchar *), cudaMemcpyHostToDevice);

    /* Get number block and thread */
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cols / dimBlock.x + 1, rows / dimBlock.y + 1);

    getDominantVector<<<dimGrid, dimBlock>>>(gpu.imgXX.data, gpu.imgXY.data, gpu.imgYY.data, gradDevicePtr,
                                             domintVtrDevicePtr, eigenValsDevicePtr, gpu.imgXX.step, rows, cols);

    /* Deallocate device memory */
    cudaDeviceSynchronize();
    cudaFree(gradDevicePtr);
    cudaFree(domintVtrDevicePtr);
    cudaFree(eigenValsDevicePtr);
}