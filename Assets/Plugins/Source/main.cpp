#define DLL_EXPORTS
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


extern "C" void LaunchEvaluateBoardKernel(int* d_board, int* d_scores);

extern "C" __declspec(dllexport) void EvaluateBoard(int* board, int* scores) {
    int* d_board;
    int* d_scores;
    cudaMalloc(&d_board, sizeof(int) * 4 * 4 * 4);
    cudaMalloc(&d_scores, sizeof(int) * 4 * 4 * 4);

    cudaMemcpy(d_board, board, sizeof(int) * 4 * 4 * 4, cudaMemcpyHostToDevice);

    LaunchEvaluateBoardKernel(d_board, d_scores);

    cudaMemcpy(scores, d_scores, sizeof(int) * 4 * 4 * 4, cudaMemcpyDeviceToHost);

    cudaFree(d_board);
    cudaFree(d_scores);
}

