#include <cuda_runtime.h>
#include <device_launch_parameters.h>

 __global__ void evaluate_board_kernel(int* board, int* scores);

extern "C" void LaunchEvaluateBoardKernel(int* d_board, int* d_scores) {
    dim3 blockDim(256);
    dim3 gridDim((4 * 4 * 4 + blockDim.x - 1) / blockDim.x);

    evaluate_board_kernel <<<gridDim, blockDim >>> (d_board, d_scores);
}
