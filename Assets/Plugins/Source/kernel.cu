
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__device__ int check_line(int* board, int x, int y, int z, int dx, int dy, int dz);

__global__ void evaluate_board_kernel(int* board, int *scores) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int score = 0;

    if (idx < 4 * 4 * 4) {
        int x = idx / (4 * 4);
        int y = (idx / 4) % 4;
        int z = idx % 4;

        //irányokba keres
        if(x == 0){ score += check_line(board, x, y, z, 1, 0, 0); }
        if (y == 0) { score += check_line(board, x, y, z, 0, 1, 0); }
        if (z == 0) { score += check_line(board, x, y, z, 0, 0, 1); }

        //2d átlokban keres
        score += check_line(board, x, y, z, 1, 1, 0);
        score += check_line(board, x, y, z, 1, -1, 0);
        score += check_line(board, x, y, z, 1, 0, 1);
        score += check_line(board, x, y, z, -1, 0, 1);
        score += check_line(board, x, y, z, 0, 1, 1);
        score += check_line(board, x, y, z, 0, 1, -1);

        // 3ds átlokban keres
        score += check_line(board, x, y, z, 1, 1, 1);
        score += check_line(board, x, y, z, -1, 1, 1);
        score += check_line(board, x, y, z, 1, -1, 1);
        score += check_line(board, x, y, z, 1, 1, -1);
        scores[idx] = score;
    }
}

__device__ int check_line(int* board, int x, int y, int z, int dx, int dy, int dz) {
    int ai = 0;
    int player = 0;
    int ix = 0;
    int iy = 0;
    int iz = 0;

    for (int i = 0; i < 4; i++) {
        ix = x + i * dx;
        iy = y + i * dy;
        iz = z + i * dz;
        int idx = ix * 16 + iy * 4 + iz;

        if (ix >= 0 && ix < 4 && iy >= 0 && iy < 4 && iz >= 0 && iz < 4) {
            if (board[idx] == 2) {
                ai++;
            }
            else if (board[idx] == 1) {
                player++;
            }
        }
        else { return 0; }//ha egy sort valamiért a közepén kezd meg akkor értéktelen az adat
    }

    if(player == 0){
        switch (ai)
        {
        case 4: return 10000;
        case 3: return 100;
        case 2: return 20;
        default:
            return 0;
        }
    }
    if(ai == 0){
        switch (player)
        {
        case 4: return -10000;
        case 3: return -100;
        case 2: return -20;
        default:
            return 0;
        }
    }
    return 0;
}