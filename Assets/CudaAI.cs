using UnityEngine;
using System.Runtime.InteropServices;

public class CudaAI : MonoBehaviour
{
    // Import the native function from the DLL/shared library

    [DllImport("CudaAI")]

    private static extern void EvaluateBoard(int[] board, int[] scores);

    // Expose a C# method that calls the native CUDA function
    public void EvaluateBoard(int[] board, out int[] scores)
    {
        scores = new int[64];
        EvaluateBoard(board, scores);
    }
}
