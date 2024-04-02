using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    private static GameManager _instance;
    private static CudaAI CudaAI;

    public Texture defaultTexture;
    public Texture highlightTexture;
    public Material playerMaterial;
    public Material aiMaterial;

    public bool playerTurn = true;
    public bool GameOver = false;
    public bool isPlayer1AI { get; set; }

    private GameObject lastHighlighted = null;

    public int[,,] gameState = new int[4, 4, 4];
    private int cache;

    public static GameManager Instance =>
        _instance ? _instance : new GameObject("Game Manager").AddComponent<GameManager>();


    private void Awake()
    {
        _instance = this;
        DontDestroyOnLoad(gameObject);
    }
    void Start()
    {
        CudaAI = gameObject.AddComponent<CudaAI>();
        cache = 0;
    }


    void Update()
    {
        if (!GameOver)
        {
            if (cache == 0)
            {
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                RaycastHit hit;
                if (Physics.Raycast(ray, out hit))
                {
                    GameObject hitObject = hit.transform.gameObject;
                    PlaneClick planeClick = hit.transform.GetComponent<PlaneClick>();
                    if (hitObject != lastHighlighted && planeClick != null)
                    {
                        if (lastHighlighted != null)
                        {
                            lastHighlighted.GetComponent<Renderer>().material.mainTexture = defaultTexture;
                        }
                        if (planeClick.y == 3)
                        {
                            hitObject.GetComponent<Renderer>().material.mainTexture = highlightTexture;
                            lastHighlighted = hitObject;
                        }
                    }



                    if (Input.GetMouseButtonDown(0))
                    {
                        cache = 5;
                        if (planeClick != null && planeClick.y == 3)
                        {
                            if (gameState[planeClick.x, planeClick.y, planeClick.z] == 0)
                            {
                                gameState = MakeMove(gameState, (planeClick.x + planeClick.z * 4), playerTurn ? 1 : 2, planeClick);
                                Debug.Log(planeClick.x + " , " + planeClick.y + " , " + planeClick.z);
                                if (IsGameOver(gameState))
                                {
                                    Debug.Log("Player " + (playerTurn ? 1 : 2) + " Win!");
                                    GameOver = true;
                                }
                                playerTurn = !playerTurn;
                                if (!GameOver)
                                {
                                    gameState = MakeMove(gameState, AiMove(gameState), playerTurn ? 1 : 2, planeClick);
                                }
                                playerTurn = !playerTurn;
                            }


                        }
                    }
                }
                else if (lastHighlighted != null)
                {
                    lastHighlighted.GetComponent<Renderer>().material.mainTexture = defaultTexture;
                    lastHighlighted = null;
                    cache = 5;
                }
            }
            else { cache--; }
        }
    }

    public bool IsGameOver(int[,,] board)
    {
        //sorok, oszlopo, mélység egyenes vonalban
        for (int x = 0; x < 4; x++)
        {
            for (int y = 0; y < 4; y++)
            {
                for (int z = 0; z < 4; z++)
                {
                    if (board[x, y, z] != 0 &&
                        board[x, y, z] == board[x, y, (z + 1) % 4] &&
                        board[x, y, z] == board[x, y, (z + 2) % 4] &&
                        board[x, y, z] == board[x, y, (z + 3) % 4])
                    {
                        return true;
                    }
                    if (board[x, y, z] != 0 &&
                    board[x, y, z] == board[(x + 1) % 4, y, z] &&
                    board[x, y, z] == board[(x + 2) % 4, y, z] &&
                    board[x, y, z] == board[(x + 3) % 4, y, z])
                    {
                        return true;
                    }
                    if (board[x, y, z] != 0 &&
                    board[x, y, z] == board[x, (y + 1) % 4, z] &&
                    board[x, y, z] == board[x, (y + 2) % 4, z] &&
                    board[x, y, z] == board[x, (y + 3) % 4, z])
                    {
                        return true;
                    }
                }
            }
        }
        //diagonal
        //egyszerü diagonális
        //a x,y,z tengelyekböl 2-t kiválasztva kapunk egy síkot, amit a kimaradt tengely mentén csúsztatunk. Ezen a síkon tudjuk tesztelni hogy van-e átló 
        for (int i = 0; i < 4; i++)
        {
            if ((board[i, 0, 0] != 0 && board[i, 1, 1] == board[i, 0, 0] && board[i, 2, 2] == board[i, 0, 0] && board[i, 3, 3] == board[i, 0, 0]) ||
                (board[i, 0, 3] != 0 && board[i, 1, 2] == board[i, 0, 3] && board[i, 2, 1] == board[i, 0, 3] && board[i, 3, 0] == board[i, 0, 3]))
            {
                return true;
            }
            if ((board[0, 0, i] != 0 && board[1, 1, i] == board[0, 0, i] && board[2, 2, i] == board[0, 0, i] && board[3, 3, i] == board[0, 0, i]) ||
                (board[0, 3, i] != 0 && board[1, 2, i] == board[0, 3, i] && board[2, 1, i] == board[0, 3, i] && board[3, 0, i] == board[0, 3, i]))
            {
                return true;
            }
            if ((board[0, i, 0] != 0 && board[1, i, 1] == board[0, i, 0] && board[2, i, 2] == board[0, i, 0] && board[3, i, 3] == board[0, i, 0]) ||
                (board[0, i, 3] != 0 && board[1, i, 2] == board[0, i, 3] && board[2, i, 1] == board[0, i, 3] && board[3, i, 0] == board[0, i, 3]))
            {
                return true;
            }
        }
        //van 4 átlója ami keresztül megy az egész kockán. ezek az átlók összekötik az alját képző négyzet sarkait a tetejét képző négyzetek sarkaival
        if ((board[0, 0, 0] != 0 && board[1, 1, 1] == board[0, 0, 0] && board[2, 2, 2] == board[0, 0, 0] && board[3, 3, 3] == board[0, 0, 0]) ||
            (board[3, 0, 0] != 0 && board[2, 1, 1] == board[3, 0, 0] && board[1, 2, 2] == board[3, 0, 0] && board[0, 3, 3] == board[3, 0, 0]) ||
            (board[0, 3, 0] != 0 && board[1, 2, 1] == board[0, 3, 0] && board[2, 1, 2] == board[0, 3, 0] && board[3, 0, 3] == board[0, 3, 0]) ||
            (board[0, 0, 3] != 0 && board[1, 1, 2] == board[0, 0, 3] && board[2, 2, 1] == board[0, 0, 3] && board[3, 3, 0] == board[0, 0, 3]))
        {
            return true;
        }


        return false;
    }

    public int[] PosibleMoves(int[,,] board)
    {
        List<int> posm = new List<int>();
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                if (board[i, 3, j] == 0)
                {
                    int index = i * 4 + j;
                    posm.Add(index);
                }
            }
        }
        
        return posm.ToArray();
    }

    public bool IsOver(int[,,] board)
    {
        if(PosibleMoves(board).Length != 0)
        {
            return false;
        }
        return true;
    }

    public int[,,] MakeMove(int[,,] board, int move, int player, PlaneClick pk) {
        int[,,] boardout = CopyArray(board);
        int movecpy = move;
        for (int i = 0; i < 4; i++)
        {
            if(boardout[move%4,i,move/4] == 0 && move >=0)
            {
                boardout[move % 4, i, move / 4] = player;
                pk.OnPlaneClicked(y: i, x: move%4, z: move / 4);
                break;
            }
        }
        return boardout;
    }

    public int[,,] MakeMove(int[,,] board, int move, int player)
    {
        int[,,] boardout = CopyArray(board);
        int movecpy = move;
        for (int i = 0; i < 4; i++)
        {
            if (boardout[move / 4, i, move % 4] == 0)
            {
                boardout[move / 4, i, move % 4] = player;
                break;
            }
        }
        return boardout;
    }


    public static int[,,] CopyArray(int[,,] source)
    {

        int[,,] copy = new int[4, 4, 4];

        for (int d = 0; d < 4; d++)
        {
            for (int h = 0; h < 4; h++)
            {
                for (int w = 0; w < 4; w++)
                {
                    copy[d, h, w] = source[d, h, w];
                }
            }
        }

        return copy;
    }

    //ez kell
    public int AiMove(int[,,] board)
    {
        int[] pmove = PosibleMoves(board);
        int move = 0;
        int max = int.MinValue;
        int score = int.MinValue;
        foreach(int pm in pmove)
        {
            score = AlphaBeta(MakeMove(board, pm, 1), 3, int.MinValue, int.MaxValue, false);
            Debug.Log("Score: " + score);
            if (max < score)
            {
                max = score;
                move = pm;
            }
        }
        
            //if(pmove[i] == 1) ez azt jelenti hogy csak a 0 move számít
            
        Debug.Log("AI chose: " + move + " move.");
        return move;
    }

    public int AlphaBeta(int[,,] board, int depth, int alpha, int beta, bool isMaximizingPlayer)
    {
        if(IsGameOver(board) || IsOver(board) || depth == 0)
        {
            if (playerTurn) 
            {
                return eval(board);
            }
            return CudaEval(board);
        }
        if (isMaximizingPlayer)
        {
            int value = int.MinValue;
            foreach (int m in PosibleMoves(board))
            {
                value = Math.Max(value, AlphaBeta(MakeMove(board, m, playerTurn ? 1 : 2), depth - 1, alpha, beta, false));
                alpha = Math.Max(alpha, value);
                if (value >= beta) { break; }
            }
            return value;
        }
        else
        {
            int value = int.MaxValue;
            foreach (int m in PosibleMoves(board))
            {
                value = Math.Min(value, AlphaBeta(MakeMove(board, m, playerTurn ? 2 : 1), depth - 1, alpha, beta, true));
                beta = Math.Min(beta, value);
                if (value <= alpha) { break; }
            }
            return value;
        }
    }

    public int eval(int[,,] board)
    {
        int score = 0;
        int mult = playerTurn ? -1 : 1;
        for (int idx = 0;idx<64;idx++)
        {
            int x = idx / (4 * 4);
            int y = (idx / 4) % 4;
            int z = idx % 4;

            //irányokba keres
            if (x == 0) { score += check_line(board, x, y, z, 1, 0, 0)*mult; }
            if (y == 0) { score += check_line(board, x, y, z, 0, 1, 0)*mult; }
            if (z == 0) { score += check_line(board, x, y, z, 0, 0, 1)*mult; }

            //2d átlokban keres
            score += check_line(board, x, y, z, 1, 1, 0)*mult;
            score += check_line(board, x, y, z, 1, -1, 0)*mult;
            score += check_line(board, x, y, z, 1, 0, 1)*mult;
            score += check_line(board, x, y, z, -1, 0, 1)*mult;
            score += check_line(board, x, y, z, 0, 1, 1)*mult;
            score += check_line(board, x, y, z, 0, 1, -1)*mult;

            // 3ds átlokban keres
            score += check_line(board, x, y, z, 1, 1, 1)*mult;
            score += check_line(board, x, y, z, -1, 1, 1)*mult;
            score += check_line(board, x, y, z, 1, -1, 1)*mult;
            score += check_line(board, x, y, z, 1, 1, -1)*mult;
        }
        return score;
    }

    public int check_line(int[,,] board, int x, int y, int z, int dx, int dy, int dz)
    {
        int ai = 0;
        int player = 0;
        int ix = 0;
        int iy = 0;
        int iz = 0;

        for (int i = 0; i < 4; i++)
        {
            ix = x + i * dx;
            iy = y + i * dy;
            iz = z + i * dz;

            if (ix >= 0 && ix < 4 && iy >= 0 && iy < 4 && iz >= 0 && iz < 4)
            {
                if (board[ix,iy,iz] == 2)
                {
                    ai++;
                }
                else if (board[ix,iy,iz] == 1)
                {
                    player++;
                }
            }
            else { return 0; }//ha egy sort valamiért a közepén kezd meg akkor értéktelen az adat
        }

        if (player == 0)
        {
            switch (ai)
            {
                case 4: return 10000;
                case 3: return 100;
                case 2: return 20;
                default:
                    return 0;
            }
        }
        if (ai == 0)
        {
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

    public int CudaEval(int[,,] board)
    {
        int score = 0;
        CudaAI.EvaluateBoard(boardIntoCuda(board), out int[] scores);
        foreach (int s in scores) { score += s; }
        return score;
    }

    public int[] boardIntoCuda(int[,,] board)
    {
        int[] cudaboard = new int[64];
        int[,,] cpy = CopyArray(board);

        for (int d = 0; d < 4; d++)
        {
            for (int h = 0; h < 4; h++)
            {
                for (int w = 0; w < 4; w++)
                {
                    cudaboard[d * 16 + h * 4 + w] = cpy[d, h, w];
                }
            }
        }
        return cudaboard;
    }

    public int[] MakeCudaMove(int[] board, int move)
    {
        int[] boardout = CopyArray1D(board);

        for (int i = 0; i < 4; i++)
        {
            if (boardout[(move - 1 + i * 16)] == 0 && move >= 0)
            {
                boardout[(move - 1 + i * 16)] = 2;
                break;
            }
        }
        return boardout;
    }

    public static int[] CopyArray1D(int[] source)
    {

        int[] copy = new int[64];

        for (int d = 0; d < 64; d++)
        {
            copy[d] = source[d];
        }

        return copy;
    }
}
