using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GridGenerator : MonoBehaviour
{
    public GameObject cellPrefab; // Reference to the prefab for a single cell
    public GameObject spherePrefab;
    public int gridSize = 4; // Size of the grid (4x4x4)
    public float cellSpacing = 1.0f; // Distance between cells

    void Start()
    {
        GenerateGrid();
    }

    void GenerateGrid()
    {
        for (int z = 0; z < gridSize; z++)
        {
            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    Vector3 position = new Vector3(x * cellSpacing, y * cellSpacing, z * cellSpacing);
                    GameObject cell = Instantiate(cellPrefab, position, Quaternion.identity);
                    cell.GetComponent<MeshRenderer>().enabled = true;
                    cell.transform.parent = transform;
                    PlaneClick planeClick = cell.AddComponent<PlaneClick>();
                    planeClick.x = x;
                    planeClick.y = y;
                    planeClick.z = z;
                    planeClick.spherePrefab = spherePrefab;
                    if(cell.GetComponent<Collider>() == null)
                    {
                        cell.AddComponent<MeshCollider>();
                    }
                }
                /*Vector3 position2 = new Vector3((gridSize-1)*1.25f, y * cellSpacing, (z + 0.5f) * cellSpacing);
                GameObject separator = Instantiate(separatorPrefab, position2, Quaternion.identity);
                separator.GetComponent<MeshRenderer>().enabled = true;
                separator.transform.parent = transform;
                position2.Set(z * cellSpacing, y * cellSpacing, (gridSize - 1) * 1.25f);
                GameObject separator2 = Instantiate(separatorPrefab, position2, Quaternion.identity);
                separator2.transform.Rotate(0, 90, 0);
                separator2.GetComponent<MeshRenderer>().enabled = true;
                separator2.transform.parent = transform;*/

            }
        }
    }
}

public class PlaneClick : MonoBehaviour
{
    public int x;
    public int y;
    public int z;
    public GameObject spherePrefab;


    public void OnPlaneClicked(int y)
    {
        Vector3 position = new Vector3((float)(x * 2.5), (float)(y * 2.5), (float)(z * 2.5));
        GameObject cell = Instantiate(spherePrefab, position, Quaternion.identity);
        cell.GetComponent<MeshRenderer>().enabled = true;
        if (GameManager.Instance.playerTurn) { cell.GetComponent<MeshRenderer>().material = GameManager.Instance.playerMaterial; } else { cell.GetComponent<MeshRenderer>().material = GameManager.Instance.aiMaterial; }
        cell.transform.parent = transform.parent;
        Debug.Log("Plane clicked at row " + x + " and column " + z + "!");
    }
}
