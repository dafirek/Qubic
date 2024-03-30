using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuManager : MonoBehaviour
{
    public void StartGame()
    {
        SceneManager.LoadScene("GameScene", LoadSceneMode.Single);
        GameManager.Instance.isPlayer1AI = false;
    }

    public void StarAiGame()
    {
        SceneManager.LoadScene("GameScene");
        GameManager.Instance.isPlayer1AI = true;
    }

}
