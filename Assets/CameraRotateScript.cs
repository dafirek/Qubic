using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Transform targetTransform; // Reference to the game board's transform
    public float distance = 10f; // Distance of the camera from the target
    public float xSpeed = 250f; // Speed for horizontal rotation
    public float ySpeed = 120f; // Speed for vertical rotation

    private float yaw = 0f; // Horizontal rotation angle
    private float pitch = 0f; // Vertical rotation angle

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        yaw = angles.y;
        pitch = angles.x;
    }

    void LateUpdate()
    {
        // Check if the target transform is assigned
        if (targetTransform == null)
            return;

        // Rotation input from the mouse
        yaw += xSpeed * Input.GetAxis("Mouse X") * Time.deltaTime;
        pitch -= ySpeed * Input.GetAxis("Mouse Y") * Time.deltaTime;

        // Clamp the vertical rotation angle
        pitch = ClampAngle(pitch, -90f, 90f);

        // Calculate the desired rotation quaternion
        Quaternion rotation = Quaternion.Euler(pitch, yaw, 0f);

        // Calculate the desired position of the camera
        Vector3 position = rotation * new Vector3(0f, 0f, -distance) + targetTransform.position;

        // Update the camera's rotation and position
        transform.rotation = rotation;
        transform.position = position;
    }

    private float ClampAngle(float angle, float min, float max)
    {
        if (angle < -360f)
            angle += 360f;
        if (angle > 360f)
            angle -= 360f;
        return Mathf.Clamp(angle, min, max);
    }
}
