using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LineCode : MonoBehaviour
{
    LineRenderer lineRenderer;

    // Make it public so we can attatch it to the points
    public Transform origin;
    public Transform destination;

    // Start is called before the first frame update
    void Start()
    {
        // Get the line rendered componen insidse the line and give it a width
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.startWidth = 0.08f;
        lineRenderer.endWidth = 0.08f;
    }

    // Update is called once per frame
    void Update()
    {
        // Constantly update the position of the line
        lineRenderer.SetPosition(0, origin.position); // Has to come from the point
        lineRenderer.SetPosition(1, destination.position);
    }
}
