using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;

public class AnimationCode : MonoBehaviour
{
    public UDPReceive udpReceive;
    public GameObject[] Body;

    List<string> lines;
    int counter = 0;

    // Start is called before the first frame update
    void Start()
    {
        // lines = System.IO.File.ReadLines("Assets/PoseData.txt").ToList();
    }

    // Update is called once per frame
    void Update()
    {
        string data = udpReceive.data;

        public bool printDebug = false;

        // Remove the leading and trailing double square brackets
        data = data.TrimStart('[').TrimEnd(']');

        // Remove all additional square brackets and split by spaces
        data = data.Replace("[", "").Replace("]", "");

        // Print the cleaned-up data for debugging
        // print(data);

        // Split the data into lines
        string[] points = data.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);

        for (int i = 0; i <= 32; i++)
        {
            // Clean up the point data
            string cleanedPoint = points[i].Trim();

            // Replace dots with commas if necessary
            cleanedPoint = cleanedPoint.Replace('.', ',');

            // Split each line into coordinates
            string[] coordinates = cleanedPoint.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

            // Print the coordinates for debugging
            if (printDebug) { 
                Debug.Log("Coordinates: " + string.Join(", ", coordinates)); 
            }

            float x = float.Parse(coordinates[0]) / 35 + 3;  // Divide to fix the size and adjust to the screens center
            float y = float.Parse(coordinates[1]) / 35 - 2; 
            float z = float.Parse(coordinates[2]) / 35;

            // Invert y so that it matches the Unity coordinate system
            y = -y;

            // Print the converted coordinates
            // if (printDebug) { Debug.Log($"Converted Coordinates: x = {x}, y = {y}, z = {z}"); }

            Body[i].transform.localPosition = new Vector3(x, y, z); // Set the position of the keypoint of that body part
        }

        // Optional delay
        Thread.Sleep(10);
    }
}
