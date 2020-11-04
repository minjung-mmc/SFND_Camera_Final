# SFND_3D_Object_Tracking

## FP.1 Match 3D Objects
match the current and previous boundingbox that share the largest number of keypoints, and return index of boundingbox to std::map<int, int> &bbBestmatches.
```
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // match(queryIdx = prev, trainIdx = curr)
    cv::Mat matrix(currFrame.boundingBoxes.size(),prevFrame.boundingBoxes.size(),cv::DataType<int>::type, cv::Scalar(0));

    for (auto it0 = matches.begin(); it0 != matches.end(); ++it0)
    {
        int currIdx = -1, prevIdx = -1;

        cv::Point currPt = currFrame.keypoints[it0->trainIdx].pt;
        for (auto it1 = currFrame.boundingBoxes.begin(); it1 != currFrame.boundingBoxes.end(); ++it1)
        {            
            if (it1->roi.contains(currPt) == true)
            {
                currIdx = it1->boxID;
                break;
            }
        }
        if(currIdx < 0) continue;

        cv::Point prevPt = prevFrame.keypoints[it0->queryIdx].pt;
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2)
        {
            if (it2->roi.contains(prevPt) == true)
            {
                prevIdx = it2->boxID;
                break;
            }
        }

        if (prevIdx >= 0)
        {
            ++(matrix.at<int>(currIdx, prevIdx));
        }
    }
    
    // Find best match (prevBoxIdx, currentBoxIdx)
    if (matrix.rows < matrix.cols)
    {
        for (int r = 0; r < matrix.rows; ++r)
        {
            int maxIdx = 0;
            for (int c = 0; c < matrix.cols; ++c)
            {
                if (matrix.at<int>(r, c) > matrix.at<int>(r, maxIdx))
                {
                    maxIdx = c;
                }
            }
            bbBestMatches.insert(pair<int, int>(maxIdx, r));
        }
    }
    else
    {
        for (int c = 0; c < matrix.rows; ++c)
        {
            int maxIdx = 0;
            for (int r = 0; r < matrix.cols; ++r)
            {
                if (matrix.at<int>(r, c) > matrix.at<int>(maxIdx, c))
                {
                    maxIdx = r;
                }
            }
            bbBestMatches.insert(pair<int, int>(c, maxIdx));
        }
    }
}
```

## FP.2 Compute Lidar-based TTC
Compute the time-to-collision for all matched 3D objects based on Lidar measurements alone. Implement the estimation in a way that makes it robust against outliers which might be way too close and thus lead to faulty estimates of the TTC.
-> I found Lidar points within ego line, and computed median value of those lidar points to estimate TTC.
```
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double laneWidth = 4.0;
    double dT = 1 / frameRate;

    // find Lidar points within ego lane
    vector<double> Xprev, Xcurr;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (abs( it->y ) <= laneWidth/2.0 )
        {
            Xprev.push_back(it->x);
        }
    }
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if (abs( it->y ) <= laneWidth/2.0 )
        {
            Xcurr.push_back(it->x);
        }
    }

    // compute median value of Lidar point X
    std::sort(Xprev.begin(), Xprev.end());
    int medIndexPrev = floor(Xprev.size() / 2.0); 
    double medXprev = Xprev.size() % 2 == 0 ? (Xprev[medIndexPrev - 1] + Xprev[medIndexPrev]) / 2.0 : Xprev[medIndexPrev];

    std::sort(Xcurr.begin(), Xcurr.end());
    int medIndexCurr = floor(Xcurr.size() / 2.0); 
    double medXcurr = Xcurr.size() % 2 == 0 ? (Xcurr[medIndexCurr - 1] + Xcurr[medIndexCurr]) / 2.0 : Xcurr[medIndexCurr]; 
    
    // compute TTC
    TTC = medXcurr * dT / (medXprev - medXcurr);
}
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
Find all keypoint matches that belong to each 3D object. You can do this by simply checking whether the corresponding keypoints are within the region of interest in the camera image. Compute a robust mean of all the euclidean distances between keypoint matches and then remove those that are too far away from the mean. Store the result in the "kptMatches" property of the respective bounding box.
-> I used bottom 80% of the euclidean distance, and assumed the rest as outliers.
```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<double> euclidDistance;
    std::vector<cv::DMatch> boxedMatches;
    double outlierRatio = 0.2;

    // Find matches inside the bounding box
    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        if (boundingBox.roi.contains(kptsCurr[it->trainIdx].pt) == true)
        {
            euclidDistance.push_back(cv::norm(cv::Mat(kptsCurr[it->trainIdx].pt), cv::Mat(kptsPrev[it->queryIdx].pt)) );
        }
    }

    sort(euclidDistance.begin(), euclidDistance.end());

    for (auto it = kptMatches.begin(); it != kptMatches.end(); ++it)
    {
        double distance;
        if (boundingBox.roi.contains(kptsCurr[it->trainIdx].pt) == true)
        {
            distance = cv::norm(cv::Mat(kptsCurr[it->trainIdx].pt), cv::Mat(kptsPrev[it->queryIdx].pt));

            if (distance <= threshold)
            {
                boundingBox.kptMatches.push_back(*it);
                boundingBox.keypoints.push_back(kptsCurr[it->trainIdx]);
            }
        }
    }
}

```

## FP.4 Compute Camera-based TTC
Compute the time-to-collision for all matched 3D objects based on Camera measurements. Implement the estimation in a way that makes it robust against outliers which might be way too close and thus lead to faulty estimates of the TTC.
-> I used median value of distance ratio. 

```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    } //eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // use median distance ratio
    std::sort(distRatios.begin(), distRatios.end()); // sort distRatios
    int medIndex = floor(distRatios.size() / 2.0); //get medindex   ex.)if 3.5 -> 3
    // if even number, get two median values and mean of them. index starts with 0.
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; 
    
    double dT = 1 / frameRate;
    TTC = - dT / (1 - medDistRatio);
}
```

## FP.5 Performance Evaluation 1
Look for several examples where you have the impression that the Lidar-based TTC estimate is way off. Once you have found those, describe your observations and provide a sound argumentation why you think this happened. 


Since the model of constant-velocity is used, TTC of both Lidar and Camera should decrease.
However, as you can see down below, the best performance model from Midterm project (based on matching rate) showed a little increase at frame 3~5, 8~9, and 11. This tendency has also been shown for other detectors and descriptors.

### Shi-Tomasi detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     13.6733 | 
|  Frame 2   |        12.6142    |     14.3375 | 
|  Frame 3   |        **14.091**    |     15.4413 | 
|  Frame 4   |        **16.6894**    |     12.0075 | 
|  Frame 5   |        **15.7465**    |     13.2483 | 
|  Frame 6   |        12.7835    |     15.3175 | 
|  Frame 7   |        11.9844    |     14.1036 | 
|  Frame 8   |        **13.1241**    |     14.3413 | 
|  Frame 9   |        **13.0241**    |     12.2953 | 
|  Frame 10   |        11.1746    |     14.0712 | 
|  Frame 11   |        **12.8086**    |     12.0367 | 
|  Frame 12   |        8.95978    |     12.1312 | 
|  Frame 13   |        9.96439    |     12.9306 | 
|  Frame 14   |        9.59863    |     13.4865 | 
|  Frame 15   |        8.52157    |     13.0675 | 
|  Frame 16   |        9.51552    |     12.8883 | 
|  Frame 17   |        9.61241    |     12.9008 | 
|  Frame 18   |        8.3988    |     10.8327 | 

[Photo of Lidar Points](./Reference/photoOfLidarPoints.jpg.jpg)


**Why sudden Increase?**
We  manually estimated the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points. 
There is the photo of the cloud of the preceding vehicle's rear part. As you can see in the picture, there are points on the back of the car that are significantly different from other Lidar points. The median value is calculated by x coordinates of these points, which pushed the median value to the back. As a result, the value of medXprev among TTC = medXcurr * dT / (medXprev - medXcurr) was calculated shorter, which led to a larger TTC value.

This can be improved by eliminating the points which is noticeably off, and then calculate median. 
One of the other ways to solve this problem is to use a bigger shrinkFactor(=0.1~0.2) in order to get more stable and reliable cloud points within interesting bounding box.

## FP.6 Performance Evaluation 2
Run the different detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. Describe your observations again and also look into potential reasons.

1. The use of Harris Detector tended to cause poor performance of TTC using camera.

### Harris detector & BRISK descriptor 

|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:-----------------:|:----------------------:|:------------------------:|
|  Frame 1   |        12.5156    |     10.9082 | 
|  Frame 2   |        12.6142    |     10.586 | 
|  Frame 3   |        14.091    |     26.2905 | 
|  Frame 4   |        16.6894    |     11.7693 | 
|  Frame 5   |        15.7465    |     -inf | 
|  Frame 6   |        12.7835    |     12.9945 | 
|  Frame 7   |        11.9844    |     12.2792 | 
|  Frame 8   |        13.1241    |     12.9162 | 
|  Frame 9   |        13.0241    |     nan | 
|  Frame 10   |        11.1746    |     -inf | 
|  Frame 11   |        12.8086    |     11.2142 | 
|  Frame 12   |        8.95978    |     12.245 | 
|  Frame 13   |        9.96439    |     13.456 | 
|  Frame 14   |        9.59863    |     5.6061 | 
|  Frame 15   |        8.52157    |     -13.6263 | 
|  Frame 16   |        9.51552    |     6.33866 | 
|  Frame 17   |        9.61241    |     12.5848 | 
|  Frame 18   |        8.3988    |     -inf | 

### Harris detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     10.9082 | 
|  Frame 2   |        12.6142    |     10.586 | 
|  Frame 3   |        14.091    |     -10.8522 | 
|  Frame 4   |        16.6894    |     12.1284 | 
|  Frame 5   |        15.7465    |     35.3833 | 
|  Frame 6   |        12.7835    |     13.5907 | 
|  Frame 7   |        11.9844    |     12.2792 | 
|  Frame 8   |        13.0241    |     3.30058 | 
|  Frame 9   |        11.1746    |     20.5862 | 
|  Frame 10   |        12.8086    |     11.8135 | 
|  Frame 11   |        8.95978    |     nan | 
|  Frame 12   |        9.96439    |     -inf | 
|  Frame 13   |        9.59863    |     5.6061 | 
|  Frame 14   |        8.52157    |     -13.6263 | 
|  Frame 15   |        9.51552    |     6.6376 | 
|  Frame 16   |        9.61241    |     12.5848 | 
|  Frame 17   |        8.3988    |     -inf | 


This can be explained by number of keypoints that harris detect. Number of keypoints Harris detected was significantly small, compared to other detectors.

### Number of Keypoints

|Detector       | Keypoint #(10 frames - ROI)	 	 	 | Keypoint #(Average - ROI)|
| ------------- | ---------------------------------------------	 |:---------------------:   |
| Harris        |17, 14, 18, 21, 26, 43, 18, 31, 26, 34 	 |24.80 		    |
| Shi-Tomasi	|125, 118, 123, 120, 120, 113, 114, 123, 111 ,112|117.90      		    |
| FAST	        |149, 152, 150, 155, 149, 149, 156, 150, 138, 143|149.10      		    |
| BRISK		|264, 282, 282, 277, 297, 279, 289, 272, 266, 254|276.20       		    |
| ORB		|92, 102, 106, 113, 109, 125, 130, 129, 127, 128 |116.10       		    |
| AKAZE		|166, 157, 161, 155, 163, 164, 173, 175, 177, 179|167.00      		    |
| SIFT    	|138, 132, 124, 137, 134, 140, 137, 148, 159, 137|138.60       		    |

And this led to small number of keypoints within current bounding box, and this lack of data led to miscalculation of distance ratios between current and previous frame. And as a result, the calculated value of the TTC is not stable.

For example, look at the Euclid distance computation of Harris & BRISK.

[Euclidean distance within frame 2 & 3]
 0 0 0 1 1.41421 1.41421

Then, by 
```
double outlierRatio = 0.2;
int idx = floor(euclidDistance.size() * (1 - outlierRatio));
```
only 4 keypoints remain in the bounding box. 

2. ORB detector, which also has a small number of key points and poor matching rates, also showed poor performance in general.

### ORB detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.4733 | 
|  Frame 2   |        12.6142    |     20.5211 | 
|  Frame 3   |        14.091    |     10.5726 | 
|  Frame 4   |        15.7465    |     -inf | 
|  Frame 5   |        11.9844    |     -inf | 
|  Frame 6   |        13.1241    |     9.02151 | 
|  Frame 7   |        13.0241    |     22.8248 | 
|  Frame 8   |        11.1746    |     -inf | 
|  Frame 9   |        12.8086    |     8.76336 | 
|  Frame 10   |        8.95978    |     8.41658 | 
|  Frame 11   |        9.96439    |     8.51436 | 
|  Frame 12   |        9.59863    |     27.8458 | 
|  Frame 13   |        8.52157    |     -inf | 
|  Frame 14   |        9.51552    |     10.8449 | 
|  Frame 15   |        9.61241    |     9.93894 | 
|  Frame 16   |        8.3988    |     7.62862 | 

**Why nan?**
```
if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
{ // avoid division by zero

    double distRatio = distCurr / distPrev;
    distRatios.push_back(distRatio);
}

...
...

// only continue if list of distance ratios is not empty
if (distRatios.size() == 0)
{
    TTC = NAN;
    return;
}
'''
When distance between two keypoints in current frame image is smaller than minDist, size or distRatios becomes zero then returns TTC = nan. 

**Why -inf?**
This occurs when lack of keypoint data, all calculated inliers of euclidean distance between previous and current keypoints are 0. Then median value of Distance ratio becomes 1.
```
TTC = - dT / (1 - medDistRatio);
```
Then TTC becomes -inf.

**The best Combinations**

The least "increase of TTC between frame (=error)" was rated as having the best performance. The Shi-Tomasi detector & BRIEF descriptor, which has the smallest value of 2.5984s, was chosen as the best combination.

1. Run Time
Best Performance : FAST + BRIEF

2. Matching Rate
Best Performance : Shi-Tomasi + BRIEF

3. Increase of TTC (=error)
Best Performance : Shi-Tomasi + BRIEF

**TTC of Detector & Descriptor Combinations**

※ Difference between the number of image frame occured due to no observed bounding box using YOLO. This can be fixed by setting greater size of divided cells in pixels.
```
    cv::Size size = cv::Size(416, 416);
    // cv::Size size = cv::Size(608, 608); 
```


### Harris detector & BRISK descriptor 

|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:-----------------:|:----------------------:|:------------------------:|
|  Frame 1   |        12.5156    |     10.9082 | 
|  Frame 2   |        12.6142    |     10.586 | 
|  Frame 3   |        14.091    |     26.2905 | 
|  Frame 4   |        16.6894    |     11.7693 | 
|  Frame 5   |        15.7465    |     -inf | 
|  Frame 6   |        12.7835    |     12.9945 | 
|  Frame 7   |        11.9844    |     12.2792 | 
|  Frame 8   |        13.1241    |     12.9162 | 
|  Frame 9   |        13.0241    |     nan | 
|  Frame 10   |        11.1746    |     -inf | 
|  Frame 11   |        12.8086    |     11.2142 | 
|  Frame 12   |        8.95978    |     12.245 | 
|  Frame 13   |        9.96439    |     13.456 | 
|  Frame 14   |        9.59863    |     5.6061 | 
|  Frame 15   |        8.52157    |     -13.6263 | 
|  Frame 16   |        9.51552    |     6.33866 | 
|  Frame 17   |        9.61241    |     12.5848 | 
|  Frame 18   |        8.3988    |     -inf | 

### Harris detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     10.9082 | 
|  Frame 2   |        12.6142    |     10.586 | 
|  Frame 3   |        14.091    |     -10.8522 | 
|  Frame 4   |        16.6894    |     12.1284 | 
|  Frame 5   |        15.7465    |     35.3833 | 
|  Frame 6   |        12.7835    |     13.5907 | 
|  Frame 7   |        11.9844    |     12.2792 | 
|  Frame 8   |        13.0241    |     3.30058 | 
|  Frame 9   |        11.1746    |     20.5862 | 
|  Frame 10   |        12.8086    |     11.8135 | 
|  Frame 11   |        8.95978    |     nan | 
|  Frame 12   |        9.96439    |     -inf | 
|  Frame 13   |        9.59863    |     5.6061 | 
|  Frame 14   |        8.52157    |     -13.6263 | 
|  Frame 15   |        9.51552    |     6.6376 | 
|  Frame 16   |        9.61241    |     12.5848 | 
|  Frame 17   |        8.3988    |     -inf | 

### Harris detector & ORB descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     10.9082 | 
|  Frame 2   |        12.6142    |     10.586 | 
|  Frame 3   |        14.091    |     -11.4731 | 
|  Frame 4   |        16.6894    |     11.7693 | 
|  Frame 5   |        15.7465    |     35.3833 | 
|  Frame 6   |        12.7835    |     13.5907 | 
|  Frame 7   |        11.9844    |     13.497 | 
|  Frame 8   |        13.1241    |     17.6204 | 
|  Frame 9   |        13.0241    |     nan | 
|  Frame 10   |        11.1746    |     nan | 
|  Frame 11   |        12.8086    |     11.2142 | 
|  Frame 12   |        8.95978    |     11.9536 | 
|  Frame 13   |        9.96439    |     13.4327 | 
|  Frame 14   |        9.59863    |     5.85828 | 
|  Frame 15   |        8.52157    |     -12.639 | 
|  Frame 16   |        9.51552    |     6.60338 | 
|  Frame 17   |        9.61241    |     12.5848 | 
|  Frame 18   |        8.3988    |     -inf |

### Harris detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     9.74953 | 
|  Frame 2   |        12.6142    |     10.586 | 
|  Frame 3   |        14.091    |     nan | 
|  Frame 4   |        16.6894    |     nan | 
|  Frame 5   |        15.7465    |     44.9166 | 
|  Frame 6   |        12.7835    |     14.1981 | 
|  Frame 7   |        11.9844    |     12.2 | 
|  Frame 8   |        13.0241    |     nan | 
|  Frame 9   |        11.1746    |     10.2931 | 
|  Frame 10   |        12.8086    |     23.627 | 
|  Frame 11   |        8.95978    |     11.9536 | 
|  Frame 12   |        9.96439    |     nan | 
|  Frame 13   |        9.59863    |     nan | 
|  Frame 14   |        8.52157    |     -25.2781 | 
|  Frame 15   |        9.51552    |     6.87135 | 
|  Frame 16   |        9.61241    |     11.1009 | 
|  Frame 17   |        8.3988    |     -inf | 

### Harris detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     10.9082 | 
|  Frame 2   |        12.6142    |     11.0081 | 
|  Frame 3   |        14.091    |     -11.4731 | 
|  Frame 4   |        15.7465    |     35.3833 | 
|  Frame 5   |        11.9844    |     13.1905 | 
|  Frame 6   |        13.1241    |     17.6204 | 
|  Frame 7   |        13.0241    |     3.30058 | 
|  Frame 8   |        11.1746    |     20.5862 | 
|  Frame 9   |        12.8086    |     11.7414 | 
|  Frame 10   |        8.95978    |     12.245 | 
|  Frame 11   |        9.96439    |     568.322 | 
|  Frame 12   |        9.59863    |     5.6061 | 
|  Frame 13   |        8.52157    |     -13.6263 | 
|  Frame 14   |        9.51552    |     7.03775 | 
|  Frame 15   |        9.61241    |     12.5848 | 
|  Frame 16   |        8.3988    |     -inf |

### Shi-Tomasi detector & BRISK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     13.049 | 
|  Frame 2   |        12.6142    |     12.6968 | 
|  Frame 3   |        14.091    |     14.4793 | 
|  Frame 4   |        16.6894    |     12.7502 | 
|  Frame 5   |        15.7465    |     12.5109 | 
|  Frame 6   |        12.7835    |     15.1536 | 
|  Frame 7   |        11.9844    |     13.1788 | 
|  Frame 8   |        13.1241    |     14.2932 | 
|  Frame 9   |        13.0241    |     12.5494 | 
|  Frame 10   |        11.1746    |     13.3275 | 
|  Frame 11   |        12.8086    |     11.494 | 
|  Frame 12   |        8.95978    |     12.287 | 
|  Frame 13   |        9.96439    |     13.3352 | 
|  Frame 14   |        9.59863    |     12.6532 | 
|  Frame 15   |        8.52157    |     11.8987 | 
|  Frame 16   |        9.51552    |     11.6088 | 
|  Frame 17   |        9.61241    |     12.828 | 
|  Frame 18   |        8.3988    |     11.4955 |

### Shi-Tomasi detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     13.6733 | 
|  Frame 2   |        12.6142    |     14.3375 | 
|  Frame 3   |        14.091    |     15.4413 | 
|  Frame 4   |        16.6894    |     12.0075 | 
|  Frame 5   |        15.7465    |     13.2483 | 
|  Frame 6   |        12.7835    |     15.3175 | 
|  Frame 7   |        11.9844    |     14.1036 | 
|  Frame 8   |        13.1241    |     14.3413 | 
|  Frame 9   |        13.0241    |     12.2953 | 
|  Frame 10   |        11.1746    |     14.0712 | 
|  Frame 11   |        12.8086    |     12.0367 | 
|  Frame 12   |        8.95978    |     12.1312 | 
|  Frame 13   |        9.96439    |     12.9306 | 
|  Frame 14   |        9.59863    |     13.4865 | 
|  Frame 15   |        8.52157    |     13.0675 | 
|  Frame 16   |        9.51552    |     12.8883 | 
|  Frame 17   |        9.61241    |     12.9008 | 
|  Frame 18   |        8.3988    |     10.8327 | 

### Shi-Tomasi detector & ORB descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.1064 | 
|  Frame 2   |        12.6142    |     12.3445 | 
|  Frame 3   |        14.091    |     13.629 | 
|  Frame 4   |        16.6894    |     12.4346 | 
|  Frame 5   |        15.7465    |     13.2534 | 
|  Frame 6   |        12.7835    |     15.917 | 
|  Frame 7   |        11.9844    |     15.0065 | 
|  Frame 8   |        13.1241    |     15.0765 | 
|  Frame 9   |        13.0241    |     11.6077 | 
|  Frame 10   |        11.1746    |     14.0712 | 
|  Frame 11   |        12.8086    |     11.6215 | 
|  Frame 12   |        8.95978    |     12.1102 | 
|  Frame 13   |        9.96439    |     12.6502 | 
|  Frame 14   |        9.59863    |     12.5716 | 
|  Frame 15   |        8.52157    |     10.9042 | 
|  Frame 16   |        9.51552    |     13.1503 | 
|  Frame 17   |        9.61241    |     11.1201 | 
|  Frame 18   |        8.3988    |     11.1899 |

### Shi-Tomasi detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     13.0182 | 
|  Frame 2   |        12.6142    |     12.5187 | 
|  Frame 3   |        14.091    |     14.2771 | 
|  Frame 4   |        15.7465    |     12.7336 | 
|  Frame 5   |        12.7835    |     13.4761 | 
|  Frame 6   |        11.9844    |     13.4548 | 
|  Frame 7   |        13.1241    |     12.0155 | 
|  Frame 8   |        13.0241    |     12.5073 | 
|  Frame 9   |        11.1746    |     14.0134 | 
|  Frame 10   |        12.8086    |     11.7837 | 
|  Frame 11   |        8.95978    |     12.9405 | 
|  Frame 12   |        9.96439    |     13.6254 | 
|  Frame 13   |        9.59863    |     12.241 | 
|  Frame 14   |        8.52157    |     12.6641 | 
|  Frame 15   |        9.51552    |     10.3773 | 
|  Frame 16   |        9.61241    |     13.1247 | 
|  Frame 17   |        8.3988    |     10.6119 | 
|  Frame 18   |        6.915    |     1.00815 | 

### Shi-Tomasi detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.0746 | 
|  Frame 2   |        12.6142    |     14.6618 | 
|  Frame 3   |        14.091    |     17.181 | 
|  Frame 4   |        16.6894    |     13.0192 | 
|  Frame 5   |        15.7465    |     13.3479 | 
|  Frame 6   |        12.7835    |     14.1323 | 
|  Frame 7   |        11.9844    |     13.935 | 
|  Frame 8   |        13.1241    |     14.5085 | 
|  Frame 9   |        13.0241    |     12.2319 | 
|  Frame 10   |        11.1746    |     13.7396 | 
|  Frame 11   |        12.8086    |     12.3562 | 
|  Frame 12   |        8.95978    |     12.269 | 
|  Frame 13   |        9.96439    |     12.8498 | 
|  Frame 14   |        9.59863    |     12.7102 | 
|  Frame 15   |        8.52157    |     12.3292 | 
|  Frame 16   |        9.51552    |     13.1068 | 
|  Frame 17   |        9.61241    |     11.5312 | 
|  Frame 18   |        8.3988    |     10.9222 | 

### FAST detector & BRISK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     12.313 | 
|  Frame 2   |        12.6142    |     13.5384 | 
|  Frame 3   |        14.091    |     13.9572 | 
|  Frame 4   |        16.6894    |     13.0212 | 
|  Frame 5   |        15.7465    |     99.5628 | 
|  Frame 6   |        11.9844    |     13.1858 | 
|  Frame 7   |        13.1241    |     11.9077 | 
|  Frame 8   |        13.0241    |     13.4042 | 
|  Frame 9   |        11.1746    |     14.3255 | 
|  Frame 10   |        12.8086    |     12.3427 | 
|  Frame 11   |        8.95978    |     13.5201 | 
|  Frame 12   |        9.96439    |     12.7784 | 
|  Frame 13   |        9.59863    |     12.879 | 
|  Frame 14   |        8.52157    |     13.0655 | 
|  Frame 15   |        9.51552    |     12.9162 | 
|  Frame 16   |        9.61241    |     11.0958 | 
|  Frame 17   |        8.3988    |     12.3793 | 

### FAST detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     12.5 | 
|  Frame 2   |        12.6142    |     13.0256 | 
|  Frame 3   |        14.091    |     13.9811 | 
|  Frame 4   |        15.7465    |     331.161 | 
|  Frame 5   |        12.7835    |     13.6183 | 
|  Frame 6   |        11.9844    |     12.13 | 
|  Frame 7   |        13.1241    |     12.0007 | 
|  Frame 8   |        13.0241    |     14.808 | 
|  Frame 9   |        11.1746    |     13.1088 | 
|  Frame 10   |        12.8086    |     14.7288 | 
|  Frame 11   |        8.95978    |     11.7494 | 
|  Frame 12   |        9.96439    |     12.8743 | 
|  Frame 13   |        9.59863    |     12.5199 | 
|  Frame 14   |        8.52157    |     12.1879 | 
|  Frame 15   |        9.51552    |     12.5159 | 
|  Frame 16   |        9.61241    |     9.82934 | 
|  Frame 17   |        8.3988    |     12.5588 | 

### FAST detector & ORB descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     10.7912 | 
|  Frame 2   |        12.6142    |     12.7571 | 
|  Frame 3   |        14.091    |     20.9306 | 
|  Frame 4   |        15.7465    |     244.85 | 
|  Frame 5   |        11.9844    |     13.3198 | 
|  Frame 6   |        13.0241    |     13.4602 | 
|  Frame 7   |        11.1746    |     13.5653 | 
|  Frame 8   |        12.8086    |     14.558 | 
|  Frame 9   |        8.95978    |     13.6506 | 
|  Frame 10   |        9.96439    |     12.8506 | 
|  Frame 11   |        9.59863    |     12.1368 | 
|  Frame 12   |        8.52157    |     13.9397 | 
|  Frame 13   |        9.51552    |     12.2753 | 
|  Frame 14   |        9.61241    |     11.7698 | 
|  Frame 15   |        8.3988    |     13.2302 | 

### FAST detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     8.44411 | 
|  Frame 2   |        12.6142    |     27.8578 | 
|  Frame 3   |        14.091    |     12.1376 | 
|  Frame 4   |        15.7465    |     13.0459 | 
|  Frame 5   |        12.7835    |     12.6664 | 
|  Frame 6   |        11.9844    |     12.1002 | 
|  Frame 7   |        13.1241    |     11.7849 | 
|  Frame 8   |        13.0241    |     13.1008 | 
|  Frame 9   |        11.1746    |     13.8308 | 
|  Frame 10   |        12.8086    |     12.5157 | 
|  Frame 11   |        8.95978    |     13.3321 | 
|  Frame 12   |        9.96439    |     12.8514 | 
|  Frame 13   |        9.59863    |     12.1368 | 
|  Frame 14   |        8.52157    |     12.0969 | 
|  Frame 15   |        9.51552    |     12.2808 | 
|  Frame 16   |        9.61241    |     11.8476 | 
|  Frame 17   |        8.3988    |     12.6702 | 

### FAST detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     12.3317 | 
|  Frame 2   |        12.6142    |     13.4194 | 
|  Frame 3   |        14.091    |     22.7505 | 
|  Frame 4   |        15.7465    |     199.157 | 
|  Frame 5   |        12.7835    |     13.9137 | 
|  Frame 6   |        11.9844    |     12.2781 | 
|  Frame 7   |        13.1241    |     12.6536 | 
|  Frame 8   |        13.0241    |     13.3673 | 
|  Frame 9   |        11.1746    |     14.0112 | 
|  Frame 10   |        12.8086    |     15.8801 | 
|  Frame 11   |        8.95978    |     12.7731 | 
|  Frame 12   |        9.96439    |     12.9215 | 
|  Frame 13   |        9.59863    |     12.8223 | 
|  Frame 14   |        8.52157    |     12.5997 | 
|  Frame 15   |        9.51552    |     12.9069 | 
|  Frame 16   |        9.61241    |     10.5264 | 
|  Frame 17   |        8.3988    |     12.0819 | 

### BRISK detector & BRISK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.4528 | 
|  Frame 2   |        12.6142    |     36.8812 | 
|  Frame 3   |        14.091    |     21.4969 | 
|  Frame 4   |        16.6894    |     18.4003 | 
|  Frame 5   |        15.7465    |     30.382 | 
|  Frame 6   |        11.9844    |     21.0793 | 
|  Frame 7   |        13.1241    |     23.1712 | 
|  Frame 8   |        13.0241    |     20.4435 | 
|  Frame 9   |        11.1746    |     16.2296 | 
|  Frame 10   |        12.8086    |     15.1292 | 
|  Frame 11   |        8.95978    |     15.3676 | 
|  Frame 12   |        9.96439    |     16.8161 | 
|  Frame 13   |        9.59863    |     15.91 | 
|  Frame 14   |        8.52157    |     16.2868 | 
|  Frame 15   |        9.51552    |     14.7862 | 
|  Frame 16   |        9.61241    |     12.2143 | 
|  Frame 17   |        8.3988    |     16.2886 | 

### BRISK detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.8273 | 
|  Frame 2   |        12.6142    |     19.013 | 
|  Frame 3   |        14.091    |     16.8644 | 
|  Frame 4   |        16.6894    |     20.3782 | 
|  Frame 5   |        15.7465    |     15.4176 | 
|  Frame 6   |        11.9844    |     17.0509 | 
|  Frame 7   |        13.1241    |     18.824 | 
|  Frame 8   |        13.0241    |     21.0641 | 
|  Frame 9   |        11.1746    |     17.3253 | 
|  Frame 10   |        12.8086    |     15.8866 | 
|  Frame 11   |        8.95978    |     16.0764 | 
|  Frame 12   |        9.96439    |     17.4274 | 
|  Frame 13   |        9.59863    |     15.0951 | 
|  Frame 14   |        8.52157    |     16.0729 | 
|  Frame 15   |        9.51552    |     15.9585 | 
|  Frame 16   |        9.61241    |     10.8117 | 
|  Frame 17   |        8.3988    |     15.6634 | 

### BRISK detector & ORB descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.1922 | 
|  Frame 2   |        12.6142    |     30.0422 | 
|  Frame 3   |        14.091    |     17.943 | 
|  Frame 4   |        16.6894    |     16.6317 | 
|  Frame 5   |        15.7465    |     22.2484 | 
|  Frame 6   |        12.7835    |     34.0073 | 
|  Frame 7   |        11.9844    |     16.1467 | 
|  Frame 8   |        13.1241    |     17.5377 | 
|  Frame 9   |        13.0241    |     21.8078 | 
|  Frame 10   |        11.1746    |     12.8042 | 
|  Frame 11   |        12.8086    |     15.3598 | 
|  Frame 12   |        8.95978    |     14.7782 | 
|  Frame 13   |        9.96439    |     13.4943 | 
|  Frame 14   |        9.59863    |     20.4401 | 
|  Frame 15   |        8.52157    |     13.3384 | 
|  Frame 16   |        9.51552    |     14.2821 | 
|  Frame 17   |        9.61241    |     9.70105 | 
|  Frame 18   |        8.3988    |     15.1448 | 

### BRISK detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     22.3338 | 
|  Frame 2   |        12.6142    |     24.1462 | 
|  Frame 3   |        14.091    |     17.3415 | 
|  Frame 4   |        15.7465    |     45.6935 | 
|  Frame 5   |        12.7835    |     19.6499 | 
|  Frame 6   |        11.9844    |     20.719 | 
|  Frame 7   |        13.1241    |     19.7292 | 
|  Frame 8   |        13.0241    |     19.3079 | 
|  Frame 9   |        11.1746    |     13.1305 | 
|  Frame 10   |        12.8086    |     16.1616 | 
|  Frame 11   |        8.95978    |     14.6918 | 
|  Frame 12   |        9.96439    |     15.3799 | 
|  Frame 13   |        9.59863    |     15.9249 | 
|  Frame 14   |        8.52157    |     14.4594 | 
|  Frame 15   |        9.51552    |     13.3721 | 
|  Frame 16   |        9.61241    |     12.1469 | 
|  Frame 17   |        8.3988    |     13.8154 |

### BRISK detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     13.7651 | 
|  Frame 2   |        12.6142    |     17.4898 | 
|  Frame 3   |        14.091    |     16.9848 | 
|  Frame 4   |        15.7465    |     24.0727 | 
|  Frame 5   |        12.7835    |     18.8188 | 
|  Frame 6   |        11.9844    |     16.4298 | 
|  Frame 7   |        13.1241    |     16.8197 | 
|  Frame 8   |        13.0241    |     16.3258 | 
|  Frame 9   |        11.1746    |     16.2161 | 
|  Frame 10   |        12.8086    |     12.0368 | 
|  Frame 11   |        8.95978    |     11.8363 | 
|  Frame 12   |        9.96439    |     13.6835 | 
|  Frame 13   |        9.59863    |     11.48 | 
|  Frame 14   |        8.52157    |     13.3989 | 
|  Frame 15   |        9.51552    |     11.0579 | 
|  Frame 16   |        9.61241    |     11.4685 | 
|  Frame 17   |        8.3988    |     13.1412 | 

### ORB detector & BRISK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     19.6003 | 
|  Frame 2   |        12.6142    |     19.6382 | 
|  Frame 3   |        14.091    |     17.7441 | 
|  Frame 4   |        15.7465    |     156.224 | 
|  Frame 5   |        11.9844    |     12.9706 | 
|  Frame 6   |        13.1241    |     10.911 | 
|  Frame 7   |        13.0241    |     -inf | 
|  Frame 8   |        11.1746    |     -inf | 
|  Frame 9   |        12.8086    |     7.52121 | 
|  Frame 10   |        8.95978    |     -inf | 
|  Frame 11   |        9.96439    |     22.4629 | 
|  Frame 12   |        9.59863    |     35.3185 | 
|  Frame 13   |        8.52157    |     13.594 | 
|  Frame 14   |        9.51552    |     14.2951 | 
|  Frame 15   |        9.61241    |     23.6651 | 
|  Frame 16   |        8.3988    |     46.6458 | 

### ORB detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     16.5073 | 
|  Frame 2   |        12.6142    |     -inf | 
|  Frame 3   |        14.091    |     89.2313 | 
|  Frame 4   |        15.7465    |     24.0009 | 
|  Frame 5   |        11.9844    |     -inf | 
|  Frame 6   |        13.1241    |     -inf | 
|  Frame 7   |        13.0241    |     -inf | 
|  Frame 8   |        11.1746    |     12.8303 | 
|  Frame 9   |        12.8086    |     24.9891 | 
|  Frame 10   |        8.95978    |     16.9677 | 
|  Frame 11   |        9.96439    |     -inf | 
|  Frame 12   |        9.59863    |     21.8391 | 
|  Frame 13   |        8.52157    |     164.864 | 
|  Frame 14   |        9.51552    |     13.0005 | 
|  Frame 15   |        9.61241    |     16.8713 | 
|  Frame 16   |        8.3988    |     23.002 | 

### ORB detector & ORB descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     19.4709 | 
|  Frame 2   |        12.6142    |     -inf | 
|  Frame 3   |        14.091    |     23.9536 | 
|  Frame 4   |        15.7465    |     29.6544 | 
|  Frame 5   |        11.9844    |     -inf | 
|  Frame 6   |        13.1241    |     -inf | 
|  Frame 7   |        13.0241    |     -inf | 
|  Frame 8   |        11.1746    |     -inf | 
|  Frame 9   |        12.8086    |     9.35938 | 
|  Frame 10   |        8.95978    |     -inf | 
|  Frame 11   |        9.96439    |     -inf | 
|  Frame 12   |        9.59863    |     177.585 | 
|  Frame 13   |        8.52157    |     -inf | 
|  Frame 14   |        9.51552    |     26.3083 | 
|  Frame 15   |        9.61241    |     16.6667 | 
|  Frame 16   |        8.3988    |     -inf |

### ORB detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.4733 | 
|  Frame 2   |        12.6142    |     20.5211 | 
|  Frame 3   |        14.091    |     10.5726 | 
|  Frame 4   |        15.7465    |     -inf | 
|  Frame 5   |        11.9844    |     -inf | 
|  Frame 6   |        13.1241    |     9.02151 | 
|  Frame 7   |        13.0241    |     22.8248 | 
|  Frame 8   |        11.1746    |     -inf | 
|  Frame 9   |        12.8086    |     8.76336 | 
|  Frame 10   |        8.95978    |     8.41658 | 
|  Frame 11   |        9.96439    |     8.51436 | 
|  Frame 12   |        9.59863    |     27.8458 | 
|  Frame 13   |        8.52157    |     -inf | 
|  Frame 14   |        9.51552    |     10.8449 | 
|  Frame 15   |        9.61241    |     9.93894 | 
|  Frame 16   |        8.3988    |     7.62862 | 

### ORB detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     16.726 | 
|  Frame 2   |        12.6142    |     19.2415 | 
|  Frame 3   |        14.091    |     12.0957 | 
|  Frame 4   |        15.7465    |     502.13 | 
|  Frame 5   |        11.9844    |     -inf | 
|  Frame 6   |        13.1241    |     11.3168 | 
|  Frame 7   |        13.0241    |     -inf | 
|  Frame 8   |        11.1746    |     -inf | 
|  Frame 9   |        12.8086    |     7.44098 | 
|  Frame 10   |        8.95978    |     -inf | 
|  Frame 11   |        9.96439    |     9.91242 | 
|  Frame 12   |        9.59863    |     35.6903 | 
|  Frame 13   |        8.52157    |     33.3507 | 
|  Frame 14   |        9.51552    |     12.1158 | 
|  Frame 15   |        9.61241    |     21.392 | 
|  Frame 16   |        8.3988    |     20.1371 |

### AKAZE detector & BRISK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:| 
|  Frame 1   |        12.5156    |     16.8563 | 
|  Frame 2   |        12.6142    |     18.4929 | 
|  Frame 3   |        14.091    |     16.3504 | 
|  Frame 4   |        16.6894    |     17.5351 | 
|  Frame 5   |        15.7465    |     18.6746 | 
|  Frame 6   |        12.7835    |     20.3714 | 
|  Frame 7   |        11.9844    |     21.7319 | 
|  Frame 8   |        13.1241    |     17.9099 | 
|  Frame 9   |        13.0241    |     22.2591 | 
|  Frame 10   |        11.1746    |     14.6515 | 
|  Frame 11   |        12.8086    |     15.0257 | 
|  Frame 12   |        8.95978    |     15.3756 | 
|  Frame 13   |        9.96439    |     13.1185 | 
|  Frame 14   |        9.59863    |     12.7676 | 
|  Frame 15   |        8.52157    |     13.6149 | 
|  Frame 16   |        9.51552    |     13.0152 | 
|  Frame 17   |        9.61241    |     11.4466 | 
|  Frame 18   |        8.3988    |     10.6396 | 

### AKAZE detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     15.4228 | 
|  Frame 2   |        12.6142    |     19.1145 | 
|  Frame 3   |        14.091    |     15.181 | 
|  Frame 4   |        15.7465    |     18.8963 | 
|  Frame 5   |        11.9844    |     21.1049 | 
|  Frame 6   |        13.1241    |     19.3563 | 
|  Frame 7   |        13.0241    |     20.7751 | 
|  Frame 8   |        11.1746    |     14.1156 | 
|  Frame 9   |        12.8086    |     14.507 | 
|  Frame 10   |        8.95978    |     12.4654 | 
|  Frame 11   |        9.96439    |     13.3461 | 
|  Frame 12   |        9.59863    |     13.3441 | 
|  Frame 13   |        8.52157    |     11.9791 | 
|  Frame 14   |        9.51552    |     11.947 | 
|  Frame 15   |        9.61241    |     10.8779 | 
|  Frame 16   |        8.3988    |     10.6942 | 

### AKAZE detector & ORB descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.0321 | 
|  Frame 2   |        12.6142    |     16.64 | 
|  Frame 3   |        14.091    |     14.7873 | 
|  Frame 4   |        16.6894    |     15.3607 | 
|  Frame 5   |        15.7465    |     18.2013 | 
|  Frame 6   |        12.7835    |     16.9835 | 
|  Frame 7   |        11.9844    |     18.621 | 
|  Frame 8   |        13.1241    |     17.178 | 
|  Frame 9   |        13.0241    |     18.9619 | 
|  Frame 10   |        11.1746    |     13.7636 | 
|  Frame 11   |        12.8086    |     13.7961 | 
|  Frame 12   |        8.95978    |     12.7123 | 
|  Frame 13   |        9.96439    |     13.0723 | 
|  Frame 14   |        9.59863    |     14.2072 | 
|  Frame 15   |        8.52157    |     14.4576 | 
|  Frame 16   |        9.51552    |     12.5881 | 
|  Frame 17   |        9.61241    |     11.6888 | 
|  Frame 18   |        8.3988    |     9.8149 | 

### AKAZE detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     12.7377 | 
|  Frame 2   |        12.6142    |     17.5354 | 
|  Frame 3   |        14.091    |     16.1137 | 
|  Frame 4   |        16.6894    |     16.713 | 
|  Frame 5   |        15.7465    |     18.5138 | 
|  Frame 6   |        12.7835    |     21.009 | 
|  Frame 7   |        11.9844    |     20.135 | 
|  Frame 8   |        13.1241    |     16.9118 | 
|  Frame 9   |        13.0241    |     21.0994 | 
|  Frame 10   |        11.1746    |     14.9509 | 
|  Frame 11   |        12.8086    |     14.9483 | 
|  Frame 12   |        8.95978    |     16.5268 | 
|  Frame 13   |        9.96439    |     14.023 | 
|  Frame 14   |        9.59863    |     12.5126 | 
|  Frame 15   |        8.52157    |     12.6712 | 
|  Frame 16   |        9.51552    |     11.866 | 
|  Frame 17   |        9.61241    |     11.3796 | 
|  Frame 18   |        8.3988    |     10.6609 | 

### AKAZE detector & AKAZE descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.7813 | 
|  Frame 2   |        12.6142    |     18.2759 | 
|  Frame 3   |        14.091    |     15.2954 | 
|  Frame 4   |        15.7465    |     18.578 | 
|  Frame 5   |        12.7835    |     18.5209 | 
|  Frame 6   |        11.9844    |     17.6798 | 
|  Frame 7   |        13.1241    |     18.4169 | 
|  Frame 8   |        13.0241    |     18.8738 | 
|  Frame 9   |        11.1746    |     14.3961 | 
|  Frame 10   |        12.8086    |     14.7002 | 
|  Frame 11   |        8.95978    |     14.0471 | 
|  Frame 12   |        9.96439    |     13.1209 | 
|  Frame 13   |        9.59863    |     14.0193 | 
|  Frame 14   |        8.52157    |     12.7913 | 
|  Frame 15   |        9.51552    |     12.3797 | 
|  Frame 16   |        9.61241    |     10.9403 | 
|  Frame 17   |        8.3988    |     10.4431 | 

### AKAZE detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.275 | 
|  Frame 2   |        12.6142    |     17.924 | 
|  Frame 3   |        14.091    |     16.1258 | 
|  Frame 4   |        15.7465    |     19.5558 | 
|  Frame 5   |        12.7835    |     19.9529 | 
|  Frame 6   |        11.9844    |     17.8047 | 
|  Frame 7   |        13.1241    |     17.777 | 
|  Frame 8   |        13.0241    |     18.4054 | 
|  Frame 9   |        11.1746    |     13.9839 | 
|  Frame 10   |        12.8086    |     14.1883 | 
|  Frame 11   |        8.95978    |     14.3416 | 
|  Frame 12   |        9.96439    |     12.9535 | 
|  Frame 13   |        9.59863    |     13.3441 | 
|  Frame 14   |        8.52157    |     13.2259 | 
|  Frame 15   |        9.51552    |     11.9771 | 
|  Frame 16   |        9.61241    |     11.04 | 
|  Frame 17   |        8.3988    |     10.2489 |

### SIFT detector & BRISK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     16.3605 | 
|  Frame 2   |        12.6142    |     18.1968 | 
|  Frame 3   |        14.091    |     21.0449 | 
|  Frame 4   |        15.7465    |     21.3614 | 
|  Frame 5   |        12.7835    |     16.0845 | 
|  Frame 6   |        11.9844    |     20.3296 | 
|  Frame 7   |        13.0241    |     17.3057 | 
|  Frame 8   |        11.1746    |     15.38 | 
|  Frame 9   |        12.8086    |     14.1564 | 
|  Frame 10   |        8.95978    |     12.5129 | 
|  Frame 11   |        9.96439    |     13.0106 | 
|  Frame 12   |        9.59863    |     12.0078 | 
|  Frame 13   |        8.52157    |     10.1541 | 
|  Frame 14   |        9.51552    |     10.6313 | 
|  Frame 15   |        9.61241    |     10.9394 | 
|  Frame 16   |        8.3988    |     13.1621 |

### SIFT detector & BRIEF descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:-----------------:|:----------------------:|:------------------------:|
|  Frame 1   |        12.5156    |     13.2917 | 
|  Frame 2   |        12.6142    |     14.8783 | 
|  Frame 3   |        14.091    |     16.2329 | 
|  Frame 4   |        15.7465    |     20.9343 | 
|  Frame 5   |        12.7835    |     16.6981 | 
|  Frame 6   |        11.9844    |     18.3806 | 
|  Frame 7   |        13.1241    |     17.5379 | 
|  Frame 8   |        13.0241    |     14.234 | 
|  Frame 9   |        11.1746    |     12.5144 | 
|  Frame 10   |        12.8086    |     14.4165 | 
|  Frame 11   |        8.95978    |     12.6976 | 
|  Frame 12   |        9.96439    |     13.1667 | 
|  Frame 13   |        9.59863    |     13.0787 | 
|  Frame 14   |        8.52157    |     11.899 | 
|  Frame 15   |        9.51552    |     9.74405 | 
|  Frame 16   |        9.61241    |     11.2425 | 
|  Frame 17   |        8.3988    |     12.7964 | 

### SIFT detector & FREAK descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:-----------------:|:----------------------:|:------------------------:|
|  Frame 1   |        12.5156    |     16.0812 | 
|  Frame 2   |        12.6142    |     17.7462 | 
|  Frame 3   |        14.091    |     17.4939 | 
|  Frame 4   |        15.7465    |     21.2313 | 
|  Frame 5   |        12.7835    |     15.6675 | 
|  Frame 6   |        11.9844    |     21.2261 | 
|  Frame 7   |        13.1241    |     17.3051 | 
|  Frame 8   |        13.0241    |     17.3556 | 
|  Frame 9   |        11.1746    |     13.8167 | 
|  Frame 10   |        12.8086    |     15.1919 | 
|  Frame 11   |        8.95978    |     13.2843 | 
|  Frame 12   |        9.96439    |     11.8665 | 
|  Frame 13   |        9.59863    |     15.199 | 
|  Frame 14   |        8.52157    |     11.7308 | 
|  Frame 15   |        9.51552    |     9.73895 | 
|  Frame 16   |        9.61241    |     10.0683 | 
|  Frame 17   |        8.3988    |     14.2603 |

### SIFT detector & SIFT descriptor
|Image Frame      |      TTC Lidar(s)   |     TTC Camera(s)     |
|:--------------:|:-------------------:|:---------------------:|
|  Frame 1   |        12.5156    |     14.738 | 
|  Frame 2   |        12.6142    |     15.8621 | 
|  Frame 3   |        14.091    |     15.4252 | 
|  Frame 4   |        15.7465    |     18.44 | 
|  Frame 5   |        12.7835    |     15.273 | 
|  Frame 6   |        11.9844    |     16.1902 | 
|  Frame 7   |        13.1241    |     18.5833 | 
|  Frame 8   |        13.0241    |     14.4655 | 
|  Frame 9   |        11.1746    |     13.0114 | 
|  Frame 10   |        12.8086    |     13.5342 | 
|  Frame 11   |        8.95978    |     14.5151 | 
|  Frame 12   |        9.96439    |     12.7747 | 
|  Frame 13   |        9.59863    |     12.9075 | 
|  Frame 14   |        8.52157    |     10.5037 | 
|  Frame 15   |        9.51552    |     10.5037 | 
|  Frame 16   |        9.61241    |     9.52376 | 
|  Frame 17   |        8.3988    |     11.4407 |

