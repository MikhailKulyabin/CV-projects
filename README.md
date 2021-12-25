# 1. Box detection
Size estimation of a box from a distance image.  
The approach consists of finding two planes that approximate the floor and the top of
the box and calculate the distance between the planes. A simple solution to finding dominant planes in
a point cloud is to use RANSAC to find the plane models with the most inliers, in our example the
floor and the top of the box.

![Input image and mask](/images/1/mask.png "Input image and mask")  
![Detected planes](/images/1/planes.png "Detected planes")  
![Detected corners](/images/1/corners.png "Detected corners")  

# 2. Seam Carving  
Upscale and downscale images, using Seam Carving method.  
![Paris upscaling](/images/2/paris_resize.gif "Paris upscaling")  
