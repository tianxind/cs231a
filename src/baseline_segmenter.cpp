//#include <opencv2/imgproc/imgproc.hpp>

#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
namespace bfs = boost::filesystem;

static const int INF = 9999999; 
static const int DEBUG = 2;
string usageString()
{
  ostringstream oss;
  oss << "Usage: baseline_segmenter DATA_DIR" << endl;
  return oss.str();
}

void drawBoxOnScene(Scene& sc, cv::Rect& bounding_box);

void segmentObjects(Scene& sc, vector<cv::Mat>& histograms) {
  // Convert the scene to hsv format and extract the hue channel.
  // Calculate back projection using each histogram in 'histograms'
  // vector
  // Use camShift to determine rectangle of the tracked objects
  // Draw rectangle on the image  
  /*  vector<TrackedObject> tracked_objects(RGBRange.size());
  Eigen::MatrixXi cam_points = sc.cam_points_;
  for (size_t i = 0; i < cam_points.rows(); ++i) {
    cv::Point point(cam_points(i, 0), cam_points(i, 1));
    cv::Vec3b point_color = sc.img_.at<cv::Vec3b>(point);
    for (size_t j = 0; j < RGBRange.size(); ++j) {
      if (inRGBRange(point_color, RGBRange[j].first, RGBRange[j].second)) {
	  tracked_objects[j].indices_.push_back(i);
	  /*cout << "Adding point (" << cam_points(i,0) << ", " << cam_points(i,1)
	    <<") to object " << j << endl;
      }
    }
  }
  for (size_t i = 0; i < tracked_objects.size(); ++i) {
    sc.addTrackedObject(tracked_objects[i]);
  }
  sc.saveSegmentation();*/
}

cv::Rect findBoundingBox(TrackedObject& object,
			 Scene& sc) {
  object.generateImageCoords(sc);
  int min_x = INF, min_y = INF;
  int max_x = -1, max_y = -1;
  vector<cv::Point>& image_coords = object.image_coords_;
  cout << "image_coords size: " << image_coords.size();
  for (size_t i = 0; i < image_coords.size(); ++i) {
    if(DEBUG==1)cout << "(" << image_coords[i].x << "," << image_coords[i].y
	 << ")" << " ";
    if (image_coords[i].x > max_x) {
      max_x = image_coords[i].x;
    }
    if (image_coords[i].x < min_x) {
      min_x = image_coords[i].x;
    }
    if (image_coords[i].y > max_y) {
      max_y = image_coords[i].y;
    }
    if (image_coords[i].y < min_y) {
      min_y = image_coords[i].y;
    }
  }
  return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}

cv::Rect camShiftTracking(cv::Rect originalBox, cv::Mat originalImage, Scene& sc)
{
	//tracking algorithms camshift
	cv::Rect trackBox = originalBox;
	 
	int hsize = 16;
	float hranges[] = {0,180}; 
	const float* phranges = hranges;
	cv::Rect trackWindow = originalBox; // start from the previous tracking windows
	cv::RotatedRect trackRotatedBox; 
	int vmin = 10, vmax = 256, smin = 30;
	cv::Mat frame, hsv, hue, mask, hist = cv::Mat::zeros(originalImage.cols, originalImage.rows, CV_8UC3), backproj;


	cv::cvtColor(originalImage, hsv, CV_BGR2HSV);// from rgb to hsv

	cv::inRange(hsv, cv::Scalar(0, smin, MIN(vmin,vmax)),
		     cv::Scalar(180, 256, MAX(vmin, vmax)), mask);
	int ch[] = {0, 0};
	hue.create(hsv.size(), hsv.depth()); 
	cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);// copy h from hsv to hue

	cv::Mat roi(hue, originalBox); 
	calcHist(&roi, 1, 0, cv::Mat(), hist, 1, &hsize, &phranges); 
	normalize(hist, hist, 0, 255, CV_MINMAX); 

	cv::calcBackProject(&hue, 1, 0, hist, backproj, &phranges);  
	//backproj &= mask;// do
	 
 
	trackRotatedBox = cv::CamShift(backproj, trackWindow,             
				  cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 )); 
 
	trackBox = trackRotatedBox.boundingRect();
	if(DEBUG==2)printf("\n## originRect x,y,width,height %d %d %d %d; targetRect x,y,width,height %d %d %d %d\n", 
			originalBox.x, originalBox.y, originalBox.width, originalBox.height, 
			trackBox.x, trackBox.y, trackBox.width, trackBox.height);
	return trackBox;

}// end of CamShiftTracking

void calcObjectHistograms(vector<TrackedObject>& objects,
			  Scene& sc,
			  vector<cv::Mat>* histograms) {
  for (size_t i = 0; i < objects.size(); ++i) {
    cv::Rect bounding_box = findBoundingBox(objects[i], sc);
    //drawBoxOnScene(sc, bounding_box);
    if(DEBUG==1)
    cout << "bounding box for object " << i
	 << "(" << bounding_box.x << ", " << bounding_box.y << ")"
	 << "(" << bounding_box.x + bounding_box.width 
	 << ", " << bounding_box.y + bounding_box.height << ")" << endl;
   
  }

}

void drawBoxOnScene(Scene& sc, cv::Rect& bounding_box)
{
    rectangle(sc.img_,
	      cv::Point(bounding_box.x, bounding_box.y),
	      cv::Point(bounding_box.x + bounding_box.width,
			bounding_box.y + bounding_box.height),
	      cv::Scalar(0, 0, 255));

}


int main(int argc, char** argv)
{
  if(argc != 2) { 
    cout << usageString();
    return 1;
  }

  string dirpath = argv[1];
  
  if(!bfs::exists(dirpath)) {
    cout << dirpath << " does not exist." << endl;
    return 1;
  }
  //  cv::Mat img1 = cv::imread("/home/sandra/cs231a/sequence01/1288572831.002852.jpg");
  // cv::namedWindow("test", CV_WINDOW_AUTOSIZE);
  Sequence seq(dirpath);
  // Read in seed frame data
  Scene& seed_frame = *seq.getScene(0);
  vector<TrackedObject>& seed_objects = seed_frame.segmentation_->tracked_objects_;
  // Converts the seed image to hsv format and extract hue channel

  // Calculate histograms for each tracked object in seed frame
  vector<cv::Mat> histograms;
  calcObjectHistograms(seed_objects, seed_frame, &histograms);

 


  size_t i = 10;

  Scene& targetframe = *seq.getScene(i);

  for(int j=0; j< seed_objects.size(); j++)
  {
    cv::Rect originalBox = findBoundingBox(seed_objects[j], seed_frame);
    cv::Rect trackedBox = camShiftTracking(originalBox, seed_frame.img_, targetframe);
    drawBoxOnScene(targetframe, trackedBox);
  }



  cv::imwrite("/home/sandra/cs231a/test.jpg", targetframe.img_);
  //double scale = 0.5;
  //if(getenv("SCALE"))
  //  scale = atof(getenv("SCALE"));
  //OpenCVView view("Image", scale);
  //HandSegmenterViewController vc(&view, dirpath);
  //view.setDelegate(&vc);
  //vc.run();

  /*  for(size_t i = 1; i < seq.size(); ++i) {
    Scene& sc = *seq.getScene(i);
    cout << "Segmenting objects for scene " << i << "..." << endl;
    segmentObjects(sc, RGBRange);
    //cv::Mat overlay = sc.getDepthOverlay();
    //cv::imshow("test", overlay);
    //cv::waitKey(0);
    }*/
  return 0;
}
