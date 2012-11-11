#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
namespace bfs = boost::filesystem;

namespace {
  static const int INF = 9999999;
  static const int DEBUG = 2;
  static const int vmin = 10, vmax = 256, smin = 30;
}

string usageString()
{
  ostringstream oss;
  oss << "Usage: baseline_segmenter DATA_DIR" << endl;
  return oss.str();
}

void drawBoxOnScene(Scene& sc, cv::Rect& bounding_box)
{
    rectangle(sc.img_,
	      cv::Point(bounding_box.x, bounding_box.y),
	      cv::Point(bounding_box.x + bounding_box.width,
			bounding_box.y + bounding_box.height),
	      cv::Scalar(0, 0, 255));

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

cv::Mat extractHueChannel(cv::Mat roi, cv::Mat* mask) {
  cv::Mat hsv, hue;
  cv::cvtColor(roi, hsv, CV_BGR2HSV); 
  // Filter out pixels that are too dark or too bright, the pixel in mask will be
  // 255 if the original pixel is in range, and 0 otherwise.
  cv::inRange(hsv, cv::Scalar(0, smin, MIN(vmin, vmax)),
	      cv::Scalar(180, 256, MAX(vmin, vmax)), *mask);
  int ch[] = {0, 0};
  // Extract only the hue channel from hsv
  hue.create(hsv.size(), hsv.depth()); 
  cv::mixChannels(&hsv, 1, &hue, 1, ch, 1);
  return hue;
}

cv::RotatedRect camShiftTracking(cv::Rect original_box,
                                 cv::Mat original_image,
                                 cv::Mat target_hue_channel,
                                 cv::Mat target_mask) {
  int hsize = 16;
  // Set hue range at [0, 180]
  float hranges[] = {0, 180}; 
  const float* phranges = hranges;
  cv::RotatedRect track_rotated_box; 
  // Upper and lower bound for value and saturation in hsv scale
  cv::Mat hsv, hist, hue, backproj;
  
  // Extract the pixels of the tracked object in the original image
  cv::Mat mask_roi(original_image, original_box);
  cv::Mat roi(original_image, original_box); 
  hue = extractHueChannel(roi, &mask_roi);
  // Calculate histogram of the tracked object in original image
  calcHist(&hue, 1, 0, mask_roi, hist, 1, &hsize, &phranges); 
  cv::normalize(hist, hist, 0, 255, CV_MINMAX); 
  // Calculate back projection of the histogram of the tracked object in target frame
  cv::calcBackProject(&target_hue_channel, 1, 0, hist, backproj, &phranges);  
  backproj &= target_mask;
  
  track_rotated_box = cv::CamShift(backproj, original_box,             
                                   cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1)); 
 
  cv::Rect track_box = track_rotated_box.boundingRect();
  if (DEBUG==2) printf("\n## originRect x,y,width,height %d %d %d %d; targetRect x,y,width,height %d %d %d %d\n", 
                       original_box.x, original_box.y, original_box.width, original_box.height, 
                       track_box.x, track_box.y, track_box.width, track_box.height);
  return track_rotated_box;
} // end of CamShiftTracking

void saveTrackedObject(cv::RotatedRect& bounding_box,
                        Scene& sc) {
  Eigen::MatrixXi cam_points = sc.cam_points_;
  cv::Mat rot_mat = getRotationMatrix2D(cv::Point(0, 0),
                                        -bounding_box.angle,
                                        1);
  //cout << rot_mat << endl;
  TrackedObject to;
  int width = bounding_box.size.width;
  int height = bounding_box.size.height;
  for (size_t i = 0; i < cam_points.rows(); ++i) {
    cv::Mat pm = (cv::Mat_<double>(3, 1) << cam_points(i, 0) - bounding_box.center.x,
                  cam_points(i, 1) - bounding_box.center.y, 1);
    // cout << pm;
    cv::Mat rotated_point = rot_mat * pm;
    if (rotated_point.at<double>(0,0) > -0.5*width && rotated_point.at<double>(0,0) < 0.5*width
                                                                                      && rotated_point.at<double>(0,1) > -0.5*height && rotated_point.at<double>(0,1) < 0.5*height) {
      to.indices_.push_back(i);
    } 
  }
  sc.addTrackedObject(to);
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
  Sequence seq(dirpath);
  // Read in seed frame data
  Scene& seed_frame = *seq.getScene(0);
  vector<TrackedObject>& seed_objects = seed_frame.segmentation_->tracked_objects_;


  for (size_t i = 1; i < seq.size(); ++i) {
    Scene& target_frame = *seq.getScene(i);
    cout << "Tracking image in frame " << i << endl;
    for (size_t j = 0; j < seed_objects.size(); j++) {
      cv::Rect original_box = findBoundingBox(seed_objects[j], seed_frame);
      //drawBoxOnScene(seed_frame, original_box);
      cv::Mat target_mask;
      cv::Mat target_hue_channel = extractHueChannel(target_frame.img_, &target_mask);
      cv::RotatedRect tracked_box = camShiftTracking(original_box,
                                                     seed_frame.img_,
                                                     target_hue_channel,
                                                     target_mask);
      cout << "after calling camshift in frame " << i 
           << "for object " << j << endl;
      saveTrackedObject(tracked_box, target_frame);
      //drawBoxOnScene(target_frame, tracked_box);
    }
    target_frame.saveSegmentation();
    //cv::imwrite("/home/sandra/cs231a/target.jpg", target_frame.img_);
    //cv::imwrite("/home/sandra/cs231a/seed.jpg", seed_frame.img_);
  }
  return 0;
}
