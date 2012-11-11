//#include <opencv2/imgproc/imgproc.hpp>

#include <segmentation_and_tracking/scene.h>
#include <utility>

using namespace std;
namespace bfs = boost::filesystem;

string usageString()
{
  ostringstream oss;
  oss << "Usage: baseline_segmenter DATA_DIR" << endl;
  return oss.str();
}

void findRGBRange(TrackedObject& obj,
		  Scene& sc,
		  vector<pair<cv::Vec3b, cv::Vec3b> >* RGBRange) {
  cv::Vec3b min_rgb(255,255,255);
  cv::Vec3b max_rgb(0,0,0);
  for (size_t j = 0; j < obj.indices_.size(); ++j) {
    int index = obj.indices_[j];
    cv::Point point(sc.cam_points_(index, 0), sc.cam_points_(index, 1));
    cv::Vec3b point_color = sc.img_.at<cv::Vec3b>(point);
    /*cout << " (" << (int)point_color[0] << ", "
	 << (int)point_color[1] << ", "
	 << (int)point_color[2] << ")";*/
    if (point_color[0] < min_rgb[0])
      min_rgb[0] = point_color[0];
    if (point_color[1] < min_rgb[1])
      min_rgb[1] = point_color[1];
    if (point_color[2] < min_rgb[2])
      min_rgb[2] = point_color[2];
    if (point_color[0] > max_rgb[0])
      max_rgb[0] = point_color[0];
    if (point_color[1] > max_rgb[1])
      max_rgb[1] = point_color[1];
    if (point_color[2] > max_rgb[2])
      max_rgb[2] = point_color[2];
  }
  RGBRange->push_back(make_pair(min_rgb, max_rgb));
  // Debug
  cout << "Max and Min for object " << obj.id_ << endl;
  cout << "Max RGB: (" << (int)max_rgb[0] << ", "
       << (int)max_rgb[1] << ", "
       << (int)max_rgb[2] << ")" << endl;
  cout << "Min RGB: (" << (int)min_rgb[0] << ", "
       << (int)min_rgb[1] << ", "
       << (int)min_rgb[2] << ")" << endl;
}

void findDepthRange(TrackedObject& obj,
		    scene& sc,
		    vector<pair<double, double> >* DepthRange) {
}

bool inRGBRange(cv::Vec3b& color, cv::Vec3b& min, cv::Vec3b& max) {
  return color[0] <= max[0] && color[0] >= min[0]
    && color[1] <= max[1] && color[1] >= min[1]
    && color[2] <= max[2] && color[2] >= min[2];
}

void segmentObjects(Scene& sc, vector<pair<cv::Vec3b, cv::Vec3b> >& RGBRange) {
  vector<TrackedObject> tracked_objects(RGBRange.size());
  Eigen::MatrixXi cam_points = sc.cam_points_;
  for (size_t i = 0; i < cam_points.rows(); ++i) {
    cv::Point point(cam_points(i, 0), cam_points(i, 1));
    cv::Vec3b point_color = sc.img_.at<cv::Vec3b>(point);
    for (size_t j = 0; j < RGBRange.size(); ++j) {
      if (inRGBRange(point_color, RGBRange[j].first, RGBRange[j].second)) {
	  tracked_objects[j].indices_.push_back(i);
	  /*cout << "Adding point (" << cam_points(i,0) << ", " << cam_points(i,1)
	    <<") to object " << j << endl;*/
      }
    }
  }
  for (size_t i = 0; i < tracked_objects.size(); ++i) {
    sc.addTrackedObject(tracked_objects[i]);
  }
  sc.saveSegmentation();
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
  cv::Mat overlay = seed_frame.getDepthOverlay();
  vector<TrackedObject>& seed_objects = seed_frame.segmentation_->tracked_objects_; 
  vector<pair<cv::Vec3b, cv::Vec3b> > RGBRange;
  vector<pair<double, double> > DepthRange;
  for (size_t i = 0; i < seed_objects.size(); ++i) {
    findRGBRange(seed_objects[i], seed_frame, &RGBRange);
    //findDepthRange(seed_objects[i]);
  }
  for(size_t i = 1; i < seq.size(); ++i) {
    Scene& sc = *seq.getScene(i);
    cout << "Segmenting objects for scene " << i << "..." << endl;
    segmentObjects(sc, RGBRange);
    // cv::Mat overlay = sc.getDepthOverlay();
    //cv::imshow("test", overlay);
    //cv::waitKey(0);
  }
  return 0;
}
