#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
//#include <boost/shared_ptr.hpp>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
//#include <maxflow/graph.h>
#include <graphcuts/typedefs.h>
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

void saveTrackedObject(cv::RotatedRect& bounding_box,
                        Scene& sc) {
  /*  Eigen::MatrixXi cam_points = sc.cam_points_;
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
  sc.addTrackedObject(to);*/
}

pcl::KdTreeFLANN<pcl::PointXYZ>* buildForegroundKdTree(Scene& seed_frame) {
  vector<TrackedObject>& seed_objects = seed_frame.segmentation_->tracked_objects_;
  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_tree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  int num_points = 0;
  for (size_t i = 0; i < seed_objects.size(); ++i) {
    num_points += seed_objects[i].indices_.size();
  }
  fg_cloud->width = num_points;
  fg_cloud->height = 1;
  fg_cloud->points.resize(fg_cloud->width * fg_cloud->height);
  int cloud_index = 0;
  for (size_t i = 0; i < seed_objects.size(); ++i) {
    vector<int>& indices = seed_objects[i].indices_;
    for (size_t j = 0; j < indices.size(); ++j) {
      fg_cloud->points[cloud_index].x = seed_frame.cloud_cam_(indices[j], 0); 
      fg_cloud->points[cloud_index].y = seed_frame.cloud_cam_(indices[j], 1);
      fg_cloud->points[cloud_index].z = seed_frame.cloud_cam_(indices[j], 2);  
    }
  }
  fg_tree->setInputCloud(fg_cloud);
  return fg_tree;
}

pcl::KdTreeFLANN<pcl::PointXYZ>* buildKdTree(Eigen::MatrixXf& cloud_cam_) {
  int num_points = cloud_cam_.size();
  pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width = num_points;
  cloud->height = 1;
  cloud->points.resize(cloud->width * cloud->height);
  for (int i = 0; i < num_points; ++i) {
    cloud->points[i].x = cloud_cam_(i, 0); 
    cloud->points[i].y = cloud_cam_(i, 1);
    cloud->points[i].z = cloud_cam_(i, 2);  
  }
  kdtree->setInputCloud(cloud);
  return kdtree;
}

double distToPrevForeground(Eigen::MatrixXf& cloud_cam_,
                            int index,
                            pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree) {
  pcl::PointXYZ search_point;
  search_point.x = cloud_cam_(index, 0);
  search_point.y = cloud_cam_(index, 1);
  search_point.z = cloud_cam_(index, 2);
  vector<int> nearest(1);
  vector<float> distance(1);
  fg_kdtree->nearestKSearch(search_point, 1, nearest, distance);
  return distance[0];
}

void addEdgesWithinRadius(Eigen::MatrixXf& cloud_cam_,
                          int index,
                          float radius,
                          pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree,
                          graphcuts::Graph3dPtr graph) {
  pcl::PointXYZ search_point;
  search_point.x = cloud_cam_(index, 0);
  search_point.y = cloud_cam_(index, 1);
  search_point.z = cloud_cam_(index, 2);
  vector<int> neighbors;
  vector<float> distances;
  kdtree->radiusSearch(search_point, radius, neighbors, distances);
  cout << "Found " << neighbors.size() << " neighbors for point ("
       << search_point.x << ", "
       << search_point.y << ", "
       << search_point.z << ") " << endl;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    graph->add_edge(index, neighbors[i], exp(-distances[i]), exp(-distances[i]));
  }
}

void generateSegmentationFromGraph(graphcuts::Graph3dPtr graph,
                                   Scene& target_frame) {
  TrackedObject to;
  for (int i = 0; i < target_frame.cloud_cam_.size(); ++i) {
    if (graph->what_segment(i, Graph<double, double, double>::SINK) ==
        Graph<double, double, double>::SOURCE) {
      to.indices_.push_back(i);
    }
  }
  target_frame.addTrackedObject(to);
  target_frame.saveSegmentation();
}

void graphCutsSegmentation(pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree,
                           Scene& target_frame) {
  Eigen::MatrixXf cloud_cam_ = target_frame.cloud_cam_;
  int num_nodes = cloud_cam_.size();
  graphcuts::Graph3dPtr graph_ = 
    graphcuts::Graph3dPtr(new graphcuts::Graph3d(num_nodes, cloud_cam_.size() * 10));
  // Add nodes and node potentials
  graph_->add_node(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    double dist = distToPrevForeground(cloud_cam_, i, fg_kdtree);
    graph_->add_tweights(i, exp(-dist), exp(dist));
  }
  // Add edges and edge potentials
  // First construct kdtree for current image
  pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree = buildKdTree(cloud_cam_); 
  for (int i = 0; i < num_nodes; ++i) {
    addEdgesWithinRadius(cloud_cam_, i, 100, kdtree, graph_);
  }
  graph_->maxflow();
  generateSegmentationFromGraph(graph_, target_frame);
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
  // Construct seed segmentation KdTree
  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree = buildForegroundKdTree(seed_frame);


  for (size_t i = 1; i < seq.size(); ++i) {
    Scene& target_frame = *seq.getScene(i);
    cout << "Tracking image in frame " << i << endl;
    graphCutsSegmentation(fg_kdtree, target_frame);
    fg_kdtree = buildForegroundKdTree(target_frame);
    // Find the closest object each segmented point belongs to 
    // saveTrackedObject(tracked_box, target_frame);
    // target_frame.saveSegmentation();
    //cv::imwrite("/home/sandra/cs231a/target.jpg", target_frame.img_);
    //cv::imwrite("/home/sandra/cs231a/seed.jpg", seed_frame.img_);
  }
  return 0;
}
