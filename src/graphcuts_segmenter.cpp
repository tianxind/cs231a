#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <graphcuts/typedefs.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define DEPG_SIGMA (getenv("DEPG_SIGMA") ? atof(getenv("DEPG_SIGMA")) : 0.15)
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

pcl::KdTreeFLANN<pcl::PointXYZ>* buildForegroundKdTree(Scene& seed_frame) {
  vector<TrackedObject>& seed_objects = seed_frame.segmentation_->tracked_objects_;
  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_tree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
  pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  int num_points = 0;
  // if (seed_objects.size() > 1)
  //   num_points += seed_objects[1].indices_.size();
  // else
  //num_points += seed_objects[0].indices_.size();
  for (size_t i = 0; i < seed_objects.size(); ++i) {
    num_points += seed_objects[i].indices_.size();
  }
  fg_cloud->width = num_points;
  fg_cloud->height = 1;
  fg_cloud->points.resize(fg_cloud->width * fg_cloud->height);
  int cloud_index = 0;
  //vector<int>& indices = seed_objects[0].indices_;
  // if (seed_objects.size() > 1)
  //  indices = seed_objects[1].indices_;
  // else 
  //  indices = seed_objects[0].indices_;
  /*for (size_t j = 0; j < indices.size(); ++j) {
    fg_cloud->points[cloud_index].x = seed_frame.cloud_cam_(indices[j], 0); 
    fg_cloud->points[cloud_index].y = seed_frame.cloud_cam_(indices[j], 1);
    fg_cloud->points[cloud_index].z = seed_frame.cloud_cam_(indices[j], 2);  
  }*/
  for (size_t i = 0; i < seed_objects.size(); ++i) {
    vector<int>& indices = seed_objects[i].indices_;
    for (size_t j = 0; j < indices.size(); ++j) {
      fg_cloud->points[cloud_index].x = seed_frame.cloud_cam_(indices[j], 0); 
      fg_cloud->points[cloud_index].y = seed_frame.cloud_cam_(indices[j], 1);
      fg_cloud->points[cloud_index].z = seed_frame.cloud_cam_(indices[j], 2);  
      cloud_index++;
    }
  }
  cout<<"foreground cloud size: "<< fg_cloud->width<< " "<<cloud_index<<endl; 
  if(fg_cloud->width == 0)
  {
    return NULL;
  }

  fg_tree->setInputCloud(fg_cloud);
  return fg_tree;
}

pcl::KdTreeFLANN<pcl::PointXYZ>* buildKdTree(Eigen::MatrixXf& cloud_cam_) {
  int num_points = cloud_cam_.rows();
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
  /*cout << "Searching nearest neighbor for point ("
       << search_point.x << ", "
       << search_point.y << ", "
       << search_point.z << ") " << endl;*/
 
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
  // cout << "Found " << neighbors.size() << " neighbors for point ("
  //     << search_point.x << ", "
  //     << search_point.y << ", "
  //     << search_point.z << ") " << endl;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    if (index != neighbors[i]) {
      graph->add_edge(index, neighbors[i], exp(-distances[i]), exp(-distances[i]));
      // cout << "Edge weight is " << exp(-distances[i]);
    }
  }
}

void generateSegmentationFromGraph(graphcuts::Graph3dPtr graph,
                                   Scene& target_frame) {
  TrackedObject to;
  int foreground_num = 0;
  for (int i = 0; i < target_frame.cloud_cam_.rows(); ++i) {
    if (graph->what_segment(i, Graph<double, double, double>::SINK) ==
        Graph<double, double, double>::SOURCE) {
      to.indices_.push_back(i);
      foreground_num++;
      //cout << "Found foreground!!!!";
    }
  }
  target_frame.addTrackedObject(to);
  cout << "About to save segmentation..."<< foreground_num  << endl;
  target_frame.saveSegmentation();
}

double sinkPotential(double dist, double sigma) {
  return 1.0 - exp(-dist/sigma);
}

double sourcePotential(double dist, double sigma) {
  return exp(-dist/sigma);
}

void graphCutsSegmentation(pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree,
                           Scene& target_frame) {
  Eigen::MatrixXf cloud_cam_ = target_frame.cloud_cam_;
  int num_nodes = cloud_cam_.rows();
  graphcuts::Graph3dPtr graph_ = 
    graphcuts::Graph3dPtr(new graphcuts::Graph3d(num_nodes, cloud_cam_.rows() * 10));
  // Add nodes and node potentials
  cout << "Adding nodes to graph..." << endl;
  graph_->add_node(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    double dist = distToPrevForeground(cloud_cam_, i, fg_kdtree);
    double src_pot = sourcePotential(dist, DEPG_SIGMA);
    double snk_pot = sinkPotential(dist, DEPG_SIGMA);
    graph_->add_tweights(i, src_pot, snk_pot);
    //cout << "Node dist: " << dist << " (source) " << src_pot << " (sink) " << snk_pot << endl;
  }
  // Add edges and edge potentials
  // First construct kdtree for current image
  cout << "Build kdtree for current image..." << endl;
  pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree = buildKdTree(cloud_cam_);
  cout << "Adding edges and edge potentials..." << endl; 
  for (int i = 0; i < num_nodes; ++i) {
    addEdgesWithinRadius(cloud_cam_, i, 0.15, kdtree, graph_);
  }
  cout << "Running max flow..." << endl;
  graph_->maxflow();
  cout << "Finished running max flow..." << endl;
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
  cout << "Construct seed frame kdtree..." << endl;
  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree = buildForegroundKdTree(seed_frame);


  for (size_t i = 1; i < seq.size(); ++i) {
    Scene& target_frame = *seq.getScene(i);
    cout << "Tracking image in frame " << i << endl;
    graphCutsSegmentation(fg_kdtree, target_frame);
    fg_kdtree = buildForegroundKdTree(target_frame);
    // quit the program if there is no segmentation
    if(fg_kdtree==NULL)break;
    // Find the closest object each segmented point belongs to 
  }
  return 0;
}
