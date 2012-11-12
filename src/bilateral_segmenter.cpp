#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <graphcuts/typedefs.h>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>

#define DEPG_SIGMA (getenv("DEPG_SIGMA") ? atof(getenv("DEPG_SIGMA")) : 0.0625)
#define BILATERAL_W (getenv("BILATERAL_W") ? atof(getenv("BILATERAL_W")) : 1)
#define BILATERAL_SIGMA (getenv("BILATERAL_SIGMA") ? atof(getenv("BILATERAL_SIGMA")) : 0.0625)
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

pcl::KdTreeFLANN<pcl::PointXYZ>* buildForegroundKdTree(Scene& seed_frame, pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud) {
  vector<TrackedObject>& seed_objects = seed_frame.segmentation_->tracked_objects_;
  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_tree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
  
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
      fg_cloud->points[cloud_index].x = seed_frame.cloud_smooth_(indices[j], 0); 
      fg_cloud->points[cloud_index].y = seed_frame.cloud_smooth_(indices[j], 1);
      fg_cloud->points[cloud_index].z = seed_frame.cloud_smooth_(indices[j], 2);  
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

pcl::KdTreeFLANN<pcl::PointXYZ>* buildKdTree(Eigen::MatrixXf& cloud_smooth_, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  int num_points = cloud_smooth_.rows();
  pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree = new pcl::KdTreeFLANN<pcl::PointXYZ>();
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  //cloud = (new pcl::PointCloud<pcl::PointXYZ>);
  cloud->width = num_points;
  cloud->height = 1;
  cloud->points.resize((cloud->width) * (cloud->height));
  for (int i = 0; i < num_points; ++i) {
    cloud->points[i].x = cloud_smooth_(i, 0); 
    cloud->points[i].y = cloud_smooth_(i, 1);
    cloud->points[i].z = cloud_smooth_(i, 2);  
  }
  kdtree->setInputCloud(cloud);
  //whole_cloud = cloud;
  return kdtree;
}

double distToPrevForeground(pcl::PointXYZ& node,
                            pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree) {
  vector<int> nearest(1);
  vector<float> distance(1);
  fg_kdtree->nearestKSearch(node, 1, nearest, distance);
  return distance[0];
}

void addEdgesWithinRadius(Eigen::MatrixXf& cloud_smooth_,
                          int index,
                          float radius,
                          pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree,
                          graphcuts::Graph3dPtr graph) {
  pcl::PointXYZ search_point;
  search_point.x = cloud_smooth_(index, 0);
  search_point.y = cloud_smooth_(index, 1);
  search_point.z = cloud_smooth_(index, 2);
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
  for (int i = 0; i < target_frame.cloud_smooth_.rows(); ++i) {
    if (graph->what_segment(i, Graph<double, double, double>::SINK) ==
        Graph<double, double, double>::SOURCE) {
      to.indices_.push_back(i);
      foreground_num++;
    }
  }
  target_frame.addTrackedObject(to);
  cout << "About to save segmentation..."<< foreground_num  << endl;
  target_frame.saveSegmentation();
}

void addDistToForegroundPotential(pcl::PointXYZ& node,
                                  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree,
                                  vector<double>& src_potentials,
                                  vector<double>& snk_potentials) {
  double dist = distToPrevForeground(node, fg_kdtree);
  src_potentials.push_back(exp(-dist/DEPG_SIGMA));
  snk_potentials.push_back(1.0 - exp(-dist/DEPG_SIGMA));
}

void aggregatePotential(int node_index,
                        vector<double>& src_potentials,
                        vector<double>& snk_potentials,
                        graphcuts::Graph3dPtr graph_) {
  double src_pot = src_potentials[0] + BILATERAL_W * src_potentials[1];
  double snk_pot = snk_potentials[0] + BILATERAL_W * snk_potentials[1];
  graph_->add_tweights(node_index, src_pot, snk_pot);
}

bool isForeground(pcl::PointXYZ& node,
                  int neighbor_id,
                  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud) {
  vector<int> nearest(1);
  vector<float> distance(1);
  //cout<<"###### isFoureground ing ";
  //cout<< node.x<<endl;
  fg_kdtree->nearestKSearch(node, 1, nearest, distance);
  bool isForeground = true;
  pcl::PointXYZ& nearest_node = fg_cloud->points[nearest[0]];
  if(abs(nearest_node.x - node.x) > 1E-6) isForeground = false;
  if(abs(nearest_node.y - node.y) > 1E-6) isForeground = false;
  if(abs(nearest_node.z - node.z) > 1E-6) isForeground = false;
  return isForeground;

  //return nearest[0]==neighbor_id;
}

void addBilateralPotential(pcl::PointXYZ& node,
                           pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree,
                           pcl::KdTreeFLANN<pcl::PointXYZ>* whole_kdtree,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud,
                           double radius,
                           vector<double>& src_potentials,
                           vector<double>& snk_potentials) {
  // TODO: implement this
  vector<int> neighbors;
  vector<float> distances;
  whole_kdtree->radiusSearch(node, radius, neighbors, distances);
  // cout << "Found " << neighbors.size() << " neighbors for point ("
  //     << search_point.x << ", "
  //     << search_point.y << ", "
  //     << search_point.z << ") " << endl;
  double sum_terms = 0.0;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    int label = 0;
    int neighbor_id = neighbors[i];
    pcl::PointXYZ& node = whole_cloud->points[neighbors[i]];
    if (isForeground(node, neighbor_id, fg_kdtree, fg_cloud)) {
      label = 1;
    } else {
      label = -1;
    }
    sum_terms = label * exp(-sqrt(distances[i]/BILATERAL_SIGMA));
  }
  double energy = 2.0 / (1.0 + exp(-sum_terms)) - 1.0;
  if (energy > 0) {
    src_potentials.push_back(energy);
    snk_potentials.push_back(0);
  } else {
    src_potentials.push_back(0);
    snk_potentials.push_back(-energy);
  }
}

void graphCutsSegmentation(pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree,
                           pcl::KdTreeFLANN<pcl::PointXYZ>* whole_kdtree,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud,
                           Scene& target_frame) {
  Eigen::MatrixXf cloud_smooth_ = target_frame.cloud_smooth_;
  int num_nodes = cloud_smooth_.rows();
  graphcuts::Graph3dPtr graph_ = 
    graphcuts::Graph3dPtr(new graphcuts::Graph3d(num_nodes, cloud_smooth_.rows() * 10));
  // Add nodes and node potentials
  cout << "Adding nodes to graph..." << endl;
  graph_->add_node(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    pcl::PointXYZ node;
    node.x = cloud_smooth_(i, 0);
    node.y = cloud_smooth_(i, 1);
    node.z = cloud_smooth_(i, 2);
    vector<double> src_potentials, snk_potentials;
    // Add node potential computed from distance to previous foreground
    addDistToForegroundPotential(node, fg_kdtree, src_potentials, snk_potentials);

    addBilateralPotential(node, fg_kdtree, whole_kdtree, fg_cloud, whole_cloud, 0.15, src_potentials, snk_potentials);
    // Aggregate two potentials using predefined weights, and add the potential to graph
    aggregatePotential(i, src_potentials, snk_potentials, graph_);
    //cout << "Node dist: " << dist << " (source) " << src_pot << " (sink) " << snk_pot << endl;
  }
  // Add edges and edge potentials
  // First construct kdtree for current image
  cout << "Build kdtree for current image..." << endl;
  cout << "Adding edges and edge potentials..." << endl; 
  //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ>* kdtree = buildKdTree(target_frame.cloud_smooth_, whole_cloud);
  for (int i = 0; i < num_nodes; ++i) {
    addEdgesWithinRadius(cloud_smooth_, i, 0.15, kdtree, graph_);
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

  pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud (new pcl::PointCloud<pcl::PointXYZ>);

  pcl::KdTreeFLANN<pcl::PointXYZ>* fg_kdtree = buildForegroundKdTree(seed_frame, fg_cloud);
  pcl::KdTreeFLANN<pcl::PointXYZ>* whole_kdtree = buildKdTree(seed_frame.cloud_smooth_, whole_cloud);

  for (size_t i = 1; i < seq.size(); ++i) {
    Scene& target_frame = *seq.getScene(i);
    cout << "Tracking image in frame " << i << endl;
    graphCutsSegmentation(fg_kdtree, whole_kdtree, fg_cloud, whole_cloud, target_frame);
    fg_kdtree = buildForegroundKdTree(target_frame, fg_cloud);
    // quit the program if there is no segmentation
    if(fg_kdtree == NULL)break;
    // Find the closest object each segmented point belongs to 
  }
  return 0;
}
