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
#define DEPG_W (getenv("DEPG_W") ? atof(getenv("DEPG_W")) : 1)
#define EDGE_SIGMA (getenv("EDGE_SIGMA") ? atof(getenv("EDGE_SIGMA")) : 1)
#define COLOR_SIGMA (getenv("COLOR_SIGMA") ? atof(getenv("COLOR_SIGMA")) : 30.0)
#define DIST_W (getenv("DIST_W") ? atof(getenv("DIST_W")) : 1)
#define COLOR_W (getenv("COLOR_W") ? atof(getenv("COLOR_W")) : 1)
#define BILATERAL_SIGMA (getenv("BILATERAL_SIGMA") ? atof(getenv("BILATERAL_SIGMA")) : 0.05)
#define DEBUG (getenv("DEBUG") ? atof(getenv("DEBUG")) : 0) 
#define RADIUS (getenv("RADIUS") ? atof(getenv("RADIUS")) : 0.15)

using namespace std;
namespace bfs = boost::filesystem;

// typedef scoped_ptr<pcl::KdTreeFLANN<pcl::PointXYZ> > KdTreePtr;

namespace {
  static const int INF = 9999999;
  static const int vmin = 10, vmax = 256, smin = 30;
}

string usageString()
{
  ostringstream oss;
  oss << "Usage: baseline_segmenter DATA_DIR" << endl;
  return oss.str();
}

// Builds foreground kdtree for one tracked object
void
buildForegroundKdTree(Scene& seed_frame,
                      TrackedObject& to, 
                      pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                      pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_tree) {
  
  fg_cloud->width = to.indices_.size();
  fg_cloud->height = 1;
  fg_cloud->points.resize(fg_cloud->width * fg_cloud->height);
  for (size_t i = 0; i < to.indices_.size(); ++i) {
    fg_cloud->points[i].x = seed_frame.cloud_smooth_(to.indices_[i], 0); 
    fg_cloud->points[i].y = seed_frame.cloud_smooth_(to.indices_[i], 1);
    fg_cloud->points[i].z = seed_frame.cloud_smooth_(to.indices_[i], 2);      
  }

  // return empty tree
  if (fg_cloud->width == 0) {
  //  return NULL;
      return;
  }
  fg_tree->setInputCloud(fg_cloud);
}

void
buildKdTree(Eigen::MatrixXf& cloud_smooth_, 
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree) {
  int num_points = cloud_smooth_.rows();
  cloud->width = num_points;
  cloud->height = 1;
  cloud->points.resize((cloud->width) * (cloud->height));
  for (int i = 0; i < num_points; ++i) {
    cloud->points[i].x = cloud_smooth_(i, 0); 
    cloud->points[i].y = cloud_smooth_(i, 1);
    cloud->points[i].z = cloud_smooth_(i, 2);  
  }
  kdtree->setInputCloud(cloud);
}

double distToPrevForeground(pcl::PointXYZ& node,
                            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree) {
  vector<int> nearest(1);
  vector<float> distance(1);
  fg_kdtree->nearestKSearch(node, 1, nearest, distance);
  return sqrt(distance[0]);
}

double getColorDistance(int node_index,
                        int neighbor_index,
                        Eigen::MatrixXi& cam_points_,
                        cv::Mat& img) {
  cv::Point point(cam_points_(node_index, 0),
                  cam_points_(node_index, 1));
  cv::Point neighbor(cam_points_(neighbor_index, 0),
                  cam_points_(neighbor_index, 1));
  cv::Vec3b point_color = img.at<cv::Vec3b>(point);
  cv::Vec3b neighbor_color = img.at<cv::Vec3b>(neighbor);
  return sqrt((point_color[0] - neighbor_color[0]) * (point_color[0] - neighbor_color[0])
    + (point_color[1] - neighbor_color[1]) * (point_color[1] - neighbor_color[1])
              + (point_color[2] - neighbor_color[2]) * (point_color[2] - neighbor_color[2]));
}

void addEdgesWithinRadius(Eigen::MatrixXf& cloud_smooth_,
                          Eigen::MatrixXi& cam_points_,
                          cv::Mat& img,
                          int index,
                          float radius,
                          pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree,
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
      double color_potential = exp(-getColorDistance(index, neighbors[i], cam_points_, img)
                                   /COLOR_SIGMA);
      double dist_potential = exp(-sqrt(distances[i])/EDGE_SIGMA);
      double edge_potential = COLOR_W * color_potential + DIST_W * dist_potential; 
      graph->add_edge(index, neighbors[i], edge_potential, edge_potential);
    }
  }
}

void generateSegmentationFromGraph(graphcuts::Graph3dPtr graph,
                                   int index,
                                   Scene& target_frame) {
  TrackedObject to;
  to.id_ = index;
  int foreground_num = 0;
  for (int i = 0; i < target_frame.cloud_smooth_.rows(); ++i) {
    if (graph->what_segment(i, Graph<double, double, double>::SINK) ==
        Graph<double, double, double>::SOURCE) {
      to.indices_.push_back(i);
      foreground_num++;
    }
  }
  cout << "Points in foreground: " << foreground_num << endl; 
  target_frame.addTrackedObject(to);
}

void saveSequence(Sequence& seq) {
  for (size_t j = 1; j < seq.size(); ++j) {
    Scene& target_frame = *seq.getScene(j);
    if (target_frame.segmentation_) {
      cout << "About to save segmentation " << j << " ..." << endl;
      target_frame.saveSegmentation();
      // Save node potential for this scene to file
      target_frame.saveBilateralPotential();
      target_frame.saveDistToFgPotential();
    }
  }
}

void addDistToForegroundPotential(pcl::PointXYZ& node,
                                  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                                  vector<double>& src_potentials,
                                  vector<double>& snk_potentials,
                                  Scene& target_frame) {
  double dist = distToPrevForeground(node, fg_kdtree);
  double src_pot = exp(-dist/DEPG_SIGMA);
  src_potentials.push_back(src_pot);
  snk_potentials.push_back(1.0 - src_pot);
  target_frame.distToFg_potential_.potentials_.push_back(src_pot);
}

void aggregatePotential(int node_index,
                        pcl::PointXYZ& node,
                        vector<double>& src_potentials,
                        vector<double>& snk_potentials,
                        graphcuts::Graph3dPtr graph_) {
  double src_pot = DEPG_W * src_potentials[0] + BILATERAL_W * src_potentials[1];
  double snk_pot = DEPG_W * snk_potentials[0] + BILATERAL_W * snk_potentials[1];
  if (DEBUG == 1) {
    cout << "Node (" << node.x << ", " << node.y << ", " << node.z << "): srcdepth-" << src_potentials[0]
         << " srcbilateral-" << src_potentials[1] << " snkdepth-" << snk_potentials[0] << " snkbilateral-"
         << snk_potentials[1] << "    src: " << src_pot << " snk: " << snk_pot << endl;
  }
  graph_->add_tweights(node_index, src_pot, snk_pot);
}

bool isForeground(pcl::PointXYZ& node,
                  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                  pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud) {
  vector<int> nearest(1);
  vector<float> distance(1);
  fg_kdtree->nearestKSearch(node, 1, nearest, distance);
  bool isForeground = true;
  pcl::PointXYZ& nearest_node = fg_cloud->points[nearest[0]];
  if (abs(nearest_node.x - node.x) > 1E-6) isForeground = false;
  if (abs(nearest_node.y - node.y) > 1E-6) isForeground = false;
  if (abs(nearest_node.z - node.z) > 1E-6) isForeground = false;
  return isForeground;
}

void addBilateralPotential(pcl::PointXYZ& node,
                           pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                           pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr whole_kdtree,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud,
                           double radius,
                           vector<double>& src_potentials,
                           vector<double>& snk_potentials,
                           Scene& target_frame) {
  // TODO: implement this
  vector<int> neighbors;
  vector<float> distances;
  whole_kdtree->radiusSearch(node, radius, neighbors, distances);
  /*cout << "Found " << neighbors.size() << " neighbors for point ("
       << node.x << ", "
       << node.y << ", "
       << node.z << ") " << endl;*/
  double sum_terms = 0.0;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    int label = 0;
    pcl::PointXYZ& node = whole_cloud->points[neighbors[i]];
    if (isForeground(node, fg_kdtree, fg_cloud)) {
      label = 1;
    } else {
      label = -1;
    }
    sum_terms += label * exp(-sqrt(distances[i])/BILATERAL_SIGMA);
  }
  double energy = 2.0 / (1.0 + exp(-sum_terms)) - 1.0;
  /*cout << "Energy for point (" 
       << node.x << ", "
       << node.y << ", "
       << node.z << ") is " << energy << endl;*/
  // Save bilateral node potential to vector
  target_frame.bilateral_potential_.potentials_.push_back(energy);
  if (energy > 0) {
    src_potentials.push_back(energy);
    snk_potentials.push_back(0);
  } else {
    src_potentials.push_back(0);
    snk_potentials.push_back(-energy);
  }
}

void graphCutsSegmentation(pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                           pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr whole_kdtree,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud,
                           int index,
                           Scene& target_frame) {
  Eigen::MatrixXf cloud_smooth_ = target_frame.cloud_smooth_;
  int num_nodes = cloud_smooth_.rows();
  graphcuts::Graph3dPtr graph_ = 
    graphcuts::Graph3dPtr(new graphcuts::Graph3d(num_nodes, cloud_smooth_.rows() * 10));
  // Add nodes and node potentials
  cout << "Adding nodes to graph..." << endl;
  graph_->add_node(num_nodes);
  // target_frame.clearBilateralPotential();
  // target_frame.clearDistToFgPotential();
  for (int i = 0; i < num_nodes; ++i) {
    pcl::PointXYZ node;
    node.x = cloud_smooth_(i, 0);
    node.y = cloud_smooth_(i, 1);
    node.z = cloud_smooth_(i, 2);
    vector<double> src_potentials, snk_potentials;
    // Add node potential computed from distance to previous foreground
    addDistToForegroundPotential(node, fg_kdtree, src_potentials, snk_potentials, target_frame);

    addBilateralPotential(node, fg_kdtree, whole_kdtree, fg_cloud, whole_cloud, 0.15, src_potentials, snk_potentials, target_frame);
    // Aggregate two potentials using predefined weights, and add the potential to graph
    aggregatePotential(i, node, src_potentials, snk_potentials, graph_);
  }
  // Add edges and edge potentials
  // First construct kdtree for current image
  cout << "Build kdtree for current image..." << endl;
  cout << "Adding edges and edge potentials..." << endl; 
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree (new pcl::KdTreeFLANN<pcl::PointXYZ>);
  buildKdTree(target_frame.cloud_smooth_, whole_cloud, kdtree);
  for (int i = 0; i < num_nodes; ++i) {
    addEdgesWithinRadius(cloud_smooth_,
                         target_frame.cam_points_, 
                         target_frame.img_,
                         i, RADIUS, kdtree, graph_);
  }
  cout << "Running max flow..." << endl;
  graph_->maxflow();
  cout << "Finished running max flow ..." << endl;
  generateSegmentationFromGraph(graph_, index, target_frame);
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
  size_t num_to = seed_frame.segmentation_->tracked_objects_.size();
  for (size_t i = 0; i < num_to; ++i) {
    cout << "Construct seed frame kdtree for object " << i <<" ..." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud (new pcl::PointCloud<pcl::PointXYZ>);    
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree (new pcl::KdTreeFLANN<pcl::PointXYZ>);
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr whole_kdtree (new pcl::KdTreeFLANN<pcl::PointXYZ>);

    buildForegroundKdTree(seed_frame, seed_frame.getTrackedObject(i + 1), fg_cloud, fg_kdtree);
    buildKdTree(seed_frame.cloud_smooth_, whole_cloud, whole_kdtree);
    // Track object i in every frame in this sequence
    for (size_t j = 1; j < seq.size(); ++j) {
      Scene& target_frame = *seq.getScene(j);
      cout << "Tracking object " << i << " in frame " << j << endl;
      graphCutsSegmentation(fg_kdtree, whole_kdtree, fg_cloud, whole_cloud, i + 1, target_frame);
      buildForegroundKdTree(target_frame,
                            target_frame.getTrackedObject(i + 1),
                            fg_cloud,
                            fg_kdtree);
      buildKdTree(target_frame.cloud_smooth_, whole_cloud, whole_kdtree);

 
      // quit the program if there is no segmentation
      //if (fg_kdtree == NULL) break;

      // Find the closest object each segmented point belongs to 
    }
  }
  // Save segmentation of the whole sequence
  saveSequence(seq);
  return 0;
}
