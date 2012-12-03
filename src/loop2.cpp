#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <graphcuts/typedefs.h>
#include <graphcuts/maxflow_inference.h>
#include <graphcuts/structural_svm.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>

#define DEPG_SIGMA (getenv("DEPG_SIGMA") ? atof(getenv("DEPG_SIGMA")) : 0.0625)
#define EDGE_SIGMA (getenv("EDGE_SIGMA") ? atof(getenv("EDGE_SIGMA")) : 1)
#define COLOR_SIGMA (getenv("COLOR_SIGMA") ? atof(getenv("COLOR_SIGMA")) : 30.0)
#define BILATERAL_SIGMA (getenv("BILATERAL_SIGMA") ? atof(getenv("BILATERAL_SIGMA")) : 0.05)
#define NORMAL_SIGMA (getenv("NORMAL_SIGMA") ? atof(getenv("NORMAL_SIGMA")) : 0.1)
#define DIST_W (getenv("DIST_W") ? atof(getenv("DIST_W")) : 1)
#define COLOR_W (getenv("COLOR_W") ? atof(getenv("COLOR_W")) : 1)
#define BILATERAL_W (getenv("BILATERAL_W") ? atof(getenv("BILATERAL_W")) : 1)
#define DEPG_W (getenv("DEPG_W") ? atof(getenv("DEPG_W")) : 1)
#define NORMAL_W (getenv("NORMAL_W") ? atof(getenv("NORMAL_W")) : 1)
#define DEBUG (getenv("DEBUG") ? atof(getenv("DEBUG")) : 0) 
#define RADIUS (getenv("RADIUS") ? atof(getenv("RADIUS")) : 0.15)

using namespace std;
namespace bfs = boost::filesystem;
namespace gc = graphcuts;

int num_nan = 0;


namespace {
  static const int INF = 9999999;
  static const int vmin = 10, vmax = 256, smin = 30;
}

string usageString()
{
  ostringstream oss;
  oss << "Usage: ssvm_learning2 Gold_DIR (e.g. ./ssvm_learning2 ~/cs231a/golden04)" << endl;
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
  if (fg_cloud->width == 0)
    return;
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
                            pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud) {
  vector<int> nearest(1);
  vector<float> distance(1);
  fg_kdtree->nearestKSearch(node, 1, nearest, distance);
  pcl::PointXYZ neighbor = fg_cloud->points[nearest[0]];
  return pcl::euclideanDistance(node, neighbor);
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

double computeNormalPotential(int index,
                              int neighbor_index,
                              Eigen::MatrixXf& normals) {
  // Check for validity of normal vectors
  if (isnan(normals(index, 0)) || isinf(normals(index, 0)) ||
      isnan(normals(index, 1)) || isinf(normals(index, 1)) ||
      isnan(normals(index, 2)) || isinf(normals(index, 2)) ||
      isnan(normals(neighbor_index, 0)) || isinf(normals(neighbor_index, 0)) ||
      isnan(normals(neighbor_index, 1)) || isinf(normals(neighbor_index, 1)) ||
      isnan(normals(neighbor_index, 2)) || isinf(normals(neighbor_index, 2))) {
    return 0.0;
  }

  // Compute the dot product of normal vector at the current point and its
  // neighbor
  double dot = normals(neighbor_index, 0) * normals(index, 0) +
    normals(neighbor_index, 1) * normals(index, 1) +
    normals(neighbor_index, 2) * normals(index, 2);

  double angle = acos(dot);
  if (fabs(dot - 1e-6) > 1 || fabs(dot + 1e-6) > 1)
    angle = 0.0;
  if (isnan(angle)) {
    cout << "NaN in SurfaceNormal" << dot << " " << acos(dot) << endl;
    abort();
  }

  return exp(-angle/NORMAL_SIGMA);
}


void addEdgesWithinRadius(Eigen::MatrixXf& cloud_smooth_,
                          Eigen::MatrixXi& cam_points_,
			  Eigen::MatrixXf& normals,
                          cv::Mat& img,
                          int index,
                          float radius,
                          pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree,
                          /*graphcuts::Graph3dPtr graph*/
                          vector<gc::DynamicSparseMat>& edge_pot) {
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
      // update distance
      Eigen::Vector3f pt = cloud_smooth_.block(neighbors[i], 0, 1, 3).transpose();
      double dist = sqrt((pt(0) - search_point.x) * (pt(0) - search_point.x) +
                         (pt(1) - search_point.y) * (pt(1) - search_point.y) +
                         (pt(2) - search_point.z) * (pt(2) - search_point.z));
      double dist_potential = exp(-dist/EDGE_SIGMA);
      double color_potential = exp(-getColorDistance(index, neighbors[i], cam_points_, img)
                                   /COLOR_SIGMA);
      double normal_potential = computeNormalPotential(index, neighbors[i], normals);

      // double edge_potential = COLOR_W * color_potential + DIST_W * dist_potential; 
      // Push edge potential into our vector for learning
      edge_pot[0].coeffRef(index, neighbors[i]) = dist_potential;
      edge_pot[0].coeffRef(neighbors[i], index) = dist_potential;
      edge_pot[1].coeffRef(neighbors[i], index) = color_potential;
      edge_pot[1].coeffRef(index, neighbors[i]) = color_potential;
      edge_pot[2].coeffRef(index, neighbors[i]) = normal_potential;
      edge_pot[2].coeffRef(neighbors[i], index) = normal_potential;
      // Add this edge potential to graphcuts
      // graph->add_edge(index, neighbors[i], edge_potential, edge_potential);
    }
  }
}

void generateSegmentationFromGraph(/*graphcuts::Graph3dPtr graph,*/
                                   int index,
                                   /*Scene& target_frame,*/
                                   Scene& gold_frame,
                                   gc::VecXiPtr label) {
  TrackedObject gold_to = gold_frame.getTrackedObject(index);
  for (int i = 0; i < gold_frame.cloud_smooth_.rows(); ++i) {
    (*label)[i] = 0;
  }
  for(int i = 0; i < gold_to.indices_.size(); i++){
    int obj_index = gold_to.indices_[i];
    (*label)[obj_index] = 1;
    // cout<< obj_index << ' ';
  }
  // target_frame.addTrackedObject(gold_to);
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
                                  int node_index,
                                  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
			          pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                                  Eigen::VectorXd& src_potentials,
                                  Eigen::VectorXd& snk_potentials,
                                  Scene& target_frame) {
  double dist = distToPrevForeground(node, fg_kdtree, fg_cloud);
  double src_pot = exp(-dist/DEPG_SIGMA);
  src_potentials[node_index] = src_pot;
  snk_potentials[node_index] = 1.0 - src_pot;
  target_frame.distToFg_potential_.potentials_.push_back(src_pot);
}

void aggregatePotential(int node_index,
                        pcl::PointXYZ& node,
                        vector<Eigen::VectorXd>& src_potentials,
                        vector<Eigen::VectorXd>& snk_potentials
                        /*graphcuts::Graph3dPtr graph_*/) {
  double src_pot = DEPG_W * src_potentials[0][node_index] +
    BILATERAL_W * src_potentials[1][node_index];
  double snk_pot = DEPG_W * snk_potentials[0][node_index] +
    BILATERAL_W * snk_potentials[1][node_index];
  if (DEBUG == 1) {
    cout << "Node (" << node.x << ", " << node.y << ", " << node.z << "): srcdepth-"
         << src_potentials[0][node_index]
         << " srcbilateral-" << src_potentials[1][node_index] << " snkdepth-"
         << snk_potentials[0][node_index] << " snkbilateral-"
         << snk_potentials[1][node_index] << "    src: " << src_pot << " snk: " << snk_pot << endl;
  }
  //graph_->add_tweights(node_index, src_pot, snk_pot);
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
                           int node_index,
                           pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                           pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr whole_kdtree,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud,
                           double radius,
                           Eigen::VectorXd& src_potentials,
                           Eigen::VectorXd& snk_potentials,
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
    pcl::PointXYZ& neighbor = whole_cloud->points[neighbors[i]];
    if (isForeground(neighbor, fg_kdtree, fg_cloud)) {
      label = 1;
    } else {
      label = -1;
    }
    sum_terms += label * exp(-pcl::euclideanDistance(node, neighbor)/BILATERAL_SIGMA);

  }
  double energy = 2.0 / (1.0 + exp(-sum_terms)) - 1.0;
  /*cout << "Energy for point (" 
       << node.x << ", "
       << node.y << ", "
       << node.z << ") is " << energy << endl;*/
  // Save bilateral node potential to vector
  target_frame.bilateral_potential_.potentials_.push_back(energy);
  if (energy > 0) {
    src_potentials[node_index] = energy;
    snk_potentials[node_index] = 0;
  } else {
    src_potentials[node_index] = 0;
    snk_potentials[node_index] = -energy;
  }
}

void computeNormalAtCenter(pcl::PointXYZ& center,
                           int index,
                           vector<int>& neighbors,
                           vector<float>& distances,
                           Eigen::MatrixXf& cloud_smooth,
                           Eigen::MatrixXf& normals) {
  if (DEBUG) {
    cout << "Computing normal for point (" << center.x << ", "
         << center.y << ", " << center.z << ")..." << endl;
  }
  vector<double> weights;
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  double total_weight = 0;
  for (size_t i = 0; i < neighbors.size(); ++i) {
    // Gets 3D coord of the neighboring point
    Eigen::Vector3f pt = cloud_smooth.block(neighbors[i], 0, 1, 3).transpose();
    double dist = sqrt((pt(0) - center.x) * (pt(0) - center.x) +
                       (pt(1) - center.y) * (pt(1) - center.y) +
                       (pt(2) - center.z) * (pt(2) - center.z));
    if (DEBUG) cout << pt << "with distance " << dist << endl;
    // Compute weight according to distance to center
    weights.push_back(exp(-dist/0.5));
    total_weight += weights[i];
    mean += pt;
  }
  // Compute mean vector of all neighbors
  mean /= neighbors.size();
  // Normalize weight so that they sum to 1
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] /= total_weight;
  }
  // Compute the 3x3 covariance matrix
  Eigen::Matrix3f X = Eigen::Matrix3f::Zero();
  if (DEBUG) cout << "Weights are: ";
  for (size_t i = 0; i < neighbors.size(); ++i) {
    if (DEBUG) cout << weights[i] << " for point (";
    Eigen::Vector3f pt = weights[i] * (cloud_smooth.block(neighbors[i], 0, 1, 3).transpose() -
                                center.getVector3fMap());
    X += pt * pt.transpose();
  }
  if (DEBUG) cout << endl;
  // Solve for surface normal
  float curvature;
  pcl::solvePlaneParameters(X,
                            normals(index, 0),
                            normals(index, 1),
                            normals(index, 2),
                            curvature);
  pcl::flipNormalTowardsViewpoint(center, 0, 0, 0,
                                  normals(index, 0),
                                  normals(index, 1),
                                  normals(index, 2));
  if (isnan(normals(index, 0)) || isnan(normals(index, 1)) ||
      isnan(normals(index, 2)))
    num_nan++;

  if (DEBUG) {
    cout << "Surface normal is ("
         << normals(index, 0) << ", " << normals(index, 1)
         << ", " << normals(index, 2) << ")" << endl;
  }
}

void computeNormals(pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree,
                    Eigen::MatrixXf& cloud_smooth,
                    double radius,
                    Eigen::MatrixXf& normals) {
  int num_nodes = cloud_smooth.rows();
  for (int i = 0; i < num_nodes; ++i) {
    // Use the current point as the center of the normal
    pcl::PointXYZ center;
    center.x = cloud_smooth(i, 0);
    center.y = cloud_smooth(i, 1);
    center.z = cloud_smooth(i, 2);
    // Search within radius and get all valid neighbors
    vector<float> distances;
    vector<int> neighbors;
    vector<int> valid;
    vector<float> valid_dist;
    kdtree->radiusSearch(center, radius, neighbors, distances);
    for (size_t j = 0; j < neighbors.size(); ++j) {
      if (isnan(cloud_smooth(neighbors[j], 0)) ||
          isnan(cloud_smooth(neighbors[j], 1)) ||
          isnan(cloud_smooth(neighbors[j], 2))) {
        cout << "Invalid coord!" << endl;
        continue;
      }
      valid.push_back(neighbors[j]);
      valid_dist.push_back(neighbors[j]);
    }
    // Compute normal at this center by estimating a plane using valid neighbors
    computeNormalAtCenter(center, i, valid, valid_dist, cloud_smooth, normals);
  }
  cout << "Percentage nan normals: " << (double)num_nan/(double)num_nodes;
  num_nan = 0;
}


void graphCutsSegmentation(pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr fg_kdtree,
                           pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr whole_kdtree,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr fg_cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr whole_cloud,
                           int index,
                           Scene& target_frame,
                           /*Scene& gold_frame,*/
                           vector<gc::PotentialsCache::Ptr>& caches,
                           vector<gc::VecXiPtr>& labels) {
  Eigen::MatrixXf cloud_smooth_ = target_frame.cloud_smooth_;
  int num_nodes = cloud_smooth_.rows();
  /*graphcuts::Graph3dPtr graph_ = 
    graphcuts::Graph3dPtr(new graphcuts::Graph3d(num_nodes, cloud_smooth_.rows() * 10));*/
  // Add nodes and node potentials
  //cout << "Adding nodes to graph..." << endl;
  //graph_->add_node(num_nodes);
  // target_frame.clearBilateralPotential();
  // target_frame.clearDistToFgPotential();
  // We have two node potentials, so we construct a vector of two VectorXd.
  vector<Eigen::VectorXd> src_pot(2, Eigen::VectorXd(num_nodes));
  vector<Eigen::VectorXd> snk_pot(2, Eigen::VectorXd(num_nodes));
  gc::PotentialsCache::Ptr cache(new gc::PotentialsCache);
  for (int i = 0; i < num_nodes; ++i) {
    pcl::PointXYZ node;
    node.x = cloud_smooth_(i, 0);
    node.y = cloud_smooth_(i, 1);
    node.z = cloud_smooth_(i, 2);
    // Add node potential computed from distance to previous foreground
    addDistToForegroundPotential(node, i, fg_kdtree, fg_cloud, src_pot[0], snk_pot[0], target_frame);

    addBilateralPotential(node, i, fg_kdtree, whole_kdtree, fg_cloud, whole_cloud, 0.15,
                          src_pot[1], snk_pot[1], target_frame);
    // Aggregate two potentials using predefined weights, and add the potential to graph
    //aggregatePotential(i, node, src_pot, snk_pot/*,*graph_*/);
  }
  // Push node potentials to cache
  cache->npot_names_.addName("DistToFgPotential");
  cache->sink_.push_back(snk_pot[0]);
  cache->source_.push_back(src_pot[0]);
  cache->npot_names_.addName("BilateralPotential");
  cache->sink_.push_back(snk_pot[1]);
  cache->source_.push_back(src_pot[1]);
  // Add edges and edge potentials
  // Initializ a vector for 3d edge potential and color edge potential
  vector<gc::DynamicSparseMat> edge_pot(3, gc::DynamicSparseMat(num_nodes, num_nodes));
  // First construct kdtree for current image
  cout << "Build kdtree for current image..." << endl;
  cout << "Adding edges and edge potentials..." << endl; 
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree (new pcl::KdTreeFLANN<pcl::PointXYZ>);
  buildKdTree(target_frame.cloud_smooth_, whole_cloud, kdtree);
  // Computes surface normal for each depth point
  Eigen::MatrixXf normals = Eigen::MatrixXf::Zero(num_nodes, 3);
  computeNormals(kdtree, target_frame.cloud_smooth_, RADIUS, normals);

  for (int i = 0; i < num_nodes; ++i) {
    addEdgesWithinRadius(cloud_smooth_,
                         target_frame.cam_points_, 
                         normals,
                         target_frame.img_,
                         i, RADIUS, kdtree, /*graph_,*/
                         edge_pot);
  }
  // Push edge potentials to cache
  cache->epot_names_.addName("DistEdgePotential");
  cache->edge_.push_back(gc::SparseMat(edge_pot[0]));
  cache->epot_names_.addName("ColorEdgePotential");
  cache->edge_.push_back(gc::SparseMat(edge_pot[1]));
  cache->epot_names_.addName("NormalPotential");
  cache->edge_.push_back(gc::SparseMat(edge_pot[2]));
  // Running maxflow and generate training labels
  // cout << "Running max flow..." << endl;
  // graph_->maxflow();
  gc::VecXiPtr label(new gc::VecXi(num_nodes));
  generateSegmentationFromGraph(/*graph_,*/ index, target_frame, /*gold_frame,*/ label);
  // Push training features and labels of this frame to caches and labels
  caches.push_back(cache);
  labels.push_back(label);
}

void calculateLabel(string dirpath,
                    vector<gc::PotentialsCache::Ptr>& caches,
                    vector<gc::VecXiPtr>& labels) {
  Sequence seq_gold(dirpath);
  Scene& seed_frame = *seq_gold.getScene(0);
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
    for (size_t j = 1; j < seq_gold.size(); ++j) {
      // Scene& target_frame = *seq.getScene(j);
      Scene& target_frame = *seq_gold.getScene(j);

      cout << "Tracking object " << i << " in frame " << j << endl;
      graphCutsSegmentation(fg_kdtree,
                            whole_kdtree,
                            fg_cloud,
                            whole_cloud,
                            i + 1,
                            target_frame,
                            /*gold_frame,*/
                            caches,
                            labels);
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
}

int main(int argc, char** argv)
{
  /*if(argc < 2) { 
    cout << usageString();
    return 1;
  }

  // string dirpath = argv[1];
  string goldpath = argv[1];

  if(!bfs::exists(goldpath)) {
    cout << goldpath << " does not exist." << endl;
    return 1;
    }
  // Sequence seq(dirpath);
  Sequence seq_gold(goldpath);*/

  // Initialize an instance of ssvm
  double c = 10;
  double precision = 1e-4;
  int num_threads = 1;
  int debug_level = 0;
  gc::StructuralSVM ssvm(c, precision, num_threads, debug_level); 
  // Initialize a vector of pointers to caches that hold node/edge potentials
  vector<gc::PotentialsCache::Ptr> caches;
  // Initialize a vector of pointers to labels
  vector<gc::VecXiPtr> labels;
  // Read in seed frame data

  string rootpath = argv[1];
  int seq_begin = argv[2][0]-'0';
  int seq_n = argv[3][0]-'0';
  int seq_left = argv[4][0]-'0';

  cout<< "data sequences begin: "<< seq_begin << "; end:" << (seq_begin + seq_n -1) << "; skip"<< seq_left << endl;
  int xxx = seq_begin+ seq_n + seq_left ;

  string dirpath;
  for(int i = seq_begin ; i < seq_begin + seq_n; i++){
      if(i == seq_left)
          continue;
    char ch = '0' + i;
    dirpath = rootpath + "0" + ch;
    calculateLabel(dirpath, caches, labels);
  } 
  // Save segmentation of the whole sequence
  // saveSequence(seq);
  // Train model
  cout << "Train model..." << endl;
  gc::Model model = ssvm.train(caches, labels);
  cout << model << endl;
  return 0;
}
