#include <segmentation_and_tracking/scene.h>
#include <segmentation_and_tracking/hand_segmenter_view_controller.h>
#include <utility>
#include <set>
#include <algorithm>
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

double sum_f1 = 0;

string usageString()
{
  ostringstream oss;
  oss << "Usage: evaluator RESULT_DIR GOLDEN_DIR" << endl;
  return oss.str();
}

void insertIndexIntoSet(vector<TrackedObject>& to, set<int>& s) {
  for (size_t i = 0; i < to.size(); ++i) {
    s.insert(to[i].indices_.begin(), to[i].indices_.end());
  } 
}

double reportMetrics(set<int>& result, set<int>& golden) {
  vector<int> true_positives(result.size() + golden.size());
  vector<int>::iterator it;
  it = set_intersection(result.begin(), result.end(), golden.begin(), golden.end(), true_positives.begin());
  int num_tp = int(it - true_positives.begin());
  double norm_ = 0.001;
  double precision = (double) num_tp / (result.size()+norm_);
  double recall = (double) num_tp / (golden.size()+norm_);
  double f1 = (precision * recall * 2)/(precision + recall+norm_);
  cout << "True positives: " << num_tp << ", Precision: " << precision
       << " Recall: " << recall
       << " F-1 score: " << f1 << endl; 
  sum_f1 += f1;
}

double evaluateSegmentation(Scene& result_scene, Scene& golden_scene) {
  set<int> result;
  set<int> golden;
  insertIndexIntoSet(result_scene.segmentation_->tracked_objects_, result); 
  insertIndexIntoSet(golden_scene.segmentation_->tracked_objects_, golden); 
  return reportMetrics(result, golden);
}

int main(int argc, char** argv)
{
  if(argc != 3) { 
    cout << usageString();
    return 1;
  }

  string result_path = argv[1];
  string golden_path = argv[2];
  
  if(!bfs::exists(result_path) || !bfs::exists(golden_path)) {
    cout << result_path << " or " << golden_path << " does not exist." << endl;
    return 1;
  }
  Sequence result_seq(result_path);
  Sequence golden_seq(golden_path);
  double f1 = 0.0;
  for (size_t i = 1; i < result_seq.size(); ++i) {
    cout << "Evaluating frame " << i << endl;
    evaluateSegmentation(*result_seq.getScene(i), *golden_seq.getScene(i));
    cout<< "Accumulative f1 score " << (double)sum_f1/result_seq.size()<<endl; 
  }
  cout<< "Average f1 score " << (double)sum_f1/result_seq.size()<<endl; 
  return 0;
}
