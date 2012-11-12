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
  double norm_accuracy;
  it = set_intersection(result.begin(), result.end(), golden.begin(), golden.end(), true_positives.begin());
  int num_tp = int(it - true_positives.begin());
  int num_fp = result.size() - num_tp;
  int num_wrong = result.size() + golden.size() - 2 * num_tp;
  if (golden.size() == 0) {
    norm_accuracy = 0;
  } else {
    double accuracy = double(num_wrong) / golden.size();
    norm_accuracy = 1.0 - ((1.0 < accuracy) ? 1.0 : accuracy);
  }
  cout << "True positives: " << num_tp
       << " False positives: " << num_fp
       << " Foreground size: " << golden.size() 
       << " Norm accuracy: " << norm_accuracy << endl;

  return norm_accuracy;
}

double evaluateSegmentation(Scene& result_scene, Scene& golden_scene) {
  set<int> result;
  set<int> golden;
  if (result_scene.segmentation_ != NULL) {
    insertIndexIntoSet(result_scene.segmentation_->tracked_objects_, result); 
  }
  if (golden_scene.segmentation_ != NULL) {
    insertIndexIntoSet(golden_scene.segmentation_->tracked_objects_, golden); 
  }
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
  double sum_accuracy = 0.0;
  for (size_t i = 1; i < result_seq.size(); ++i) {
    cout << "Evaluating frame " << i << endl;
    sum_accuracy += evaluateSegmentation(*result_seq.getScene(i), *golden_seq.getScene(i));
  }
  cout << "Average accuracy across all frames: " << sum_accuracy / (result_seq.size() - 1) << endl;
  return 0;
}
