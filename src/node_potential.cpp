#include <segmentation_and_tracking/scene.h>

using namespace std;
namespace bfs = boost::filesystem;
using namespace Eigen;

NodePotential::NodePotential()
{
}

// Reads in node potentials from file 'path'
NodePotential::NodePotential(const string& path)
{
  assert(bfs::exists(path));
  ifstream file(path.c_str());
  deserialize(file);
  file.close();
}

void NodePotential::save(const std::string& path) const
{
  ofstream file(path.c_str());
  serialize(file);
  file.close();
}

void NodePotential::deserialize(std::istream& in)
{
  size_t num_points;
  in.read((char*)&num_points, sizeof(size_t));

  potentials_.resize(num_points);
  for(size_t i = 0; i < num_points; ++i)
    in.read((char*)(&(potentials_[i])), sizeof(double));
}

void NodePotential::serialize(std::ostream& out) const
{
  size_t num_points = potentials_.size();
  out.write((char*)(&num_points), sizeof(size_t));
  for (int i = 0; i < potentials_.size(); ++i) {
    out.write((char*)(&(potentials_[i])), sizeof(double));
  }
}
