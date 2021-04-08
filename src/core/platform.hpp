#ifndef platform_hpp_
#define platform_hpp_
#include <occa.hpp>
#include <vector>

class deviceVector_t{
public:
// allow implicit conversion between this and the underlying occa::memory object
  operator occa::memory&(){ return o_vector; }
// allow implicit conversion between this and kernelArg (for passing to kernels)
  operator occa::kernelArg(){ return o_vector; }
  deviceVector_t(const dlong _vectorSize, const dlong _nVectors, const dlong _wordSize, std::string _vectorName = "");
  occa::memory& at(const int);
private:
  occa::memory o_vector;
  std::vector<occa::memory> slices;
  const dlong vectorSize;
  const dlong nVectors;
  const dlong wordSize;
  const std::string vectorName;
};

#endif