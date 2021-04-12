#include "platform.hpp"

deviceVector_t::deviceVector_t(const dlong _vectorSize, const dlong _nVectors, const dlong _wordSize, const dlong _rank, occa::device& device, const std::string _vectorName)
: vectorSize(_vectorSize),
  nVectors(_nVectors),
  wordSize(_wordSize),
  vectorName(_vectorName),
  rank(_rank)
{
  if(vectorSize <= 0 || nVectors <= 0 || wordSize <= 0) return; // bail
  void* hostBuffer = calloc(vectorSize*nVectors, wordSize);
  o_vector = device.malloc(vectorSize * nVectors*wordSize, hostBuffer);
  // set slices
  for(int s = 0 ; s < nVectors; ++s){
    slices.push_back(o_vector + s * vectorSize * wordSize);
  }
}

occa::memory&
deviceVector_t::at(const int i)
{
  if(i >= nVectors){
    if(rank == 0){
      printf("ERROR: deviceVector_t(%s) has %d size, but an attempt to access entry %i was made!\n",
        vectorName.c_str(),
        nVectors,
        i
      );
    }
    ABORT(EXIT_FAILURE);
    return o_vector;
  }
  occa::memory slice = o_vector + i * vectorSize * wordSize;
  return slices[i];
}