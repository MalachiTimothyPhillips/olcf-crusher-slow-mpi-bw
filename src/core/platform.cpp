#include "platform.hpp"

deviceVector_t::deviceVector_t(const dlong _vectorSize, const dlong _nVectors, const dlong _wordSize, const std::string _vectorName)
: vectorSize(_vectorSize),
  nVectors(_nVectors),
  wordSize(_wordSize),
  vectorName(_vectorName)
{
  if(vectorSize <= 0 || nVectors <= 0 || wordSize <= 0) return; // bail
  o_vector = platform->device.malloc(vectorSize * nVectors, wordSize);
  // set slices
  for(int s = 0 ; s < nVectors; ++s){
    slices.push_back(o_vector + s * vectorSize * wordSize);
  }
}

occa::memory&
deviceVector_t::at(const int i)
{
  if(i >= nVectors){
    if(platform->comm.mpiRank == 0){
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