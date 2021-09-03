#include <cstdlib>
#include <strings.h>
#include "platform.hpp"
#include "nrs.hpp"
#include "linAlg.hpp"
#include "omp.h"
#include <iostream>

comm_t::comm_t(MPI_Comm _comm)
{
  mpiComm = _comm;
  MPI_Comm_rank(_comm, &mpiRank);
  MPI_Comm_size(_comm, &mpiCommSize);

  long int hostId = gethostid();

  std::vector<long int> hostIds(mpiCommSize, 0);
  MPI_Allgather(&hostId,1,MPI_LONG,hostIds.data(),1,MPI_LONG,_comm);

  std::vector<int> ranksInLocalComm;
  for (int r = 0; r < mpiCommSize; r++){
    if (hostIds[r] == hostId) {
      ranksInLocalComm.push_back(r);
    }
  }

  localCommSize = ranksInLocalComm.size();
  MPI_Group local_group;
  MPI_Group global_group;
  MPI_Comm_group(_comm, &global_group);
  MPI_Group_incl(global_group, localCommSize, ranksInLocalComm.data(), &local_group);

  MPI_Comm_create_group(_comm, local_group, 0, &localComm);

  MPI_Comm_rank(localComm, &localRank);

}

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




platform_t* platform_t::singleton = nullptr;
platform_t::platform_t(setupAide& _options, MPI_Comm _comm)
: options(_options),
  warpSize(32), // CUDA specific warp size
  device(options, _comm),
  timer(_comm, device, 0),
  comm(_comm),
  kernels(*this)
{
  kernelInfo["defines/" "p_NVec"] = 3;
  kernelInfo["defines/" "p_blockSize"] = BLOCKSIZE;
  kernelInfo["defines/" "dfloat"] = dfloatString;
  kernelInfo["defines/" "pfloat"] = pfloatString;
  kernelInfo["defines/" "dlong"] = dlongString;
  kernelInfo["defines/" "hlong"] = hlongString;

  if(device.mode() == "CUDA" && !getenv("OCCA_CUDA_COMPILER_FLAGS")) {
    kernelInfo["compiler_flags"] += "--ftz=true ";
    kernelInfo["compiler_flags"] += "--prec-div=false ";
    kernelInfo["compiler_flags"] += "--prec-sqrt=false ";
    kernelInfo["compiler_flags"] += "--use_fast_math ";
    kernelInfo["compiler_flags"] += "--fmad=true ";

    //kernelInfo["compiler_flags"] += "-Xptxas -dlcm=ca";
  }

  if(device.mode() == "OpenCL") {
    if(!getenv("OCCA_OPENCL_COMPILER_FLAGS")) {
      kernelInfo["compiler_flags"] += " -cl-std=CL2.0 ";
      kernelInfo["compiler_flags"] += " -cl-mad-enable ";
      kernelInfo["compiler_flags"] += " -cl-no-signed-zeros ";
      kernelInfo["compiler_flags"] += " -cl-unsafe-math-optimizations ";
      kernelInfo["compiler_flags"] += " -cl-fast-relaxed-math ";
    }
    kernelInfo["defines/" "hlong"]="long";
  }

  if(device.mode() == "HIP" && !getenv("OCCA_HIP_COMPILER_FLAGS")) {
    warpSize = 64;
    kernelInfo["compiler_flags"] += " -O3 ";
    kernelInfo["compiler_flags"] += " -ffp-contract=fast ";
    kernelInfo["compiler_flags"] += " -funsafe-math-optimizations ";
    kernelInfo["compiler_flags"] += " -ffast-math ";
  }
}
void memPool_t::allocate(const dlong offset, const dlong fields)
{
  ptr = (dfloat*) calloc(offset*fields, sizeof(dfloat));
  slice0 = ptr + 0 * offset;
  slice1 = ptr + 1 * offset;
  slice2 = ptr + 2 * offset;
  slice3 = ptr + 3 * offset;
  slice4 = ptr + 4 * offset;
  slice5 = ptr + 5 * offset;
  slice6 = ptr + 6 * offset;
  slice7 = ptr + 7 * offset;
  slice9 = ptr + 9 * offset;
  slice12 = ptr + 12 * offset;
  slice15 = ptr + 15 * offset;
  slice18 = ptr + 18 * offset;
  slice19 = ptr + 19 * offset;
}
void deviceMemPool_t::allocate(memPool_t& hostMemory, const dlong offset, const dlong fields)
{
  bytesAllocated = fields * offset * sizeof(dfloat);
  o_ptr = platform->device.malloc(offset*fields*sizeof(dfloat), hostMemory.slice0);
  slice0 = o_ptr.slice(0 * offset * sizeof(dfloat));
  slice1 = o_ptr.slice(1 * offset * sizeof(dfloat));
  slice2 = o_ptr.slice(2 * offset * sizeof(dfloat));
  slice3 = o_ptr.slice(3 * offset * sizeof(dfloat));
  slice4 = o_ptr.slice(4 * offset * sizeof(dfloat));
  slice5 = o_ptr.slice(5 * offset * sizeof(dfloat));
  slice6 = o_ptr.slice(6 * offset * sizeof(dfloat));
  slice7 = o_ptr.slice(7 * offset * sizeof(dfloat));
  slice9 = o_ptr.slice(9 * offset * sizeof(dfloat));
  slice12 = o_ptr.slice(12 * offset * sizeof(dfloat));
  slice15 = o_ptr.slice(15 * offset * sizeof(dfloat));
  slice18 = o_ptr.slice(18 * offset * sizeof(dfloat));
  slice19 = o_ptr.slice(19 * offset * sizeof(dfloat));
}
void
platform_t::create_mempool(const dlong offset, const dlong fields)
{
  mempool.allocate(offset, fields);
  o_mempool.allocate(mempool, offset, fields);
}

occa::kernel
device_t::buildNativeKernel(const std::string &filename,
                         const std::string &kernelName,
                         const occa::properties &props) const
{
  occa::properties nativeProperties = props;
  nativeProperties["okl/enabled"] = false;
  return occa::device::buildKernel(filename, kernelName, nativeProperties);
}
occa::kernel
device_t::buildKernel(const std::string &filename,
                         const std::string &kernelName,
                         const occa::properties &props,
                         std::string suffix) const
{
  if(filename.find(".okl") != std::string::npos){
    occa::properties propsWithSuffix = props;
    propsWithSuffix["kernelNameSuffix"] = suffix;
    return occa::device::buildKernel(filename, kernelName, propsWithSuffix);
  }
  else{
    occa::properties propsWithSuffix = props;
    propsWithSuffix["defines/SUFFIX"] = suffix;
    propsWithSuffix["defines/TOKEN_PASTE_(a,b)"] = std::string("a##b");
    propsWithSuffix["defines/TOKEN_PASTE(a,b)"] = std::string("TOKEN_PASTE_(a,b)");
    propsWithSuffix["defines/FUNC(a)"] = std::string("TOKEN_PASTE(a,SUFFIX)");
    const std::string alteredName =  kernelName + suffix;
    return this->buildNativeKernel(filename, alteredName, propsWithSuffix);
  }
}
occa::memory
device_t::mallocHost(const dlong Nbytes)
{
  occa::properties props;
  props["host"] = true;
  occa::memory h_scratch = occa::device::malloc(Nbytes, props);
  return h_scratch;
}
occa::memory
device_t::malloc(const dlong Nbytes, const occa::properties& properties)
{
  return occa::device::malloc(Nbytes, nullptr, properties);
}
occa::memory
device_t::malloc(const dlong Nbytes, const void* src, const occa::properties& properties)
{
  if(!src){
    if(Nbytes > bufferSize)
    {
      if(bufferSize > 0) std::free(_buffer);
      _buffer = std::calloc(Nbytes, 1);
      bufferSize = Nbytes;
    }
  }
  const void* init_ptr = (src) ? src : _buffer;
  return occa::device::malloc(Nbytes, init_ptr, properties);
}
occa::memory
device_t::malloc(const dlong Nword , const dlong wordSize, occa::memory src)
{
  return occa::device::malloc(Nword * wordSize, src);
}
occa::memory
device_t::malloc(const dlong Nword , const dlong wordSize)
{
  const dlong Nbytes = Nword * wordSize;
  if(Nbytes > bufferSize)
  {
    if(bufferSize > 0) std::free(_buffer);
    _buffer = std::calloc(Nword, wordSize);
    bufferSize = Nbytes;
  }
  return occa::device::malloc(Nword * wordSize, _buffer);
}
device_t::device_t(setupAide& options, MPI_Comm comm)
{
  // OCCA build stuff
  char deviceConfig[BUFSIZ];
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int device_id = 0;

  if(options.compareArgs("DEVICE NUMBER", "LOCAL-RANK")) {
    long int hostId = gethostid();

    long int* hostIds = (long int*) std::calloc(size,sizeof(long int));
    MPI_Allgather(&hostId,1,MPI_LONG,hostIds,1,MPI_LONG,comm);

    int totalDevices = 0;
    for (int r = 0; r < rank; r++)
      if (hostIds[r] == hostId) device_id++;
    for (int r = 0; r < size; r++)
      if (hostIds[r] == hostId) totalDevices++;
  } else {
    options.getArgs("DEVICE NUMBER",device_id);
  }

  occa::properties deviceProps;
  std::string requestedOccaMode; 
  options.getArgs("THREAD MODEL", requestedOccaMode);

  if(strcasecmp(requestedOccaMode.c_str(), "CUDA") == 0) {
    sprintf(deviceConfig, "{mode: 'CUDA', device_id: %d}", device_id);
  }else if(strcasecmp(requestedOccaMode.c_str(), "HIP") == 0) {
    sprintf(deviceConfig, "{mode: 'HIP', device_id: %d}",device_id);
  }else if(strcasecmp(requestedOccaMode.c_str(), "OPENCL") == 0) {
    int plat = 0;
    options.getArgs("PLATFORM NUMBER", plat);
    sprintf(deviceConfig, "{mode: 'OpenCL', device_id: %d, platform_id: %d}", device_id, plat);
  }else if(strcasecmp(requestedOccaMode.c_str(), "OPENMP") == 0) {
    if(rank == 0) printf("OpenMP backend currently not supported!\n");
    ABORT(EXIT_FAILURE);
    sprintf(deviceConfig, "{mode: 'OpenMP'}");
  }else if(strcasecmp(requestedOccaMode.c_str(), "CPU") == 0 ||
           strcasecmp(requestedOccaMode.c_str(), "SERIAL") == 0) {
    sprintf(deviceConfig, "{mode: 'Serial'}");
    options.setArgs("THREAD MODEL", "SERIAL");
    options.getArgs("THREAD MODEL", requestedOccaMode);
  } else {
    if(rank == 0) printf("Invalid requested backend!\n");
    ABORT(EXIT_FAILURE);
  }

  if(rank == 0) printf("Initializing device \n");
  this->setup((std::string)deviceConfig);
  this->comm = comm;
 
  if(rank == 0)
    std::cout << "active occa mode: " << this->mode() << "\n\n";

  if(strcasecmp(requestedOccaMode.c_str(), this->mode().c_str()) != 0) {
    if(rank == 0) printf("active occa mode does not match selected backend!\n");
    ABORT(EXIT_FAILURE);
  } 

  // overwrite compiler settings to ensure
  // compatability of libocca and kernelLauchner 
  if(this->mode() != "Serial") {
    std::string buf;
    buf.assign(getenv("NEKRS_CXX"));
    setenv("OCCA_CXX", buf.c_str(), 1);
    buf.assign(getenv("NEKRS_CXXFLAGS"));
    setenv("OCCA_CXXFLAGS", buf.c_str(), 1);
  }

  int Nthreads = 1;
  if(this->mode() != "OpenMP") omp_set_num_threads(Nthreads);

  bufferSize = 0;

  _device_id = device_id;
}

void
kernelRequestManager_t::add_kernel(const std::string& m_requestName,
                const std::string& m_fileName,
                const std::string& m_kernelName,
                const occa::properties& m_props,
                std::string m_suffix,
                bool checkUnique)
{
  this->add_kernel(kernelRequest_t{m_requestName, m_fileName, m_kernelName, m_props, m_suffix}, checkUnique);
}
void
kernelRequestManager_t::add_kernel(kernelRequest_t request, bool checkUnique)
{
  auto iterAndBoolPair = kernels.insert(request);
  if(iterAndBoolPair.second) duplicates[request] = {};
  if(checkUnique)
  {
    int unique = (iterAndBoolPair.second) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &unique, 1, MPI_INT, MPI_MIN, platformRef.comm.mpiComm);
    if(!unique){
      if(platformRef.comm.mpiRank == 0)
      {
        std::cout << "Error in kernelRequestManager_t::add_kernel\n";
        std::cout << "Request details:\n";
        std::cout << request.to_string();
      }
      ABORT(1);
    }
  }

  if(!iterAndBoolPair.second)
  {
    duplicates[request].push_back(request);
  }

  const auto fileName = request.fileName;
  fileNameToRequestMap[fileName].insert(request);
}
occa::kernel
kernelRequestManager_t::load(const std::string& request, bool checkValid) const
{
  if(checkValid){
    bool issueError = 0;
    issueError = !processed();
    issueError = (requestToKernelMap.count(request) == 0);

    int errorFlag = issueError ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &errorFlag, 1, MPI_INT, MPI_MAX, platformRef.comm.mpiComm);

    if(errorFlag){
      if(platformRef.comm.mpiRank == 0)
      {
        std::cout << "\n";
        std::cout << "===========================================================\n";
        std::cout << "Error in kernelRequestManager_t::load. Failing now.\n";
        std::cout << "Requested kernel : " << request << "\n";

        std::cout << "All entries:\n";
        for(auto&& keyAndValue : requestToKernelMap)
        {
          std::cout << "\t" << keyAndValue.first << "\n";
        }
        std::cout << "===========================================================\n";
      }
      ABORT(1);
    }
  }

  return requestToKernelMap.at(request);
}

void
kernelRequestManager_t::compile()
{

  if(kernelsProcessed) return;
  kernelsProcessed = true;

  constexpr int maxCompilingRanks {100};

  int buildNodeLocal;
  if(getenv("NEKRS_BUILD_NODE_LOCAL")){
    buildNodeLocal = std::stoi(getenv("NEKRS_BUILD_NODE_LOCAL"));
  } else {
    buildNodeLocal = 0;
  }

  const int rank = buildNodeLocal ? platformRef.comm.localRank : platformRef.comm.mpiRank;
  const int ranksCompiling =
    std::min(
      maxCompilingRanks,
      buildNodeLocal ?
        platformRef.comm.localCommSize :
        platformRef.comm.mpiCommSize
    );

  std::vector<std::string> fileNames(fileNameToRequestMap.size());

  unsigned ctr = 0;
  for(auto&& fileNameAndRequests : fileNameToRequestMap){
    fileNames[ctr] = fileNameAndRequests.first;
    ctr++;
  }

  const auto& device = platformRef.device;
  auto& requestToKernel = requestToKernelMap;
  auto& fileNameToRequest = fileNameToRequestMap;
  auto compileKernels = [&fileNames, &fileNameToRequest, &requestToKernel, &device, rank, ranksCompiling](){
    if(rank >= ranksCompiling) return;
    const unsigned nFiles = fileNameToRequest.size();
    for(unsigned fileId = 0; fileId < nFiles; ++fileId)
    {
      if(fileId % ranksCompiling == rank){
        const auto& kernelRequests = fileNameToRequest[fileNames[fileId]];
        for(auto && kernelRequest : kernelRequests){
          const std::string requestName = kernelRequest.requestName;
          const std::string fileName = kernelRequest.fileName;
          const std::string kernelName = kernelRequest.kernelName;
          const std::string suffix = kernelRequest.suffix;
          const occa::properties props = kernelRequest.props;

#if 0
          std::ostringstream ss;
          ss << "requestName = " << requestName << "\n";
          ss << "fileName = " << fileName << "\n";
          ss << "kernelName = " << kernelName << "\n";
          ss << "suffix = " << suffix << "\n";
          std::cout << ss.str() << std::endl;
#endif
          auto kernel = device.buildKernel(fileName, kernelName, props, suffix);
          requestToKernel[requestName] = kernel;
        }
      }
    }
  };

  const auto& kernelRequests = this->kernels;
  auto loadKernels = [&requestToKernel, &kernelRequests,&device](){
    for(auto&& kernelRequest : kernelRequests)
    {
      const std::string requestName = kernelRequest.requestName;
      if(requestToKernel.count(requestName) == 0){
        const std::string fileName = kernelRequest.fileName;
        const std::string kernelName = kernelRequest.kernelName;
        const std::string suffix = kernelRequest.suffix;
        const occa::properties props = kernelRequest.props;
        auto kernel = device.buildKernel(fileName, kernelName, props, suffix);
        requestToKernel[requestName] = kernel;
      }
    }
  };

  MPI_Barrier(platform->comm.mpiComm);
  if(rank == 0)
  {
    std::cout << "Calling compileKernels\n";
  }
  {
    compileKernels();
  }

  MPI_Barrier(platform->comm.mpiComm);
  if(rank == 0)
  {
    std::cout << "Calling loadKernels\n";
  }

  {
    loadKernels();
  }


  // loop through the duplicates
  for(auto&& entryAndVector : duplicates){
    auto& vec = entryAndVector.second;
    if(vec.size() > 0){
      auto& entry = entryAndVector.first;
      auto& propsEntry = entry.props;
      for(auto && dup : vec){
        auto& propsDup = dup.props;
        if(propsDup.toString() != propsEntry.toString()){
          std::cout << "Entry properties:\n";
          std::cout << "{\n";
          std::cout << entry.to_string() << "\n";
          std::cout << "}\n";
          std::cout << "Duplicate properties:\n";
          std::cout << "{\n";
          std::cout << dup.to_string() << "\n";
          std::cout << "}\n";
        }
      }
    }
  }
}


