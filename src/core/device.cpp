#include <filesystem>
#include "device.hpp" 
#include "platform.hpp"
#include <unistd.h>
#include <regex>

namespace fs = std::filesystem;

namespace {

bool mkDir(const fs::path &file_path)
{
  size_t pos = 0;
  auto ret_val = true;

  std::string dir_path(file_path);
  if (!fs::is_directory((file_path)))
    dir_path = file_path.parent_path();

  while (ret_val && pos != std::string::npos) {
    pos = dir_path.find('/', pos + 1);
    const auto dir = fs::path(dir_path.substr(0, pos));
    if (!fs::exists(dir)) {
      ret_val = fs::create_directory(dir);
    }
  }

  return ret_val;
}

int fileBcastNodes(const fs::path srcPath,
                   const fs::path dstPath,
                   int rankCompile,
                   MPI_Comm comm,
                   int verbose)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  int err = 0;
  if (rank == rankCompile && !fs::exists(srcPath)) {
    err++;
    std::cout << __func__ << ": cannot stat "
              << "" << srcPath << ":"
              << " No such file or directory\n";
  }
  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_SUM, comm);
  if (err)
    return EXIT_FAILURE;

  const auto path0 = fs::current_path();

  int localRank;
  const int localRankRoot = 0;

  int color = MPI_UNDEFINED;
  MPI_Comm commLocal;
  {
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &commLocal);
    MPI_Comm_rank(commLocal, &localRank);
    if (localRank == localRankRoot)
      color = 1;
    if (rank == rankCompile)
      color = 1;
  }
  MPI_Comm commNode;
  MPI_Comm_split(comm, color, rank, &commNode);

  int nodeRank = -1;
  int nodeRankRoot = -1;
  if (color != MPI_UNDEFINED)
    MPI_Comm_rank(commNode, &nodeRank);
  if (rank == rankCompile)
    nodeRankRoot = nodeRank;
  MPI_Bcast(&nodeRankRoot, 1, MPI_INT, rankCompile, comm);

  // generate file list
  std::vector<std::string> fileList;
  if (nodeRank == nodeRankRoot) {
    if (!fs::is_directory((srcPath))) {
      fileList.push_back(srcPath);
    }
    else {
      for (const auto &dirEntry : fs::recursive_directory_iterator(srcPath)) {
        if (dirEntry.is_regular_file())
          fileList.push_back(dirEntry.path());
      }
    }
  }
  int nFiles = (nodeRank == nodeRankRoot) ? fileList.size() : 0;
  MPI_Bcast(&nFiles, 1, MPI_INT, rankCompile, comm);
  if (!nFiles)
    return EXIT_SUCCESS;

  // bcast file list
  for (int i = 0; i < nFiles; i++) {
    int bufSize = (rank == rankCompile) ? fileList.at(i).size() : 0;
    MPI_Bcast(&bufSize, 1, MPI_INT, rankCompile, comm);

    auto buf = (char *)std::malloc(bufSize * sizeof(char));
    if (rank == rankCompile)
      std::strncpy(buf, fileList.at(i).c_str(), bufSize);
    MPI_Bcast(buf, bufSize, MPI_CHAR, rankCompile, comm);
    if (rank != rankCompile)
      fileList.push_back(std::string(buf, 0, bufSize));
    free(buf);
  }

  for (const auto &file : fileList) {
    int bufSize = 0;
    const std::string filePath = dstPath / fs::path(file);

    unsigned char *buf = nullptr;
    if (color != MPI_UNDEFINED) {
      if (nodeRank == nodeRankRoot)
        bufSize = fs::file_size(file);
      MPI_Bcast(&bufSize, 1, MPI_INT, nodeRankRoot, commNode);

      if (bufSize > std::numeric_limits<int>::max()) {
        if (rank == rankCompile)
          std::cout << __func__ << ": file size of "
                    << "" << file << " too large!\n";
        return EXIT_FAILURE;
      }

      buf = (unsigned char *)std::malloc(bufSize * sizeof(unsigned char));

      if (nodeRank == nodeRankRoot) {
        std::ifstream input(file, std::ios::in | std::ios::binary);
        std::stringstream sstr;
        input >> sstr.rdbuf();
        input.close();
        std::memcpy(buf, sstr.str().c_str(), bufSize);
      }
      MPI_Bcast(buf, bufSize, MPI_BYTE, nodeRankRoot, commNode);

      if (nodeRank == nodeRankRoot && verbose)
        std::cout << __func__ << ": " << file << " -> " << filePath << " (" << bufSize << " bytes)"
                  << std::endl;
    }

    // write file to node-local storage;
    if (localRank == localRankRoot)
      mkDir(filePath); // create directory and parents if they don't already exist

    MPI_File fh;
    MPI_File_open(commLocal, filePath.c_str(), MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);

    if (localRank == localRankRoot) {
      MPI_Status status;
      MPI_File_write_at(fh, 0, buf, bufSize, MPI_BYTE, &status);
    }
    free(buf);

    MPI_File_sync(fh);
    MPI_Barrier(commLocal);
    MPI_File_sync(fh);
    MPI_File_close(&fh);
  }

  fs::current_path(path0);

  return EXIT_SUCCESS;
}
} // namespace

occa::kernel device_t::buildKernelFromStringMPI(const std::string &kernelSource,
                                                const std::string &kernelName,
                                                const occa::properties &props,
                                                int rankCompile) const
{
  std::string srcPath;
  if (_comm.mpiRank == rankCompile) {
    occa::kernel knl = _device.buildKernelFromString(kernelSource, kernelName, props);
    srcPath = fs::path(knl.sourceFilename()).parent_path();
  }

  // bcast srcPath
  int srcPathSize = (_comm.mpiRank == rankCompile) ? srcPath.size() : 0;
  MPI_Bcast(&srcPathSize, 1, MPI_INT, rankCompile, _comm.mpiComm);
  auto buf = (char *)std::malloc(srcPathSize * sizeof(char));
  if (_comm.mpiRank == rankCompile)
    std::strncpy(buf, srcPath.c_str(), srcPathSize);
  MPI_Bcast(buf, srcPathSize, MPI_CHAR, rankCompile, _comm.mpiComm);
  srcPath.assign(buf, srcPathSize);
  free(buf);

  // bcast cache entry to compute nodes
  std::string nodeScratchDir;
  if (getenv("NEKRS_NODE_SCRATCH_DIR"))
    nodeScratchDir.assign(getenv("NEKRS_NODE_SCRATCH_DIR"));
  else
    nodeScratchDir.assign(fs::temp_directory_path());

  std::string uniquePath;
  {
    int uniquePathSize;
    if (_comm.mpiRank == rankCompile) {
      char tmp[] = "occa_XXXXXX";
      mkstemp(tmp);
      uniquePath = std::string(tmp);
      uniquePathSize = uniquePath.size();
    }
    MPI_Bcast(&uniquePathSize, 1, MPI_INT, rankCompile, _comm.mpiComm);
    auto buf = (char *)std::malloc(uniquePathSize * sizeof(char));
    if (_comm.mpiRank == rankCompile)
      std::strncpy(buf, uniquePath.c_str(), uniquePathSize);
    MPI_Bcast(buf, uniquePathSize, MPI_CHAR, rankCompile, _comm.mpiComm);
    uniquePath.assign(buf, uniquePathSize);
    free(buf);
  }
  const auto dstPath = fs::path(nodeScratchDir) / fs::path(uniquePath);

  {
    const auto path0 = fs::current_path();
    fs::current_path(occa::env::OCCA_CACHE_DIR);
    int err =
        fileBcastNodes(fs::relative(srcPath, fs::current_path()), dstPath, rankCompile, _comm.mpiComm, true);
    MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_SUM, _comm.mpiComm);
    if (err)
      MPI_Abort(_comm.mpiComm, EXIT_FAILURE);
    fs::current_path(path0);
  }

  // load kernel binary from node-local cache
  const auto OCCA_CACHE_DIR0 = occa::env::OCCA_CACHE_DIR;
  occa::env::OCCA_CACHE_DIR = dstPath / "";
  occa::kernel knl = _device.buildKernelFromString(kernelSource, kernelName, props);
  occa::env::OCCA_CACHE_DIR = OCCA_CACHE_DIR0;

  MPI_Barrier(_comm.mpiComm);
  fs::remove_all(dstPath);

  return knl;
}

occa::kernel device_t::buildKernelMPI(const std::string &kernelFile,
                                      const std::string &kernelName,
                                      const occa::properties &props,
                                      int rankCompile) const
{
  int err = 0;
  if (_comm.mpiRank == rankCompile && !fs::exists(kernelFile)) {
    err++;
    std::cout << __func__ << ": cannot stat "
              << "" << kernelFile << ":"
              << " No such file or directory\n";
  }
  MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_SUM, _comm.mpiComm);
  if (err)
    MPI_Abort(_comm.mpiComm, EXIT_FAILURE);

  int kernelContentSize = (_comm.mpiRank == rankCompile) ? fs::file_size(kernelFile) : 0;
  MPI_Bcast(&kernelContentSize, 1, MPI_INT, rankCompile, _comm.mpiComm);

  auto buf = (char *)std::malloc(kernelContentSize * sizeof(char));
  if (_comm.mpiRank == rankCompile) {
    std::ostringstream streamBuf;

    std::ifstream fileStream(kernelFile, std::ios::in);
    streamBuf << fileStream.rdbuf();
    fileStream.close();

    std::strncpy(buf, streamBuf.str().c_str(), kernelContentSize);
  }
  MPI_Bcast(buf, kernelContentSize, MPI_CHAR, rankCompile, _comm.mpiComm);
  const std::string kernelFileContent(buf, kernelContentSize); // buf not null-terminated
  free(buf);

  return this->buildKernelFromStringMPI(kernelFileContent, kernelName, props, rankCompile);
}

occa::kernel
device_t::buildNativeKernel(const std::string &fileName,
                         const std::string &kernelName,
                         const occa::properties &props) const
{
  occa::properties nativeProperties = props;
  nativeProperties["okl/enabled"] = false;
  if(_verbose)
    nativeProperties["verbose"] = true;
  if(this->mode() == "OpenMP")
    nativeProperties["defines/__NEKRS__OMP__"] = 1;
  return _device.buildKernel(fileName, kernelName, nativeProperties);
}

occa::kernel
device_t::buildKernel(const std::string &fullPath,
                      const occa::properties &props) const
{
  const std::string noSuffix = std::string("");
  return this->buildKernel(fullPath, props, noSuffix);
}

occa::kernel
device_t::buildKernel(const std::string &fullPath,
                      const occa::properties &props,
                      const std::string & suffix) const
{
  const std::string fileName = fullPath;
  std::string kernelName;
  std::regex kernelNameRegex(R"((.+)\/(.+)\.)");
  std::smatch kernelNameMatch;
  const bool foundKernelName = std::regex_search(fullPath, kernelNameMatch, kernelNameRegex);

  // e.g. /path/to/install/nekrs/okl/cds/advectMeshVelocityHex3D.okl

  // Full string
  // 0:   /path/to/install/nekrs/okl/cds/advectMeshVelocityHex3D.okl

  // First capture group
  // 1:   /path/to/install/nekrs/okl/cds

  // Second capture group (kernel name)
  // 2:   advectMeshVelocityHex3D.okl
  if(foundKernelName){
    if(kernelNameMatch.size() == 3){
      kernelName = kernelNameMatch[2].str();
    }
  }

  return this->buildKernel(fileName, kernelName, props, suffix);
}

occa::kernel
device_t::buildKernel(const std::string &fileName,
                             const std::string &kernelName,
                             const occa::properties &props,
                             const std::string& suffix) const
{

  if(fileName.find(".okl") != std::string::npos){
    occa::properties propsWithSuffix = props;
    propsWithSuffix["kernelNameSuffix"] = suffix;
    if(_verbose)
      propsWithSuffix["verbose"] = true;

    if (this->mode() == "CUDA")
      propsWithSuffix["defines/smXX"] = 1;
    if (this->mode() == "HIP")
      propsWithSuffix["defines/gfxXX"] = 1;

    const std::string floatingPointType = static_cast<std::string>(propsWithSuffix["defines/dfloat"]);

    if (floatingPointType.find("float") != std::string::npos) {
      propsWithSuffix["defines/FP32"] = 1;
    }

    // if p_knl is defined, add _v(p_knl) to the kernel name
    std::string newKernelName = kernelName;
    if (props.has("defines/p_knl")) {
      const int kernelVariant = static_cast<int>(props["defines/p_knl"]);
      newKernelName += "_v" + std::to_string(kernelVariant);
    };

    return _device.buildKernel(fileName, newKernelName, propsWithSuffix);
  }
  else{
    occa::properties propsWithSuffix = props;
    propsWithSuffix["defines/SUFFIX"] = suffix;
    propsWithSuffix["defines/TOKEN_PASTE_(a,b)"] = std::string("a##b");
    propsWithSuffix["defines/TOKEN_PASTE(a,b)"] = std::string("TOKEN_PASTE_(a,b)");
    propsWithSuffix["defines/FUNC(a)"] = std::string("TOKEN_PASTE(a,SUFFIX)");
    const std::string alteredName =  kernelName + suffix;
    return this->buildNativeKernel(fileName, alteredName, propsWithSuffix);
  }
}

occa::kernel
device_t::buildKernel(const std::string &fileName,
                             const std::string &kernelName,
                             const occa::properties &props) const
{

  const std::string suffix("");
  const bool buildNodeLocal = useNodeLocalCache();
  const int rank = buildNodeLocal ? _comm.localRank : _comm.mpiRank;
  MPI_Comm localCommunicator = buildNodeLocal ? _comm.mpiCommLocal : _comm.mpiComm;

  occa::kernel constructedKernel;
  for(int pass = 0; pass < 2; ++pass){
    if((pass == 0 && rank == 0) || (pass == 1 && rank != 0)){
      constructedKernel = this->buildKernel(fileName, kernelName, props, suffix);
    }
    MPI_Barrier(localCommunicator);
  }
  return constructedKernel;

}

occa::kernel
device_t::buildKernel(const std::string &fullPath,
                         const occa::properties &props,
                         const std::string & suffix,
                         bool buildRank0) const
{

  if(buildRank0){

    const bool buildNodeLocal = useNodeLocalCache();
    const int rank = buildNodeLocal ? _comm.localRank : _comm.mpiRank;
    MPI_Comm localCommunicator = buildNodeLocal ? _comm.mpiCommLocal : _comm.mpiComm;
    occa::kernel constructedKernel;
    for(int pass = 0; pass < 2; ++pass){
      if((pass == 0 && rank == 0) || (pass == 1 && rank != 0)){
        constructedKernel = this->buildKernel(fullPath, props, suffix);
      }
      MPI_Barrier(localCommunicator);
    }
    return constructedKernel;

  }

  return this->buildKernel(fullPath, props, suffix);

}

occa::kernel
device_t::buildKernel(const std::string &fullPath,
                         const occa::properties &props,
                         bool buildRank0) const
{
  std::string noSuffix = std::string("");
  return this->buildKernel(fullPath, props, noSuffix, buildRank0);
}

occa::memory
device_t::mallocHost(const size_t Nbytes)
{
  occa::properties props;
  props["host"] = true;
  
  void* buffer = std::calloc(Nbytes, 1);
  occa::memory h_scratch = _device.malloc(Nbytes, buffer, props);
  std::free(buffer);
  return h_scratch;
}

occa::memory
device_t::malloc(const size_t Nbytes, const occa::properties& properties)
{
  void* buffer = std::calloc(Nbytes, 1);
  occa::memory o_returnValue = _device.malloc(Nbytes, buffer, properties);
  std::free(buffer);
  return o_returnValue;
}

occa::memory
device_t::malloc(const size_t Nbytes, const void* src, const occa::properties& properties)
{
  void* buffer;
  buffer = std::calloc(Nbytes, 1);
  const void* init_ptr = (src) ? src : buffer;
  occa::memory o_returnValue = _device.malloc(Nbytes, init_ptr, properties);
  std::free(buffer);
  return o_returnValue;
}

occa::memory
device_t::malloc(const hlong Nword , const dlong wordSize, occa::memory src)
{
  return _device.malloc(Nword * wordSize, src);
}

occa::memory
device_t::malloc(const hlong Nword , const dlong wordSize)
{
  const size_t Nbytes = Nword * wordSize;
  void* buffer = std::calloc(Nword, wordSize);
  occa::memory o_returnValue = _device.malloc(Nword * wordSize, buffer);
  std::free(buffer);
  return o_returnValue;
}

device_t::device_t(setupAide& options, comm_t& comm)
:_comm(comm)
{
  _verbose = options.compareArgs("BUILD ONLY", "TRUE");

  // OCCA build stuff
  char deviceConfig[BUFSIZ];
  int worldRank = _comm.mpiRank;

  int device_id = 0;

  if(options.compareArgs("DEVICE NUMBER", "LOCAL-RANK")) {
    device_id = _comm.localRank;
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
    if(worldRank == 0) printf("OpenMP backend currently not supported!\n");
    ABORT(EXIT_FAILURE);
    sprintf(deviceConfig, "{mode: 'OpenMP'}");
  }else if(strcasecmp(requestedOccaMode.c_str(), "CPU") == 0 ||
           strcasecmp(requestedOccaMode.c_str(), "SERIAL") == 0) {
    sprintf(deviceConfig, "{mode: 'Serial'}");
    options.setArgs("THREAD MODEL", "SERIAL");
    options.getArgs("THREAD MODEL", requestedOccaMode);
  } else {
    if(worldRank == 0) printf("Invalid requested backend!\n");
    ABORT(EXIT_FAILURE);
  }

  if(worldRank == 0) printf("Initializing device \n");
  this->_device.setup((std::string)deviceConfig);
 
  if(worldRank == 0)
    std::cout << "active occa mode: " << this->mode() << "\n\n";

  if(strcasecmp(requestedOccaMode.c_str(), this->mode().c_str()) != 0) {
    if(worldRank == 0) printf("active occa mode does not match selected backend!\n");
    ABORT(EXIT_FAILURE);
  } 

  // overwrite compiler settings to ensure
  // compatability of libocca and kernelLauchner 
  if(this->mode() != "Serial") {
    std::string buf;
    buf.assign(getenv("NEKRS_MPI_UNDERLYING_COMPILER"));
    setenv("OCCA_CXX", buf.c_str(), 1);
    buf.assign(getenv("NEKRS_CXXFLAGS"));
    setenv("OCCA_CXXFLAGS", buf.c_str(), 1);
  }

  _device_id = device_id;

  deviceAtomic = this->mode() == "CUDA";
}