#include "cds.hpp"
#include "avm.hpp"
#include <string>

/**
 * Artificial viscosity method (https://arxiv.org/pdf/1810.02152.pdf), [0,1] viscosity weighting currently deemed as not useful
 **/

namespace avm {

static occa::kernel filterScalarNormKernel;
static occa::kernel applyAVMKernel;
static occa::kernel computeMaxViscKernel;
static occa::kernel computeLengthScaleKernel;

static occa::memory o_artVisc;
static occa::memory o_diffOld; // diffusion from initial state

static bool setProp = false;

void allocateMemory(cds_t* cds)
{
  if(!setProp) o_diffOld = platform->device.malloc(cds->fieldOffsetSum, sizeof(dfloat), cds->o_diff);
}

void compileKernels(cds_t* cds)
{
  std::string install_dir;
  install_dir.assign(getenv("NEKRS_INSTALL_DIR"));
  const std::string oklpath = install_dir + "/okl/cds/regularization/";
  std::string filename = oklpath + "filterScalarNorm.okl";
  occa::properties info = platform->kernelInfo;
  info["defines/" "p_Nq"] = cds->mesh[0]->Nq;
  info["defines/" "p_Np"] = cds->mesh[0]->Np;
  filterScalarNormKernel =
    platform->device.buildKernel(filename,
                             "filterScalarNorm",
                             info);

  filename = oklpath + "applyAVM.okl";
  applyAVMKernel =
    platform->device.buildKernel(filename,
                             "applyAVM",
                             info);

  filename = oklpath + "computeMaxVisc.okl";
  computeMaxViscKernel =
    platform->device.buildKernel(filename,
                             "computeMaxVisc",
                             info);

  filename = oklpath + "computeLengthScale.okl";
  computeLengthScaleKernel =
    platform->device.buildKernel(filename,
                             "computeLengthScale",
                             info);
}

void setup(cds_t* cds, bool userSetProperties)
{
  setProp = userSetProperties;
  allocateMemory(cds);
  compileKernels(cds);
}

occa::memory computeEps(cds_t* cds, const dfloat time, const dlong scalarIndex, occa::memory o_S)
{
  dfloat sensorSensitivity = 1.0;
  cds->options[scalarIndex].getArgs("SENSOR SENSITIVITY", sensorSensitivity);
  mesh_t* mesh = cds->mesh[scalarIndex];
  int Nblock = (cds->mesh[scalarIndex]->Nlocal+BLOCKSIZE-1)/BLOCKSIZE;

  occa::memory& o_logShockSensor = platform->o_mempool.slice0;
  occa::memory& o_elementLengths = platform->o_mempool.slice1;

  // artificial viscosity magnitude
  occa::memory o_epsilon = platform->o_mempool.slice2;

  const dfloat p = mesh->N;
  
  filterScalarNormKernel(
    mesh->Nelements,
    scalarIndex,
    cds->fieldOffsetScan[scalarIndex],
    cds->o_filterMT,
    sensorSensitivity,
    p,
    o_S,
    o_logShockSensor
  );

  // see footnote after eq 63
  const dfloat logReferenceSensor = -2.0;

  const dfloat rampParameter = 1.0;

  computeLengthScaleKernel(
    mesh->Nelements,
    mesh->o_x,
    mesh->o_y,
    mesh->o_z,
    o_elementLengths // <- min element lengths
  );

  dfloat coeff = 0.5;
  cds->options[scalarIndex].getArgs("COEFF AVM", coeff);

  computeMaxViscKernel(
    mesh->Nelements,
    cds->vFieldOffset,
    logReferenceSensor,
    rampParameter,
    coeff,
    o_elementLengths, // h_e
    cds->o_U,
    o_logShockSensor,
    o_epsilon // max(|df/du|) <- max visc
  );

  return o_epsilon;

}

void apply(cds_t* cds, const dfloat time, const dlong scalarIndex, occa::memory o_S)
{
  mesh_t* mesh = cds->mesh[scalarIndex];
  const dlong scalarOffset = cds->fieldOffsetScan[scalarIndex];
  if(!setProp) // restore viscosity from previous state
    cds->o_diff.copyFrom(o_diffOld,
      cds->fieldOffset[scalarIndex] * sizeof(dfloat),
      cds->fieldOffsetScan[scalarIndex] * sizeof(dfloat),
      cds->fieldOffsetScan[scalarIndex] * sizeof(dfloat)
    );
  occa::memory o_eps = computeEps(cds, time, scalarIndex, o_S);

  occa::memory& o_avm = platform->o_mempool.slice0;
  platform->linAlg->fill(cds->fieldOffset[scalarIndex], 0.0, o_avm);
  applyAVMKernel(
    mesh->Nelements,
    scalarOffset,
    scalarIndex,
    o_eps,
    cds->o_diff
  );
}

} // namespace