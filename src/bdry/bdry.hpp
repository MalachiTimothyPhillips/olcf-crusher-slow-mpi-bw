#if !defined(BDRY_HPP)
#define BDARY_HPP

class nrs_t;
void applyDirichletVelocity(nrs_t *nrs, double time, occa::memory& o_U, occa::memory& o_p);
void applyDirichletScalars(nrs_t *nrs, double time, occa::memory& o_S);
void applyDirichletMeshVelocity(nrs_t *nrs, double time, occa::memory& o_U);
void createEToBV(const mesh_t* mesh, const int* EToB, occa::memory& o_EToBV);
void createZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory& o_EToBV, occa::memory &o_mask);
void applyZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory &o_mask, occa::memory &o_x);
void applyZeroNormalMask(nrs_t *nrs,
                         dlong Nelements,
                         occa::memory &o_elementList,
                         occa::memory &o_EToB,
                         occa::memory &o_mask,
                         occa::memory &o_x);

#endif
