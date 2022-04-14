#if !defined(createEToBV_hpp_)
#define createEToBV_hpp_

#include "occa.hpp"

// pre: EToB allocated, capacity mesh->Nfaces * mesh->Nelements dlong words,
//      o_EToBV allocated, capacity mesh->Nlocal dlong words
struct mesh_t;
void createEToBV(const mesh_t* mesh, const int* EToB, occa::memory& o_EToBV);

#endif