#include <cstdio>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <map>

#include "nrs.hpp"
#include "platform.hpp"
#include "udf.hpp"

#define NOTBOUNDARY 0
#define DIRICHLET 1
#define NEUMANN 2

namespace {
enum class alignment_t { X, Y, Z, UNALIGNED };
std::string to_string(alignment_t a)
{
  switch (a) {
  case alignment_t::X:
    return "X";
  case alignment_t::Y:
    return "Y";
  case alignment_t::Z:
    return "Z";
  case alignment_t::UNALIGNED:
    return "UNALIGNED";
  }

  return "";
}
alignment_t computeAlignment(mesh_t *mesh, dlong element, dlong face)
{
  const dfloat alignmentTol = 1e-3;
  dfloat nxDiff = 0.0;
  dfloat nyDiff = 0.0;
  dfloat nzDiff = 0.0;

  for (int fp = 0; fp < mesh->Nfp; ++fp) {
    const dlong sid = mesh->Nsgeo * (mesh->Nfaces * mesh->Nfp * element + mesh->Nfp * face + fp);
    const dfloat nx = mesh->sgeo[sid + NXID];
    const dfloat ny = mesh->sgeo[sid + NYID];
    const dfloat nz = mesh->sgeo[sid + NZID];
    nxDiff += std::abs(std::abs(nx) - 1.0);
    nyDiff += std::abs(std::abs(ny) - 1.0);
    nzDiff += std::abs(std::abs(nz) - 1.0);
  }

  nxDiff /= mesh->Nfp;
  nyDiff /= mesh->Nfp;
  nzDiff /= mesh->Nfp;

  if (nxDiff < alignmentTol)
    return alignment_t::X;
  if (nyDiff < alignmentTol)
    return alignment_t::Y;
  if (nzDiff < alignmentTol)
    return alignment_t::Z;

  return alignment_t::UNALIGNED;
}
}

// stores for every (field, boundaryID) pair a bcID
static std::map<std::pair<std::string, int>, int> bToBc;
static int nbid[] = {0, 0};

static std::map<std::string, int> vBcTextToID = {
  {"periodic", 0},
  {"zerovalue", 1},
  {"fixedvalue", 2},
  {"zerogradient", 3},
  {"zeroxvalue/zerogradient", 4},
  {"zeroyvalue/zerogradient", 5},
  {"zerozvalue/zerogradient", 6},
  {"zeronvalue/zerogradient", 7}
};

static std::map<int, std::string> vBcIDToText = {
  {0, "periodic"               },
  {1, "zeroValue"              },
  {2, "fixedValue"             },
  {3, "zeroGradient"           },
  {4, "zeroXValue/zeroGradient"},
  {5, "zeroYValue/zeroGradient"},
  {6, "zeroZValue/zeroGradient"},
  {7, "zeroNValue/zeroGradient"}
};

static std::map<std::string, int> sBcTextToID = {
  {"periodic", 0},
  {"fixedvalue", 1},
  {"zerogradient", 2},
  {"fixedgradient", 3}
};

static std::map<int, std::string> sBcIDToText = {
  {0, "periodic"     },
  {1, "fixedValue"   },
  {2, "zeroGradient" },
  {3, "fixedGradient"}
};

static void v_setup(std::string s);
static void m_setup(std::string s);
static void s_setup(std::string s);

static void m_setup(std::string field, std::vector<std::string> slist)
{
  for(int i = 0; i < slist.size(); i++) {
    std::string key = slist[i];
    if (key.compare("p") == 0) key = "periodic";
    if (key.compare("w") == 0) key = "zerovalue";
    if (key.compare("wall") == 0) key = "zerovalue";
    if (key.compare("inlet") == 0) key = "fixedvalue";
    if (key.compare("v") == 0) key = "zerovalue"; // non-moving boundary, which is the same as a wall
    if (key.compare("mv") == 0) key = "fixedvalue";
    if (key.compare("outlet") == 0) key = "zerogradient";
    if (key.compare("outflow") == 0) key = "zerogradient";
    if (key.compare("o") == 0) key = "zerogradient";
    if (key.compare("slipx") == 0) key = "zeroxvalue/zerogradient";
    if (key.compare("slipy") == 0) key = "zeroyvalue/zerogradient";
    if (key.compare("slipz") == 0) key = "zerozvalue/zerogradient";
    if (key.compare("symx") == 0) key = "zeroxvalue/zerogradient";
    if (key.compare("symy") == 0) key = "zeroyvalue/zerogradient";
    if (key.compare("symz") == 0) key = "zerozvalue/zerogradient";
    if (key.compare("sym") == 0) key = "zeronvalue/zerogradient";

    if (vBcTextToID.find(key) == vBcTextToID.end()) {
      std::cout << "Invalid bcType " << "\'" << key << "\'" << "!\n";
      ABORT(1);
    }

    try
    {
      bToBc[make_pair(field, i)] = vBcTextToID.at(key);
    }
    catch (const std::out_of_range& oor)
    {
      std::cout << "Out of Range error: " << oor.what() << "!\n";
      ABORT(1);
    }
  }
}
static void v_setup(std::string field, std::vector<std::string> slist)
{
  for(int i = 0; i < slist.size(); i++) {
    std::string key = slist[i];
    if (key.compare("p") == 0) key = "periodic";
    if (key.compare("w") == 0) key = "zerovalue";
    if (key.compare("wall") == 0) key = "zerovalue";
    if (key.compare("inlet") == 0) key = "fixedvalue";
    if (key.compare("v") == 0 || key.compare("mv") == 0) key = "fixedvalue";
    if (key.compare("outlet") == 0) key = "zerogradient";
    if (key.compare("outflow") == 0) key = "zerogradient";
    if (key.compare("o") == 0) key = "zerogradient";
    if (key.compare("slipx") == 0) key = "zeroxvalue/zerogradient";
    if (key.compare("slipy") == 0) key = "zeroyvalue/zerogradient";
    if (key.compare("slipz") == 0) key = "zerozvalue/zerogradient";
    if (key.compare("symx") == 0) key = "zeroxvalue/zerogradient";
    if (key.compare("symy") == 0) key = "zeroyvalue/zerogradient";
    if (key.compare("symz") == 0) key = "zerozvalue/zerogradient";
    if (key.compare("sym") == 0) key = "zeronvalue/zerogradient";

    if (vBcTextToID.find(key) == vBcTextToID.end()) {
      std::cout << "Invalid bcType " << "\'" << key << "\'" << "!\n";
      ABORT(1);
    }

    try
    {
      bToBc[make_pair(field, i)] = vBcTextToID.at(key);
    }
    catch (const std::out_of_range& oor)
    {
      std::cout << "Out of Range error: " << oor.what() << "!\n";
      ABORT(1);
    }
  }
}

static void s_setup(std::string field, std::vector<std::string> slist)
{
  for(int i = 0; i < slist.size(); i++) {
    std::string key = slist[i];
    if (key.compare("p") == 0) key = "periodic";
    if (key.compare("t") == 0) key = "fixedvalue";
    if (key.compare("inlet") == 0) key = "fixedvalue";
    if (key.compare("flux") == 0) key = "fixedgradient";
    if (key.compare("f") == 0) key = "fixedgradient";
    if (key.compare("zeroflux") == 0) key = "zerogradient";
    if (key.compare("i") == 0) key = "zerogradient";
    if (key.compare("insulated") == 0) key = "zerogradient";
    if (key.compare("outflow") == 0) key = "zerogradient";
    if (key.compare("outlet") == 0) key = "zerogradient";
    if (key.compare("o") == 0) key = "zerogradient";

    if (sBcTextToID.find(key) == sBcTextToID.end()) {
      std::cout << "Invalid bcType " << "\'" << key << "\'" << "!\n";
      ABORT(1);
    }

    try
    {
      bToBc[make_pair(field, i)] = sBcTextToID.at(key);
    }
    catch (const std::out_of_range& oor)
    {
      std::cout << "Out of Range error: " << oor.what() << "!\n";
      ABORT(1);
    }
  }
}

namespace bcMap
{
void setup(std::vector<std::string> slist, std::string field)
{
  if (slist.size() == 0 || slist[0].compare("none") == 0) return;

  if (field.compare(0, 8, "scalar00") == 0) /* tmesh */ 
    nbid[1] = slist.size();
  else 
    nbid[0] = slist.size();

  if (field.compare("velocity") == 0)
    v_setup(field, slist);
  else if (field.compare("mesh") == 0)
    m_setup(field, slist);
  else if (field.compare(0, 6, "scalar") == 0)
    s_setup(field, slist);
}

int id(int bid, std::string field)
{
  if (bid < 1) return NOTBOUNDARY;

  return bToBc[{field, bid - 1}];
}

int type(int bid, std::string field)
{
  if (bid < 1) return NOTBOUNDARY;

  int bcType = -1;

  bool unalignedSYM = false;

  if (field.compare("x-velocity") == 0) {
    const int bcID = bToBc[{"velocity", bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = DIRICHLET;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 4) bcType = DIRICHLET;
    if (bcID == 5) bcType = NEUMANN;
    if (bcID == 6) bcType = NEUMANN;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
    if (bcID == 2) oudfFindDirichlet(field);
  } else if (field.compare("y-velocity") == 0) {
    const int bcID = bToBc[{"velocity", bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = DIRICHLET;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 4) bcType = NEUMANN;
    if (bcID == 5) bcType = DIRICHLET;
    if (bcID == 6) bcType = NEUMANN;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
    if (bcID == 2) oudfFindDirichlet(field);
  } else if (field.compare("z-velocity") == 0) {
    const int bcID = bToBc[{"velocity", bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = DIRICHLET;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 4) bcType = NEUMANN;
    if (bcID == 5) bcType = NEUMANN;
    if (bcID == 6) bcType = DIRICHLET;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
    if (bcID == 2) oudfFindDirichlet(field);
  } else if (field.compare("x-mesh") == 0) {
    const int bcID = bToBc[{"mesh", bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = DIRICHLET;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 4) bcType = DIRICHLET;
    if (bcID == 5) bcType = NEUMANN;
    if (bcID == 6) bcType = NEUMANN;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
  } else if (field.compare("y-mesh") == 0) {
    const int bcID = bToBc[{"mesh", bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = DIRICHLET;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 4) bcType = NEUMANN;
    if (bcID == 5) bcType = DIRICHLET;
    if (bcID == 6) bcType = NEUMANN;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
  } else if (field.compare("z-mesh") == 0) {
    const int bcID = bToBc[{"mesh", bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = DIRICHLET;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 4) bcType = NEUMANN;
    if (bcID == 5) bcType = NEUMANN;
    if (bcID == 6) bcType = DIRICHLET;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
  } else if (field.compare("pressure") == 0) {
    const int bcID = bToBc[{"velocity", bid - 1}];
    if (bcID == 1) bcType = NEUMANN;
    if (bcID == 2) bcType = NEUMANN;
    if (bcID == 3) bcType = DIRICHLET;
    if (bcID == 4) bcType = NEUMANN;
    if (bcID == 5) bcType = NEUMANN;
    if (bcID == 6) bcType = NEUMANN;
    if (bcID == 7)
      bcType = NEUMANN, unalignedSYM = true;
    if (bcID == 3) oudfFindDirichlet(field);
  } else if (field.compare(0, 6, "scalar") == 0) {
    const int bcID = bToBc[{field, bid - 1}];
    if (bcID == 1) bcType = DIRICHLET;
    if (bcID == 2) bcType = NEUMANN;
    if (bcID == 3) bcType = NEUMANN;
    if (bcID == 1) oudfFindDirichlet(field);
    if (bcID == 3) oudfFindNeumann(field);
  }

  if(bcType < 0) {
    std::cout << __func__ << "(): Unexpected error occured!" << std::endl;
    ABORT(1);
  }

  if (unalignedSYM && platform->options.compareArgs("STRESSFORMULATION", "FALSE")) {
    std::cout << __func__ << "(): un-aligned SYM boundary condition requires stress formulation!\n" << std::endl;
    ABORT(1);
  }

  return bcType;
}

std::string text(int bid, std::string field)
{
  if (bid < 1) return std::string();

  const int bcID = bToBc[{field, bid - 1}];
  if (field.compare("velocity") == 0 || field.compare("mesh") == 0)

    return vBcIDToText[bcID];

  else if (field.compare(0, 6, "scalar") == 0)

    return sBcIDToText[bcID];


  std::cout << __func__ << "(): Unexpected error occured!" << std::endl;
  ABORT(1);
  return 0;
}

int size(int isTmesh)
{
  return isTmesh ? nbid[1] : nbid[0];
}

void check(mesh_t* mesh)
{
  
  int nid = nbid[0];
  if(mesh->cht) nid = nbid[1];

  int err = 0;
  int found = 0;

  for (int id = 1; id <= nid; id++) {
    found = 0;
    for (int f = 0; f < mesh->Nelements * mesh->Nfaces; f++) {
      if (mesh->EToB[f] == id) {
        found = 1;
        break;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
    err += (found ? 0 : 1);
    if (err && platform->comm.mpiRank == 0) 
      printf("Cannot find boundary ID %d in mesh!\n", id);
  }
  if (err) EXIT_AND_FINALIZE(EXIT_FAILURE);

  found = 0;
  for (int f = 0; f < mesh->Nelements * mesh->Nfaces; f++)
    if (mesh->EToB[f] < -1 || mesh->EToB[f] == 0 || mesh->EToB[f] > nid) found = 1;
  MPI_Allreduce(MPI_IN_PLACE, &found, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
  if (found) {
    if (platform->comm.mpiRank == 0) printf("Mesh has unmapped boundary IDs!\n");
    EXIT_AND_FINALIZE(EXIT_FAILURE);
  }


}

void setBcMap(std::string field, int* map, int nIDs)
{
  if (field.compare(0, 8, "scalar00") == 0)
    nbid[1] = nIDs;
  else
    nbid[0] = nIDs;

  try
  {
    for(int i = 0; i < nIDs; i++) bToBc[make_pair(field, i)] = map[i];
  }
  catch (const std::out_of_range& oor)
  {
    std::cout << "Out of Range error: " << oor.what() << "!\n";
    ABORT(1);
  }
}
void checkBoundaryAlignment(mesh_t *mesh, std::string field)
{
  if (field != std::string("velocity") && field != std::string("mesh"))
    return;

  std::ostringstream errorLogger;

  for (int e = 0; e < mesh->Nelements; e++) {
    for (int f = 0; f < mesh->Nfaces; f++) {
      int bc = id(mesh->EToB[f + e * mesh->Nfaces], field);
      if (bc == 4 || bc == 5 || bc == 6) {
        auto expectedAlignment = alignment_t::UNALIGNED;
        switch (bc) {
        case 4:
          expectedAlignment = alignment_t::X;
          break;
        case 5:
          expectedAlignment = alignment_t::Y;
          break;
        case 6:
          expectedAlignment = alignment_t::Z;
          break;
        }

        auto alignment = computeAlignment(mesh, e, f);
        if (alignment != expectedAlignment) {
          errorLogger << "On rank: " << platform->comm.mpiRank << " element: " << e << " face: " << f
                      << " has an unexpected alignment.\n";
          errorLogger << "\t"
                      << "alignment " << to_string(alignment) << " != expected alignment "
                      << to_string(expectedAlignment) << "\n";
        }
        if (alignment == alignment_t::UNALIGNED) {
          errorLogger << "On rank: " << platform->comm.mpiRank << " element: " << e << " face: " << f
                      << " is unaligned, despite being marked as an aligned boundary!\n";
        }
      }
    }
  }

  std::string errors = errorLogger.str();
  int errorLength = errors.size();

  MPI_Allreduce(MPI_IN_PLACE, &errorLength, 1, MPI_INT, MPI_MAX, platform->comm.mpiComm);
  if (errorLength > 0) {
    for (int rank = 0; rank < platform->comm.mpiCommSize; rank++) {
      if (platform->comm.mpiRank == rank) {
        std::cout << errors;
      }
      std::cout << std::endl;
      MPI_Barrier(platform->comm.mpiComm);
    }
    ABORT(1);
  }
}

void remapUnalignedBoundaries(mesh_t *mesh, std::string field)
{
  if (field != std::string("velocity") && field != std::string("mesh"))
    return;

  std::map<int, bool> remapBID;
  std::map<int, alignment_t> alignmentBID;

  int nid = nbid[0];
  if (mesh->cht)
    nid = nbid[1];

  for (int bid = 1; bid <= nid; ++bid) {
    int bcType = id(bid, field);
    remapBID[bid] = (bcType == 7);
  }

  for (int e = 0; e < mesh->Nelements; e++) {
    for (int f = 0; f < mesh->Nfaces; f++) {
      int bid = mesh->EToB[f + e * mesh->Nfaces];
      int bc = id(bid, field);
      auto alignment = computeAlignment(mesh, e, f);
      if (alignmentBID.count(bid) == 0) {
        alignmentBID[bid] = alignment;
      }

      auto previousAlignment = alignmentBID[bid];
      remapBID[bid] &= (alignment != alignment_t::UNALIGNED) && (alignment == previousAlignment);
    }
  }

  for (int bid = 1; bid <= nid; ++bid) {
    int canRemap = remapBID[bid];
    MPI_Allreduce(MPI_IN_PLACE, &canRemap, 1, MPI_INT, MPI_MIN, platform->comm.mpiComm);
    if (canRemap) {

      auto alignmentType = alignmentBID[bid];

      int newBcType = 0;
      switch (alignmentType) {
      case alignment_t::X:
        newBcType = 4;
        break;
      case alignment_t::Y:
        newBcType = 5;
        break;
      case alignment_t::Z:
        newBcType = 6;
        break;
      default:
        break;
      }

      bToBc[{field, bid - 1}] = newBcType;
    }
  }
}
} // namespace
