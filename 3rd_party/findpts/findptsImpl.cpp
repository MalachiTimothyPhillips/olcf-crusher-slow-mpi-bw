#include <stdlib.h>
#include <math.h>
#include "gslib.h"

#include "findptsTypes.h"
#include "findptsImpl.hpp"

#include <vector>

#define D 3

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

static slong lfloor(double x) { return floor(x); }
static slong lceil (double x) { return ceil (x); }

static ulong hash_index_aux(double low, double fac, ulong n, double x)
{
  const slong i = lfloor((x-low)*fac);
  return i<0 ? 0 : (n-1<(ulong)i ? n-1 : (ulong)i);
}

static ulong hash_index_3(const hashData_t *p, const double x[D])
{
  const ulong n = p->hash_n;
  return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
          hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
             n +
         hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

struct srcPt_t {
  double x[D];
  int index, proc;
};
struct outPt_t {
  double r[D], dist2;
  int index, code, el, proc;
};

void findpts_impl(int *const code_base,
                   int *const proc_base,
                   int *const el_base,
                   double *const r_base,
                   double *const dist2_base,
                   const double *const x_base[D],
                   const int npt,
                   gslibFindptsData_t *const fd,
                   const void *const findptsData)
{

  const int np = fd->cr.comm.np, id=fd->cr.comm.id;
  struct array hash_pt, srcPt_t, outPt_t;
  /* look locally first */
  if (npt){
    findpts_local(code_base,
                        el_base,
                        r_base,
                        dist2_base,
                        x_base,
                        npt,
                        findptsData);
  }
  /* send unfound and border points to global hash cells */
  {
    int index;
    int *code=code_base, *proc=proc_base;
    const double *xp[D];
    struct srcPt_t *pt;

    for(int d=0;d<D;++d) {
      xp[d]=x_base[d];
    }
    array_init(struct srcPt_t, &hash_pt, npt);
    pt = (struct srcPt_t*) hash_pt.ptr;

    double x[D];

    for(index=0;index<npt;++index) {
      for(int d=0;d<D;++d) {
        x[d]=*xp[d];
      }
      *proc = id;
      if(*code!=CODE_INTERNAL) {
        const int hi = hash_index_3(&fd->hash, x);
        for(int d=0;d<D;++d) {
          pt->x[d]=x[d];
        }
        pt->index=index;
        pt->proc=hi%np;
        ++pt;
      }
      for(int d=0;d<D;++d){
        xp[d]++;
      }
      code++;
      proc++;
    }
    hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
    sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, &fd->cr);
  }
  /* look up points in hash cells, route to possible procs */
  {
    const unsigned int *const hash_offset = fd->hash.offset;
    int count=0, *proc, *proc_p;
    const struct srcPt_t *p = (struct srcPt_t*) hash_pt.ptr, *const pe = p + hash_pt.n;
    struct srcPt_t *q;
    for(;p!=pe;++p) {
      const int hi = hash_index_3(&fd->hash, p->x) / np;
      const int i = hash_offset[hi], ie = hash_offset[hi+1];
      count += ie-i;
    }
    proc = tmalloc(int,count);
    proc_p = proc;
    array_init(struct srcPt_t, &srcPt_t, count), q = (struct srcPt_t*) srcPt_t.ptr;
    p = (struct srcPt_t *) hash_pt.ptr;
    for(;p!=pe;++p) {
      const int hi = hash_index_3(&fd->hash, p->x) / np;
      int i = hash_offset[hi]; const int ie = hash_offset[hi+1];
      for(;i!=ie;++i) {
        const int pp = hash_offset[i];
        if(pp==p->proc) continue; /* don't send back to source proc */
        *proc_p++ = pp;
        *q++ = *p;
      }
    }
    array_free(&hash_pt);
    srcPt_t.n = proc_p - proc;
#ifdef DIAGNOSTICS
    printf("(proc %u) hashed; routing %u/%u\n", id, (int)srcPt_t.n, count);
#endif
    sarray_transfer_ext(struct srcPt_t, &srcPt_t, reinterpret_cast<unsigned int*>(proc), sizeof(int), &fd->cr);
    free(proc);
  }
  /* look for other procs' points, send back */
  {
    int n = srcPt_t.n;
    const struct srcPt_t *spt;
    struct outPt_t *opt;
    array_init(struct outPt_t, &outPt_t, n);
    outPt_t.n = n;
    spt = (struct srcPt_t*) srcPt_t.ptr;
    opt = (struct outPt_t*) outPt_t.ptr;
    for(;n;--n,++spt,++opt) {
      opt->index=spt->index;
      opt->proc=spt->proc;
    }
    spt = (struct srcPt_t*) srcPt_t.ptr;
    opt = (struct outPt_t*) outPt_t.ptr;
    if (srcPt_t.n) {

      // result buffers
      std::vector<int> codeArr(srcPt_t.n, 0);
      std::vector<int> elArr(srcPt_t.n, 0);
      std::vector<double> rArr(3*srcPt_t.n, 0.0);
      std::vector<double> dist2Arr(srcPt_t.n, 0.0);

      std::vector<double> x0(srcPt_t.n, 0.0);
      std::vector<double> x1(srcPt_t.n, 0.0);
      std::vector<double> x2(srcPt_t.n, 0.0);

      // pack position data into arrays
      double* spt_x_base[3] = {x0.data(), x1.data(), x2.data()};

      for(int point = 0; point < srcPt_t.n; ++point){
        for(int d = 0; d < 3; ++d){
            spt_x_base[d][point] = spt[point].x[d];
        }
      }

      findpts_local(codeArr.data(),
                          elArr.data(),
                          rArr.data(),
                          dist2Arr.data(),
                          spt_x_base,
                          srcPt_t.n,
                          findptsData);

      // unpack arrays into opt
      for(int point = 0; point < srcPt_t.n; point++){
        opt[point].code = codeArr[point];
        opt[point].el = elArr[point];
        opt[point].dist2 = dist2Arr[point];
        for(int d = 0; d < D; ++d){
          opt[point].r[d] = rArr[3*point + d];
        }
      }
    }
    array_free(&srcPt_t);
    /* group by code to eliminate unfound points */
    sarray_sort(struct outPt_t, opt, outPt_t.n, code, 0, &fd->cr.data);
    n = outPt_t.n;
    while (n && opt[n - 1].code == CODE_NOT_FOUND)
      --n;
    outPt_t.n = n;
#ifdef DIAGNOSTICS
    printf("(proc %u) sending back %u found points\n", id, (int)outPt_t.n);
#endif
    sarray_transfer(struct outPt_t, &outPt_t, proc, 1, &fd->cr);
  }
  /* merge remote results with user data */
  {
    int n = outPt_t.n;
    struct outPt_t *opt = (struct outPt_t*) outPt_t.ptr;
    for (; n; --n, ++opt) {
      const int index = opt->index;
      if(code_base[index]==CODE_INTERNAL) continue;
      if(code_base[index]==CODE_NOT_FOUND
         || opt->code==CODE_INTERNAL
         || opt->dist2<dist2_base[index]) {
        for(int d=0;d<D;++d) {
          r_base[3*index + d]=opt->r[d];
        }
        dist2_base[index] = opt->dist2;
        proc_base[index] = opt->proc;
        el_base[index] = opt->el;
        code_base[index] = opt->code;
      }
    }
    array_free(&outPt_t);
  }
}

void findpts_eval_impl(double *const out_base,
                        const int *const code_base,
                        const int *const proc_base,
                        const int *const el_base,
                        const double *const r_base,
                        const int npt,
                        const void *const in,
                        gslibFindptsData_t *const fd,
                        const void *const findptsData)
{
  struct array src, outpt;
  /* copy user data, weed out unfound points, send out */
  {
    int index;
    const int *code=code_base, *proc=proc_base, *el=el_base;
    const double *r=r_base;
    evalSrcPt_t *pt;
    array_init(evalSrcPt_t, &src, npt);
    pt = (evalSrcPt_t*)src.ptr;
    for(index=0;index<npt;++index) {
      if(*code!=CODE_NOT_FOUND) {
        for(int d=0;d<D;++d) {
          pt->r[d]=r[d];
        }
        pt->index=index;
        pt->proc=*proc;
        pt->el=*el;
        ++pt;
      }
      r += D;
      code++;
      proc++;
      el++;
    }
    src.n = pt - (evalSrcPt_t *)src.ptr;
    sarray_transfer(evalSrcPt_t, &src, proc, 1, &fd->cr);
  }
  /* evaluate points, send back */
  {
    int n=src.n;
    const evalSrcPt_t *spt;
    evalOutPt_t *opt;
    /* group points by element */
    sarray_sort(evalSrcPt_t, src.ptr, n, el, 0, &fd->cr.data);
    array_init(evalOutPt_t, &outpt, n);
    outpt.n = n;
    spt=(evalSrcPt_t*)src.ptr;
    opt=(evalOutPt_t*) outpt.ptr;
    findpts_local_eval_internal(opt, spt, src.n, in, findptsData);
    spt=(evalSrcPt_t*)src.ptr;
    opt=(evalOutPt_t*)outpt.ptr;
    for(;n;--n,++spt,++opt) {
      opt->index=spt->index;
      opt->proc=spt->proc;
    }
    array_free(&src);
    sarray_transfer(evalOutPt_t, &outpt, proc, 1, &fd->cr);
  }
  /* copy results to user data */
  {
    int n=outpt.n;
    evalOutPt_t *opt = (evalOutPt_t*) outpt.ptr;
    for(;n;--n,++opt) {
      out_base[opt->index]=opt->out;
    }
    array_free(&outpt);
  }
}