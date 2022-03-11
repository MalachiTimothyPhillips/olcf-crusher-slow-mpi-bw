#include <stdlib.h>
#include <math.h>
#include "gslib.h"

#include "findptsTypes.h"
#include "internal_findpts.h"

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

#if 0
struct findpts_hash_data_3 {
  ulong hash_n;
  struct dbl_range bnd[D];
  double fac[D];
  int *offset;
};
#endif

static ulong hash_index_3(const struct hash_data_3 *p, const double x[D])
{
  const ulong n = p->hash_n;
  return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
          hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
             n +
         hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

struct src_pt_3 {
  double x[D];
  int index, proc;
};
struct out_pt_3 {
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
                   struct findpts_data_3 *const fd,
                   const void *const findptsData)
{

  const int np = fd->cr.comm.np, id=fd->cr.comm.id;
  struct array hash_pt, src_pt_3, out_pt_3;
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
    struct src_pt_3 *pt;
    int d; for(d=0;d<D;++d) xp[d]=x_base[d];
    array_init(struct src_pt_3, &hash_pt, npt), pt = (struct src_pt_3*) hash_pt.ptr;
    for(index=0;index<npt;++index) {
      double x[D]; for(d=0;d<D;++d) x[d]=*xp[d];
      *proc = id;
      if(*code!=CODE_INTERNAL) {
        const int hi = hash_index_3(&fd->hash, x);
        int d;
        for(d=0;d<D;++d) pt->x[d]=x[d];
        pt->index=index;
        pt->proc=hi%np;
        ++pt;
      }
      for(d=0;d<D;++d){
        xp[d]++;
      }
      code++;
      proc++;
    }
    hash_pt.n = pt - (struct src_pt_3 *)hash_pt.ptr;
    sarray_transfer(struct src_pt_3, &hash_pt, proc, 1, &fd->cr);
  }
  /* look up points in hash cells, route to possible procs */
  {
    const unsigned int *const hash_offset = fd->hash.offset;
    int count=0, *proc, *proc_p;
    const struct src_pt_3 *p = (struct src_pt_3*) hash_pt.ptr, *const pe = p + hash_pt.n;
    struct src_pt_3 *q;
    for(;p!=pe;++p) {
      const int hi = hash_index_3(&fd->hash, p->x) / np;
      const int i = hash_offset[hi], ie = hash_offset[hi+1];
      count += ie-i;
    }
    proc_p = proc = tmalloc(int,count);
    array_init(struct src_pt_3, &src_pt_3, count), q = (struct src_pt_3*) src_pt_3.ptr;
    p = (struct src_pt_3 *) hash_pt.ptr;
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
    src_pt_3.n = proc_p - proc;
#ifdef DIAGNOSTICS
    printf("(proc %u) hashed; routing %u/%u\n", id, (int)src_pt_3.n, count);
#endif
    sarray_transfer_ext(struct src_pt_3, &src_pt_3, reinterpret_cast<unsigned int*>(proc), sizeof(int), &fd->cr);
    free(proc);
  }
  /* look for other procs' points, send back */
  {
    int n = src_pt_3.n;
    const struct src_pt_3 *spt;
    struct out_pt_3 *opt;
    array_init(struct out_pt_3, &out_pt_3, n), out_pt_3.n = n;
    spt = (struct src_pt_3*) src_pt_3.ptr, opt = (struct out_pt_3*) out_pt_3.ptr;
    for(;n;--n,++spt,++opt) opt->index=spt->index,opt->proc=spt->proc;
    spt = (struct src_pt_3*) src_pt_3.ptr, opt = (struct out_pt_3*) out_pt_3.ptr;
    if (src_pt_3.n) {

      // result buffers
      int* codeArr = (int*) calloc(src_pt_3.n, sizeof(int));
      int* elArr = (int*) calloc(src_pt_3.n, sizeof(int));
      double* rArr = (double*) calloc(3*src_pt_3.n, sizeof(double));
      double* dist2Arr = (double*) calloc(src_pt_3.n, sizeof(double));

      // pack position data into arrays
      double* spt_x_base[3];
      spt_x_base[0] = (double*) calloc(src_pt_3.n, sizeof(double));
      spt_x_base[1] = (double*) calloc(src_pt_3.n, sizeof(double));
      spt_x_base[2] = (double*) calloc(src_pt_3.n, sizeof(double));

      for(int point = 0; point < src_pt_3.n; ++point){
        for(int d = 0; d < 3; ++d){
            spt_x_base[d][point] = spt[point].x[d];
        }
      }

      findpts_local(codeArr,
                          elArr,
                          rArr,
                          dist2Arr,
                          spt_x_base,
                          src_pt_3.n,
                          findptsData);

      // unpack arrays into opt
      for(int point = 0; point < src_pt_3.n; point++){
        opt[point].code = codeArr[point];
        opt[point].el = elArr[point];
        opt[point].dist2 = dist2Arr[point];
        for(int d = 0; d < D; ++d){
          opt[point].r[d] = rArr[3*point + d];
        }
      }

      free(codeArr);
      free(elArr);
      free(rArr);
      free(dist2Arr);
      free(spt_x_base[0]);
      free(spt_x_base[1]);
      free(spt_x_base[2]);

    }
    array_free(&src_pt_3);
    /* group by code to eliminate unfound points */
    sarray_sort(struct out_pt_3, opt, out_pt_3.n, code, 0, &fd->cr.data);
    n = out_pt_3.n;
    while (n && opt[n - 1].code == CODE_NOT_FOUND)
      --n;
    out_pt_3.n = n;
#ifdef DIAGNOSTICS
    printf("(proc %u) sending back %u found points\n", id, (int)out_pt_3.n);
#endif
    sarray_transfer(struct out_pt_3, &out_pt_3, proc, 1, &fd->cr);
  }
  /* merge remote results with user data */
  {
    int n = out_pt_3.n;
    struct out_pt_3 *opt = (struct out_pt_3*) out_pt_3.ptr;
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
    array_free(&out_pt_3);
  }
}

void findpts_eval_impl(double *const out_base,
                        const int *const code_base,
                        const int *const proc_base,
                        const int *const el_base,
                        const double *const r_base,
                        const int npt,
                        const void *const in,
                        struct findpts_data_3 *const fd,
                        const void *const findptsData)
{
  struct array src, outpt;
  /* copy user data, weed out unfound points, send out */
  {
    int index;
    const int *code=code_base, *proc=proc_base, *el=el_base;
    const double *r=r_base;
    struct eval_src_pt_3 *pt;
    array_init(struct eval_src_pt_3, &src, npt), pt = (struct eval_src_pt_3*)src.ptr;
    for(index=0;index<npt;++index) {
      if(*code!=CODE_NOT_FOUND) {
        int d;
        for(d=0;d<D;++d) pt->r[d]=r[d];
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
    src.n = pt - (struct eval_src_pt_3 *)src.ptr;
    sarray_transfer(struct eval_src_pt_3, &src, proc, 1, &fd->cr);
  }
  /* evaluate points, send back */
  {
    int n=src.n;
    const struct eval_src_pt_3 *spt;
    struct eval_out_pt_3 *opt;
    /* group points by element */
    sarray_sort(struct eval_src_pt_3, src.ptr, n, el, 0, &fd->cr.data);
    array_init(struct eval_out_pt_3, &outpt, n), outpt.n = n;
    spt=(struct eval_src_pt_3*)src.ptr, opt=(struct eval_out_pt_3*) outpt.ptr;
    findpts_local_eval_internal(opt, spt, src.n, in, &fd->local, findptsData);
    spt=(struct eval_src_pt_3*)src.ptr, opt=(struct eval_out_pt_3*)outpt.ptr;
    for(;n;--n,++spt,++opt) opt->index=spt->index,opt->proc=spt->proc;
    array_free(&src);
    sarray_transfer(struct eval_out_pt_3, &outpt, proc, 1, &fd->cr);
  }
  /* copy results to user data */
  {
    int n=outpt.n;
    struct eval_out_pt_3 *opt = (struct eval_out_pt_3*) outpt.ptr;
    for(;n;--n,++opt) {
      out_base[opt->index]=opt->out;
    }
    array_free(&outpt);
  }
}