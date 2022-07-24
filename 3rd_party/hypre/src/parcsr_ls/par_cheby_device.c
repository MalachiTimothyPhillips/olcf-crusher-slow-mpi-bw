/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Chebyshev setup and solve Device
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "float.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#include "_hypre_utilities.hpp"
#include <algorithm>
#include <vector>

namespace{
std::vector<HYPRE_Real>
optimalWeightsImpl (const int numIters)
{
  if (numIters == 0){
    return {};
  }
  if (numIters == 1){
    return {
        1.12500000000000,
    };
  }

  if (numIters == 2){
    return {
        1.02387287570313,
        1.26408905371085,
    };
  }

  if (numIters == 3){
    return {
        1.00842544782028,
        1.08867839208730,
        1.33753125909618,
    };
  }

  if (numIters == 4){
    return {
        1.00391310427285,
        1.04035811188593,
        1.14863498546254,
        1.38268869241000,
    };
  }

  if (numIters == 5){
    return {
        1.00212930146164,
        1.02173711549260,
        1.07872433192603,
        1.19810065292663,
        1.41322542791682,
    };
  }

  if (numIters == 6){
    return {
        1.00128517255940,
        1.01304293035233,
        1.04678215124113,
        1.11616489419675,
        1.23829020218444,
        1.43524297106744,
    };
  }

  if (numIters == 7){
    return {
        1.00083464397912,
        1.00843949430122,
        1.03008707768713,
        1.07408384092003,
        1.15036186707366,
        1.27116474046139,
        1.45186658649364,
    };
  }

  if (numIters == 8){
    return {
        1.00057246631197,
        1.00577427662415,
        1.02050187922941,
        1.05019803444565,
        1.10115572984941,
        1.18086042806856,
        1.29838585382576,
        1.46486073151099,
    };
  }

  if (numIters == 9){
    return {
        1.00040960072832,
        1.00412439506106,
        1.01460212148266,
        1.03561113626671,
        1.07139972529194,
        1.12688273710962,
        1.20785219140729,
        1.32121930716746,
        1.47529642820699,
    };
  }

  if (numIters == 10){
    return {
        1.00030312229652,
        1.00304840660796,
        1.01077022715387,
        1.02619011597640,
        1.05231724933755,
        1.09255743207549,
        1.15083376663972,
        1.23172250870894,
        1.34060802024460,
        1.48386124407011,
    };
  }

  if (numIters == 11){
    return {
        1.00023058595209,
        1.00231675024028,
        1.00817245396304,
        1.01982986566342,
        1.03950210235324,
        1.06965042700541,
        1.11305754295742,
        1.17290876275564,
        1.25288300576792,
        1.35725579919519,
        1.49101672564139,
    };
  }

  if (numIters == 12){
    return {
        1.00017947200828,
        1.00180189139619,
        1.00634861907307,
        1.01537864566306,
        1.03056942830760,
        1.05376019693943,
        1.08699862592072,
        1.13259183097913,
        1.19316273358172,
        1.27171293675110,
        1.37169337969799,
        1.49708418575562,
    };
  }

  if (numIters == 13){
    return {
        1.00014241921559,
        1.00142906932629,
        1.00503028986298,
        1.01216910518495,
        1.02414874342792,
        1.04238158880820,
        1.06842008128700,
        1.10399010936759,
        1.15102748242645,
        1.21171811910125,
        1.28854264865128,
        1.38432619380991,
        1.50229418757368,
    };
  }

  if (numIters == 14){
    return {
        1.00011490538261,
        1.00115246376914,
        1.00405357333264,
        1.00979590573153,
        1.01941300472994,
        1.03401425035436,
        1.05480599606629,
        1.08311420301813,
        1.12040891660892,
        1.16833095655446,
        1.22872122288238,
        1.30365305707817,
        1.39546814053678,
        1.50681646209583,
    };
  }

  if (numIters == 15){
    return {
        1.00009404750752,
        1.00094291696343,
        1.00331449056444,
        1.00800294833816,
        1.01584236259140,
        1.02772083317705,
        1.04459535422831,
        1.06750761206125,
        1.09760092545889,
        1.13613855366157,
        1.18452361426236,
        1.24432087304475,
        1.31728069083392,
        1.40536543893560,
        1.51077872501845,
    };
  }

  if (numIters == 16){
    return {
        1.00007794828179,
        1.00078126847253,
        1.00274487974401,
        1.00662291017015,
        1.01309858836971,
        1.02289448329337,
        1.03678321409983,
        1.05559875719896,
        1.08024848405560,
        1.11172607131497,
        1.15112543431072,
        1.19965584614973,
        1.25865841744946,
        1.32962412656664,
        1.41421360695576,
        1.51427891730346,
    };
  }

  return {};
}
}

/**
 * @brief waxpyz
 *
 * Performs
 * w = a*x+y.*z
 * For scalars w,x,y,z and constant a (indices 0, 1, 2, 3 respectively)
 */
template <typename T>
struct waxpyz
{
   typedef thrust::tuple<T &, T, T, T> Tuple;

   const T scale;
   waxpyz(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = scale * thrust::get<1>(t) + thrust::get<2>(t) * thrust::get<3>(t);
   }
};

/**
 * @brief wxypz
 *
 * Performs
 * o = x * (y .+ z)
 * For scalars o,x,y,z (indices 0, 1, 2, 3 respectively)
 */
template <typename T>
struct wxypz
{
   typedef thrust::tuple<T &, T, T, T> Tuple;
   __host__ __device__ void            operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t) * (thrust::get<2>(t) + thrust::get<3>(t));
   }
};
/**
 * @brief Saves u into o, then scales r placing the result in u
 *
 * Performs
 * o = u
 * u = r * a
 * For scalars o and u, with constant a
 */
template <typename T>
struct save_and_scale
{
   typedef thrust::tuple<T &, T &, T> Tuple;

   const T scale;

   save_and_scale(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t);
      thrust::get<1>(t) = thrust::get<2>(t) * scale;
   }
};

/**
 * @brief xpyz
 *
 * Performs
 * y = x + y .* z
 * For scalars x,y,z (indices 1,0,2 respectively)
 */
template <typename T>
struct xpyz
{
   typedef thrust::tuple<T &, T, T> Tuple;

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t) * thrust::get<0>(t);
   }
};

/**
 * @brief scale
 *
 * Performs
 * x = d .* x
 * For scalars x, d
 */
template <typename T>
struct scaleInPlace
{
   typedef thrust::tuple<T, T&> Tuple;

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = thrust::get<0>(t) * thrust::get<1>(t);
   }
};

/**
 * @brief add
 *
 * Performs
 * x = x + coef * y
 * For scalars x, d
 */
template <typename T>
struct add
{
   typedef thrust::tuple<T, T&> Tuple;

   const T coef;
   add(T _coef = 1.0) : coef(_coef) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = coef * thrust::get<0>(t) + thrust::get<1>(t);
   }
};

/**
 * @brief add
 *
 * Performs
 * x = x + coef * d.*y
 * For scalars x, d
 */
template <typename T>
struct scaledAdd
{
   typedef thrust::tuple<T, T, T&> Tuple;
   const T scale;
   scaledAdd(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = thrust::get<2>(t) + scale * thrust::get<0>(t) * thrust::get<1>(t);
   }
};

/**
 * @brief add
 *
 * Performs
 * r = r - D .* tmp
 * v = coef0 * r + coef1 * v
 * 
 */
template <typename T>
struct updateRAndV
{
   typedef thrust::tuple<T, T, T&, T&> Tuple;
   const T coef0;
   const T coef1;
   updateRAndV(T _coef0, T _coef1) : coef0(_coef0), coef1(_coef1) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = thrust::get<2>(t) - thrust::get<0>(t) * thrust::get<1>(t);
      thrust::get<3>(t) = coef0 * thrust::get<2>(t) + coef1 * thrust::get<3>(t);
   }
};

/**
 * @brief scale
 *
 * Performs
 * y = coef * d .* x
 * For scalars x, d, y
 */
template <typename T>
struct applySmoother
{
   typedef thrust::tuple<T, T, T&> Tuple;

   const T coef;
   applySmoother(T _coef = 1.0) : coef(_coef) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = coef * thrust::get<0>(t) * thrust::get<1>(t);
   }
};

/**
 * @brief waxpyz
 *
 * Performs
 * y = a * x
 * constant a
 */
template <typename T>
struct scaleConstant
{
   typedef thrust::tuple<T, T&> Tuple;

   const T scale;
   scaleConstant(T _scale) : scale(_scale) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = scale * thrust::get<0>(t);
   }
};

/**
 * @brief update
 *
 * Performs
 * d = scale0 * r + scale1 * d
 */
template <typename T>
struct update
{
   typedef thrust::tuple<T, T&> Tuple;

   const T scale0;
   const T scale1;
   update(T _scale0, T _scale1) : scale0(_scale0), scale1(_scale1) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<1>(t) = scale1 * thrust::get<1>(t) + scale0 * thrust::get<0>(t);
   }
};

/**
 * @brief updateCol
 *
 * Performs
 * d = scale0 * x.*r + scale1 * d
 */
template <typename T>
struct updateCol
{
   typedef thrust::tuple<T, T, T&> Tuple;

   const T scale0;
   const T scale1;
   updateCol(T _scale0, T _scale1) : scale0(_scale0), scale1(_scale1) {}

   __host__ __device__ void operator()(Tuple t)
   {
      thrust::get<2>(t) = scale1 * thrust::get<2>(t) + scale0 * thrust::get<1>(t) * thrust::get<0>(t);
   }
};
/**
 * @brief Solve using a chebyshev polynomial on the device
 *
 * @param[in] A Matrix to relax with
 * @param[in] f right-hand side
 * @param[in] ds_data Diagonal information
 * @param[in] coefs Polynomial coefficients
 * @param[in] order Order of the polynomial
 * @param[in] scale Whether or not to scale by diagonal
 * @param[in] scale Whether or not to use a variant
 * @param[in,out] u Initial/updated approximation
 * @param[out] v Temp vector
 * @param[out] v Temp Vector
 */
HYPRE_Int
hypre_ParCSRRelax_Cheby_SolveDevice(hypre_ParCSRMatrix *A, /* matrix to relax with */
                                    hypre_ParVector    *f, /* right-hand side */
                                    HYPRE_Real         *ds_data,
                                    HYPRE_Real         *coefs,
                                    HYPRE_Int           order, /* polynomial order */
                                    HYPRE_Int           scale, /* scale by diagonal?*/
                                    HYPRE_Int           variant,
                                    hypre_ParVector    *u,          /* initial/updated approximation */
                                    hypre_ParVector    *v,          /* temporary vector */
                                    hypre_ParVector    *r,          /*another temp vector */
                                    hypre_ParVector    *orig_u_vec, /*another temp vector */
                                    hypre_ParVector    *tmp_vec)       /*a potential temp vector */
{
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real      *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real      *v_data = hypre_VectorData(hypre_ParVectorLocalVector(v));

   HYPRE_Real *r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   HYPRE_Int i;
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);

   HYPRE_Real  mult;

   HYPRE_Int cheby_order;

   HYPRE_Real *tmp_data;

   /* u = u + p(A)r */

   if (variant == 0 || variant == 1){
     if (order > 4) { order = 4; }
     if (order < 1) { order = 1; }
   }

   /* we are using the order of p(A) */
   cheby_order = order - 1;

   hypre_assert(hypre_VectorSize(hypre_ParVectorLocalVector(orig_u_vec)) >= num_rows);
   HYPRE_Real *orig_u = hypre_VectorData(hypre_ParVectorLocalVector(orig_u_vec));

   if(variant == 0 || variant == 1){

     if (!scale)
     {
        /* get residual: r = f - A*u */
        hypre_ParVectorCopy(f, r);
        hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

        /* o = u; u = r .* coef */
        HYPRE_THRUST_CALL(
           for_each,
           thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
           thrust::make_zip_iterator(thrust::make_tuple(orig_u + num_rows, u_data + num_rows,
                                                        r_data + num_rows)),
           save_and_scale<HYPRE_Real>(coefs[cheby_order]));

        for (i = cheby_order - 1; i >= 0; i--)
        {
           hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, v);
           mult = coefs[i];

           /* u = mult * r + v */
           hypreDevice_ComplexAxpyn( r_data, num_rows, v_data, u_data, mult );
        }

        /* u = o + u */
        hypreDevice_ComplexAxpyn( orig_u, num_rows, u_data, u_data, 1.0);
     }
     else /* scaling! */
     {

        /*grab 1/sqrt(diagonal) */

        tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

        /* get ds_data and get scaled residual: r = D^(-1/2)f -
         * D^(-1/2)A*u */

        hypre_ParCSRMatrixMatvec(-1.0, A, u, 0.0, tmp_vec);
        /* r = ds .* (f + tmp) */

        /* TODO: It might be possible to merge this and the next call to:
         * r[j] = ds_data[j] * (f_data[j] + tmp_data[j]); o[j] = u[j]; u[j] = r[j] * coef */
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, f_data, tmp_data)) + num_rows,
                          wxypz<HYPRE_Real>());

        /* save original u, then start
           the iteration by multiplying r by the cheby coef.*/

        /* o = u;  u = r * coef */
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(orig_u, u_data, r_data)) + num_rows,
                          save_and_scale<HYPRE_Real>(coefs[cheby_order]));

        /* now do the other coefficients */
        for (i = cheby_order - 1; i >= 0; i--)
        {
           /* v = D^(-1/2)AD^(-1/2)u */
           /* tmp = ds .* u */
           HYPRE_THRUST_CALL( transform, ds_data, ds_data + num_rows, u_data, tmp_data, _1 * _2 );

           hypre_ParCSRMatrixMatvec(1.0, A, tmp_vec, 0.0, v);

           /* u_new = coef*r + v*/
           mult = coefs[i];

           /* u = coef * r + ds .* v */
           HYPRE_THRUST_CALL(for_each,
                             thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)),
                             thrust::make_zip_iterator(thrust::make_tuple(u_data, r_data, ds_data, v_data)) + num_rows,
                             waxpyz<HYPRE_Real>(mult));
        } /* end of cheby_order loop */

        /* now we have to scale u_data before adding it to u_orig*/

        /* u = orig_u + ds .* u */
        HYPRE_THRUST_CALL(
           for_each,
           thrust::make_zip_iterator(thrust::make_tuple(u_data, orig_u, ds_data)),
           thrust::make_zip_iterator(thrust::make_tuple(u_data + num_rows, orig_u + num_rows,
                                                        ds_data + num_rows)),
           xpyz<HYPRE_Real>());


     } /* end of scaling code */
   }
   else if(variant == 2)
   {
      const auto lambda_max = coefs[0];
      const auto lambda_min = coefs[1];

      const auto theta = 0.5 * (lambda_max + lambda_min);
      const auto delta = 0.5 * (lambda_max - lambda_min);
      const auto sigma = theta / delta;
      auto rho = 1.0 / sigma;

      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

      // r := f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      // TODO: consolidate two calls below

      // r = D^{-1} r
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(ds_data, r_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(ds_data, r_data)) + num_rows,
                        scaleInPlace<HYPRE_Real>());
      
      // v := 1/theta r
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, v_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, v_data)) + num_rows,
                        scaleConstant<HYPRE_Real>(1.0 / theta));
      
      for(int i = 0; i < cheby_order; ++i){
        // u += v
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                          add<HYPRE_Real>());
        // tmp = Av
        hypre_ParCSRMatrixMatvec(1.0, A, v, 0.0, tmp_vec);

        const auto rhoSave = rho;
        rho = 1.0 / (2 * sigma - rho);

        const auto vcoef = rho * rhoSave;
        const auto rcoef = 2.0 * rho / delta;

        // r = r - D^{-1} Av
        // v = rho_{k+1} rho_k * v + 2 rho_{k+1} / delta r
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(ds_data, tmp_data, r_data, v_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(ds_data, tmp_data, r_data, v_data)) + num_rows,
                          updateRAndV<HYPRE_Real>(rcoef, vcoef));
      }

      // u += v;
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                        add<HYPRE_Real>());

   }
   else if(variant == 3 || variant == 4)
   {
      const auto lambda_max = coefs[0];
      const auto lambda_min = coefs[1];

      const auto theta = 0.5 * (lambda_max + lambda_min);
      const auto delta = 0.5 * (lambda_max - lambda_min);
      const auto sigma = theta / delta;
      auto betas = optimalWeightsImpl(cheby_order+1);
      if(variant == 3){
         std::fill(betas.begin(), betas.end(), 1.0);
      }

      auto rho = 1.0 / sigma;

      tmp_data = hypre_VectorData(hypre_ParVectorLocalVector(tmp_vec));

      // r := f - A*u
      hypre_ParVectorCopy(f, r);
      hypre_ParCSRMatrixMatvec(-1.0, A, u, 1.0, r);

      // v := \dfrac{4}{3} \dfrac{1}{\rho(D^{-1}A)} D^{-1} r
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)) + num_rows,
                        applySmoother<HYPRE_Real>(4.0 / 3.0 / lambda_max));
      
      for(int i = 0; i < cheby_order; ++i){
        // u += \beta_k v
        // since this is _not_ the optimized variant, \beta := 1.0
        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                          add<HYPRE_Real>(betas.at(i)));
        // r = r - Av
        hypre_ParCSRMatrixMatvec(-1.0, A, v, 1.0, r);

        // + 2 offset is due to two issues:
        // + 1 is from https://arxiv.org/pdf/2202.08830.pdf being written in 1-based indexing
        // + 1 is from pre-computing z_1 _outside_ of the loop
        // v = \dfrac{(2i-3)}{(2i+1)} v + \dfrac{(8i-4)}{(2i+1)} \dfrac{1}{\rho(SA)} S r
        const auto id = i + 2;
        const auto vScale = (2.0 * id - 3.0) / (2.0 * id + 1.0);
        const auto rScale = (8.0 * id - 4.0) / (2.0 * id + 1.0) / lambda_max;

        HYPRE_THRUST_CALL(for_each,
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)),
                          thrust::make_zip_iterator(thrust::make_tuple(r_data, ds_data, v_data)) + num_rows,
                          updateCol<HYPRE_Real>(rScale, vScale));

      }

      // u += \beta v;
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)),
                        thrust::make_zip_iterator(thrust::make_tuple(v_data, u_data)) + num_rows,
                        add<HYPRE_Real>(betas.back()));

   }

   return hypre_error_flag;
}
#endif
