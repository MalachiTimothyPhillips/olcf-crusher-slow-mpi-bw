extern "C" void FUNC(UrstCubatureHex3D)(const dlong &Nelements,
                                        const dfloat *__restrict__ cubvgeo,
                                        const dfloat *__restrict__ cubInterpT,
                                        const dfloat *__restrict__ cubW,
                                        const dlong &offset,
                                        const dlong &cubatureOffset,
                                        const dfloat *__restrict__ U,
                                        const dfloat *__restrict__ W,
                                        dfloat *__restrict__ result)
{

  dfloat s_cubInterpT[p_Nq][p_cubNq];

  dfloat s_U[p_cubNq][p_cubNq];
  dfloat s_V[p_cubNq][p_cubNq];
  dfloat s_W[p_cubNq][p_cubNq];

  dfloat s_U1[p_Nq][p_cubNq];
  dfloat s_V1[p_Nq][p_cubNq];
  dfloat s_W1[p_Nq][p_cubNq];

  dfloat r_U[p_cubNq][p_cubNq][p_cubNq];
  dfloat r_V[p_cubNq][p_cubNq][p_cubNq];
  dfloat r_W[p_cubNq][p_cubNq][p_cubNq];

  for(int j = 0; j < p_Nq; ++j) {
    for(int i = 0; i < p_cubNq; ++i) {
      const int id = i + j * p_cubNq;

      if(i < p_Nq){
          s_D[j][i]  = D[j*p_Nq+i];
      }

      s_cubInterpT[j][i] = cubInterpT[id];
    }
  }

  for(dlong element = 0; element < Nelements; ++element) {

#pragma unroll
    for(int c = 0; c < p_Nq; ++c) {

      #pragma unroll
      for(int b = 0; b < p_Nq; ++b)
        #pragma unroll
        for(int a = 0; a < p_Nq; ++a){
          const dlong id = element * p_Np + c * p_Nq * p_Nq + b * p_Nq + a;

          dfloat Ue = U[id + 0 * offset];
          dfloat Ve = U[id + 1 * offset];
          dfloat We = U[id + 2 * offset];
          if(p_relative){
            Ue -= W[id + 0 * offset];
            Ve -= W[id + 1 * offset];
            We -= W[id + 2 * offset];
          }

          s_U[b][a] = Ue;
          s_V[b][a] = Ve;
          s_W[b][a] = We;
        }

      // interpolate in 'r'
      #pragma unroll
      for(int b = 0; b < p_Nq; ++b)
        #pragma unroll
        for(int i = 0; i < p_cubNq; ++i)
          {
            dfloat U1  = 0, V1 = 0,  W1 = 0;

            #pragma unroll
            for(int a = 0; a < p_Nq; ++a) {
              dfloat Iia = s_cubInterpT[a][i];
              U1  += Iia * s_U[b][a];
              V1  += Iia * s_V[b][a];
              W1  += Iia * s_W[b][a];
            }

            s_U1[b][i] = U1;
            s_V1[b][i] = V1;
            s_W1[b][i] = W1;
          }

      // interpolate in 's'
      #pragma unroll
      for(int j = 0; j < p_cubNq; ++j) {
        #pragma unroll
        for(int i = 0; i < p_cubNq; ++i) {
          dfloat U2 = 0, V2 = 0,  W2 = 0;

          // interpolate in b
          #pragma unroll
          for(int b = 0; b < p_Nq; ++b) {
            dfloat Ijb = s_cubInterpT[b][j];
            U2 += Ijb * s_U1[b][i];
            V2 += Ijb * s_V1[b][i];
            W2 += Ijb * s_W1[b][i];
          }

          // interpolate in c progressively
          #pragma unroll
          for(int k = 0; k < p_cubNq; ++k) {
            dfloat Ikc = s_cubInterpT[c][k];

            r_U[j][i][k] += Ikc * U2;
            r_V[j][i][k] += Ikc * V2;
            r_W[j][i][k] += Ikc * W2;
          }
        }
      }
    }

    #pragma unroll
    for(int k = 0; k < p_cubNq; ++k) {
      for(int j = 0; j < p_cubNq; ++j)
        for(int i = 0; i < p_cubNq; ++i) {
          const dlong gid = e * p_cubNp * p_Ncubvgeo + k * p_cubNq * p_cubNq + j * p_cubNq + i;
          const dfloat drdx = cubvgeo[gid + p_RXID * p_cubNp];
          const dfloat drdy = cubvgeo[gid + p_RYID * p_cubNp];
          const dfloat drdz = cubvgeo[gid + p_RZID * p_cubNp];
          const dfloat dsdx = cubvgeo[gid + p_SXID * p_cubNp];
          const dfloat dsdy = cubvgeo[gid + p_SYID * p_cubNp];
          const dfloat dsdz = cubvgeo[gid + p_SZID * p_cubNp];
          const dfloat dtdx = cubvgeo[gid + p_TXID * p_cubNp];
          const dfloat dtdy = cubvgeo[gid + p_TYID * p_cubNp];
          const dfloat dtdz = cubvgeo[gid + p_TZID * p_cubNp];

          const dfloat W = cubW[i] * cubW[j] * cubW[k];

          const dfloat Un = r_U[j][i][k];
          const dfloat Vn = r_V[j][i][k];
          const dfloat Wn = r_W[j][i][k];

          const dlong id = element * p_cubNp + k * p_cubNq * p_cubNq + j * p_cubNq + i;
          result[id + 0 * cubatureOffset] = W * (Un * drdx + Vn * drdy + Wn * drdz);
          result[id + 1 * cubatureOffset] = W * (Un * dsdx + Vn * dsdy + Wn * dsdz);
          result[id + 2 * cubatureOffset] = W * (Un * dtdx + Vn * dtdy + Wn * dtdz);
        }
    }
  }
      }
