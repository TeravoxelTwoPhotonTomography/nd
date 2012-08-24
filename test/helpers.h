/** \file
 *  Helper functions for test suite.
 *
 *  @cond TEST
 */
#pragma once
#include <cmath>
#include "nd.h"

///// helpers
template<class T> static void cast(nd_t a);
template<> void cast<uint8_t >(nd_t a) {ndcast(a,nd_u8 );}
template<> void cast<uint16_t>(nd_t a) {ndcast(a,nd_u16);}
template<> void cast<uint32_t>(nd_t a) {ndcast(a,nd_u32);}
template<> void cast<uint64_t>(nd_t a) {ndcast(a,nd_u64);}
template<> void cast< int8_t >(nd_t a) {ndcast(a,nd_i8 );}
template<> void cast< int16_t>(nd_t a) {ndcast(a,nd_i16);}
template<> void cast< int32_t>(nd_t a) {ndcast(a,nd_i32);}
template<> void cast< int64_t>(nd_t a) {ndcast(a,nd_i64);}
template<> void cast< float  >(nd_t a) {ndcast(a,nd_f32);}
template<> void cast< double >(nd_t a) {ndcast(a,nd_f64);}

/**
 * Root mean squared error
 */
template<class T>
double RMSE(size_t n, T* a, T* b)
{ double ssq=0.0;
  for(size_t i=0;i<n;++i)
  { double t = (double)b[i]-(double)a[i];
    ssq+=t*t;
  }
  return sqrt(ssq/n);
}

/**
 * Find first difference. 
 * \returns the index of the first difference, or -1 if none found
 */
template<class T>
int firstdiff(size_t n, const T* a, const T* b)
{ for(size_t i=0;i<n;++i)
    if(a[i]!=b[i])
      return i;
  return -1;
}

/**
 * Allocates an array with shape \a shape and fills it with zeros.
 */
template<class T>
T* zeros(size_t ndim, size_t* shape)
{ size_t i,nelem;
  nelem = shape[0];
  for(i=1;i<ndim;++i)
    nelem*=shape[i];
  T* v = new T[nelem];
  memset(v,0,nelem*sizeof(T));
}
/// @endcond
