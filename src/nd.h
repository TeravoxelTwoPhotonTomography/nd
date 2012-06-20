/** \file 
    N-Dimensional array type. 

    \author Nathan Clack
    \date   June 2012
*/
#pragma once
#include <stdlib.h> // for size_t
#include <stdint.h> // fixed width types

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _nd_type_id_t
{
  nd_id_unknown=-1,
  nd_u8,
  nd_u16,
  nd_u32,
  nd_u64,
  nd_i8,
  nd_i16,
  nd_i32,
  nd_i64,
  nd_f32,
  nd_f64,
  nd_id_count
} nd_type_id_t;

typedef enum _nd_kind_t
{ nd_unknown_kind=-1,
  nd_cpu,                                                 ///< default
  nd_gpu_cuda,
  nd_kind_count
} nd_kind_t;                                              ///< The "kind" is used to select the implementation of certain algorithms

typedef struct _nd_t* nd_t;

nd_t          ndinit();                                   ///< allocates an empty array.
void          ndfree(nd_t a);                             ///< frees resources used by the array.  Will not generate an error.

//nd_t          ndview(nd_t a);                             ///< caller is responsible for freeing the returned result

size_t        ndbpp    (const nd_t a);                    ///< \returns the bytes per pixel
size_t        ndnelem  (const nd_t a);                    ///< \returns the total number of elements in the array
size_t        ndnbytes (const nd_t a);                    ///< \returns the total number of bytes in the array
void*         nddata   (const nd_t a);                    ///< \returns a pointer to the origin pixel
size_t        ndndim   (const nd_t a);
const size_t* ndshape  (const nd_t a);
const size_t* ndstrides(const nd_t a);

char*         nderror(const nd_t a);                      ///< \returns NULL if no error, otherwise a descriptive string
void          ndLogError(nd_t a, const char *fmt, ...);   ///< logs an error, storing it with the array
void          ndResetLog(nd_t a);                         ///< clears the error log returning the array to an error free state
              
nd_t          ndcast(nd_t a, nd_type_id_t desc);          ///< Changes the pixel data type
nd_type_id_t  ndtype(const nd_t a);                       ///< \returns the pixel data type
nd_t          ndref (nd_t a, void *buf, size_t nelem);    ///< Binds the buffer to \a and reshapes \a as a 1d container.

nd_t          ndsetkind(nd_t a, nd_kind_t kind);
nd_kind_t     ndkind(const nd_t a);
              
nd_t          ndreshape(nd_t a,unsigned ndim,const size_t *shape);
#ifdef __cplusplus
} //extern "C" {
#endif
