/** \file
    N-Dimensional array type and core operations.

    \section nd-notes Notes

    - Keep a ref count against data?  Don't do memory management, so might not need one.  Could be useful to emit an
      event when the last reference is free'd.

    - Why slices?  Iterating on dims isn't so hard.
      - opportunity to abstract next()/seek()
        - on demand loading of frames from a video
        - infinite streams/lazy
          - these aren't so easy...just think of a copy or transpose.  They'd have to be lazy themselves.

    - Be explicit about copies.

    \todo ndref() should explicitly set kind.  Should eliminated nelem argument.

    \author Nathan Clack
    \date   June 2012
*/
#pragma once
#include <stdlib.h> // for size_t
#include <stdint.h> // fixed width types

#ifdef __cplusplus
extern "C" {
#endif

/** Type identifiers */
typedef enum _nd_type_id_t
{
  nd_id_unknown=-1,
  nd_u8,  nd_u16,  nd_u32,  nd_u64,
  nd_i8,  nd_i16,  nd_i32,  nd_i64,
                   nd_f32,  nd_f64,
  nd_id_count
} nd_type_id_t;

/** Kind identifiers.
 *
 *  Different in-memory representations may demand different implementations of
 *  algorithms.  The "kind" helps determine which implementation should be
 *  chosen, if one is available.
 */
typedef enum _nd_kind_t
{ nd_unknown_kind=-1,
  nd_heap,                                                ///< (default) Linear storage in machine local RAM.
  nd_static,                                              ///< data is statically allocated in RAM.
  nd_gpu_cuda,                                            ///< data is stored on CUDA compatible device memory.
  nd_file,                                                ///< data pointer is an ndio_t file context.
  nd_kind_count
} nd_kind_t;                                              ///< The "kind" is used to select the implementation of certain algorithms

typedef struct _nd_t* nd_t;                               ///< Abstract data type representing an N-dimensional array.

nd_t          ndinit();                                   ///< allocates an empty array.
void          ndfree(nd_t a);                             ///< frees resources used by the array.  Will not generate an error.

//nd_t          ndview(nd_t a);                             ///< caller is responsible for freeing the returned result

size_t        ndbpp    (const nd_t a);                    ///< \returns the bytes per pixel
size_t        ndnelem  (const nd_t a);                    ///< \returns the total number of elements in the array
size_t        ndnbytes (const nd_t a);                    ///< \returns the total number of bytes in the array
void*         nddata   (const nd_t a);                    ///< \returns a pointer to the origin pixel
unsigned      ndndim   (const nd_t a);
size_t*       ndshape  (const nd_t a);
size_t*       ndstrides(const nd_t a);

char*         nderror(const nd_t a);                      ///< \returns NULL if no error, otherwise a descriptive string
void          ndLogError(nd_t a, const char *fmt, ...);   ///< logs an error, storing it with the array
void          ndResetLog(nd_t a);                         ///< clears the error log returning the array to an error free state

nd_t          ndcast(nd_t a, nd_type_id_t desc);          ///< Changes the pixel data type
nd_type_id_t  ndtype(const nd_t a);                       ///< \returns the pixel data type
nd_t          ndref (nd_t a, void *buf, size_t nelem);    ///< Binds the buffer to \a and reshapes \a as a 1d container.
//nd_t          ndref (nd_t a, void *buf, nd_kind_t kind);    // (a better calling convention?)

nd_t          ndsetkind(nd_t a, nd_kind_t kind);
nd_kind_t     ndkind(const nd_t a);

nd_t          ndreshape(nd_t a,unsigned ndim,const size_t *shape);
nd_t          ndShapeSet(nd_t a, unsigned idim, size_t val);
nd_t          ndInsertDim(nd_t a, unsigned idim);
nd_t          ndRemoveDim(nd_t a, unsigned idim);

nd_t          ndoffset(nd_t a, unsigned idim, int64_t o);///< increments data pointer: data+=o*stride[idim]

/// \todo ndslice_t object. Castable to nd_t (ir is an nd_t, especially if there's a good default idim).
/// \todo ndslice_t ndslice(nd_t a, unsigned idim);                        Starts an iterator along dimension idim.  Constructs a new object.  Want shape manip to be independent of parent array.
/// \todo ndslice_t ndsubslice(ndslice_t a, int start, int step, int end); Default is beg,1,end.  Use python convention for signed numbers.
/// \todo ndslice_t ndnext(ndslice_t a);                                   Deallocates the slice (if necessary) at the end, returns null at the end
/// \todo ndslice_t ndseek(ndslice_t a, int i);
/// \todo ndslice_t ndSliceBounds(ndslice_t, int *start, int *step, int *end);

//
// === CUDA ===
//

#include "cuda_runtime_api.h"

/// \todo Add current stream as a state (global or bind to array) and use that instead of a common argument.  Especially think of kernel launches.

nd_t  ndcuda         (nd_t a           ,cudaStream_t stream);
nd_t  ndCudaSyncShape(nd_t self        ,cudaStream_t stream);
nd_t  ndCudaCopy     (nd_t dst,nd_t src,cudaStream_t stream);
void* ndCudaShape    (nd_t self);
void* ndCudaStrides  (nd_t self);

#ifdef __cplusplus
} //extern "C" {
#endif
