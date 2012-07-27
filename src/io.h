/** \file
    File IO for nD-arrays.

    \author Nathan Clack
    \date   June 2012

    \todo Need better error messages.
    */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ndio_t* ndio_t;
//typedef struct _nd_t*   nd_t;

int       ndioPreloadPlugins();

int       ndioIsFile(const char *filename);

ndio_t    ndioOpen  (const char *filename, const char *format, const char *mode);
void      ndioClose (ndio_t file);

nd_t      ndioShape (ndio_t file);  // caller must free the returned object
ndio_t    ndioRead  (ndio_t file, nd_t dst);
ndio_t    ndioWrite (ndio_t file, nd_t src);

char*     ndioError(ndio_t file);
void      ndioLogError(ndio_t file, const char *fmt, ...);
void      ndioResetLog(ndio_t file);

void*     ndioContext(ndio_t file); // returns the format-specific file context.

/// \todo ndioReadSubarray(ndio_t file, nd_t dst, int *origin); Dst has shape and dim params, origin is coordinates for offset
/// \todo ndslice_t ndioSlice(ndio_t file, int idim);           Request slice along dim i

#ifdef __cplusplus
} //extern "C" {
#endif
