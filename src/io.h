/** \file
    File IO for nD-arrays.

    \author Nathan Clack
    \date   June 2012

    \todo Need better error messages.
    \todo subvolumes/slicing


    */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _ndio_t* ndio_t;
typedef struct _ndio_fmt_t ndio_fmt_t;
//typedef struct _nd_t*   nd_t;

int       ndioPreloadPlugins();
int       ndioAddPlugin(ndio_fmt_t *api);

int       ndioIsFile(const char *filename);

ndio_t    ndioOpen  (const char *filename, const char *format, const char *mode);
void      ndioClose (ndio_t file);

nd_t      ndioShape (ndio_t file);  // caller must free the returned object
ndio_t    ndioRead  (ndio_t file, nd_t dst);
ndio_t    ndioWrite (ndio_t file, nd_t src);

const char* ndioFormatName(ndio_t file);
ndio_t    ndioSet(ndio_t file, void *param, size_t nbytes);
void*     ndioGet(ndio_t file);

char*     ndioError(ndio_t file);
void      ndioLogError(ndio_t file, const char *fmt, ...);
void      ndioResetLog(ndio_t file);

void*     ndioContext(ndio_t file); // returns the format-specific file context.

ndio_t    ndioReadSubarray(ndio_t file, nd_t dst, size_t *origin, size_t *step); // dst has shape and dim params, origin is coordinates for offset
/// \todo ndslice_t ndioSlice(ndio_t file, int idim);           Request slice along dim i

unsigned  ndioCanSeek(ndio_t file, size_t idim);

#ifdef __cplusplus
} //extern "C" {
#endif
