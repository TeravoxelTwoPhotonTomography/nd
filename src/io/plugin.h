/** \file
    Loading ndio format plugins.
*/
#pragma once

#ifdef __cplusplus
extern "C" {
#endif
#include <stddef.h>

  typedef struct _ndio_fmt_t **ndio_fmts_t;

  ndio_fmts_t ndioLoadPlugins(const char *path, size_t *n); ///< Loads the plugins contained in \a path.  \returns 0 on failure, otherwise an array with the loaded plugins. \param[out] n The number of loaded plugins.
  void        ndioFreePlugins(ndio_fmts_t fmts, size_t n);  ///< Releases resources.  Always succeeds.

#ifdef __cplusplus
}//extern "C" {
#endif
