/**
 * \file
 *  Protocol required for implementations of a ndio compatible file format.
 *
 *  Implementations of a file format must export a function with the signiture:
 *  \code
 *   const ndio_fmt_t ndio_get_format_api();
 *  \endcode
 *
 *  Required: include nd.h before this file.
 *
 *  \ingroup ndioplugins
 *  \author Nathan Clack
 *  \date   June 2012
 *
 *  \todo Change g_format so it generates ndio_fmt_t on demand from a function
 *        call.  This lets format context data be dynamically allocated.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif
  //Required: include io.h and core.h before this file
  //typedef struct _ndio_t* ndio_t;
  //typedef struct _nd_t*   nd_t;

  typedef const char* (*_ndio__fmt_name_t)(void);                                    ///< Plugin name. \returns a string with the format's name.
  typedef unsigned    (*_ndio__is_fmt_t  )(const char* path, const char *mode);      ///< Determines format support for file at \a path. \returns 1 if the format supports reading the file at \a path or writing a file with the extension indicated by \path.  Returns 0 otherwise.
  typedef void*       (*_ndio__open_t    )(ndio_fmt_t* fmt, const char* path, const char *mode);      ///< Opens the file at \a path according to the \a mode. \returns a the address of the file context on success, otherwise 0.
  typedef void        (*_ndio__close_t   )(ndio_t file);                             ///< Closes the file and releases any internally acquired resources.  Should always succeed.
  typedef nd_t        (*_ndio__shape_t   )(ndio_t file);                             ///< Gets the shape of the data in \a file. \returns 0 on failure, otherwise returns the shape, dimensions, and type of the data in a dummy nd_t object.  Caller is responsible for freeing the returned object.
  typedef unsigned    (*_ndio__read_t    )(ndio_t file, nd_t dst);                   ///< Reads the data in \a file to \a a. \returns 0 on failure, otherwise returns \a self.  \param[out] dst Destination container for the file's contents. Should have been allocated with enough storage to hold the data.
  typedef unsigned    (*_ndio__write_t   )(ndio_t file, nd_t src);                   ///< Writes the data in \a a to \a file. \returns 0 on failure, otherwise returns \a self.  \param[in]  src The data to save to the file.  Should be treated as an append operation.  Append vs overwrite behavior should be handled on opening the file.
  typedef unsigned    (*_ndio__set_t     )(ndio_fmt_t* fmt, void *param, size_t nbytes); ///< (optional) Set format specific options or metadata.
  typedef void*       (*_ndio__get_t     )(ndio_fmt_t* fmt);                             ///< (optional) Get format specific options or metadata.
  typedef unsigned    (*_ndio__canseek_t )(ndio_t file, size_t idim);                ///< (optional) Is seeking supported for dimension \a idim.
  typedef unsigned    (*_ndio__seek_t    )(ndio_t file, nd_t dst, size_t *pos);      ///< (optional) Satisfy a seek request.
  typedef unsigned    (*_ndio__subarray_t)(ndio_t file, nd_t dst, size_t *origin, size_t *step); ///< (optional) For formats that directly support reading subvolems.  Use this instead of the seek() interface.
  typedef void        (*_ndio_finalize_format_t)(ndio_fmt_t *fmt);                   ///< (optional) called when reference count goes to zero.  Do _not_ free \a fmt.

  typedef unsigned    (*_ndio__add_plugin_t)(ndio_fmt_t *api);

  /**
   * An ndio plugin must implement the functions in this interface.
   * The plugin exposes the implementation by defining a function that
   * returns the interface:
   * \code{c}
   *    const ndio_fmt_t* ndio_get_format_api(void);
   * \endcode
   * The function must be visible from a shared library.
   * \ingroup ndioplugins
   */
  struct _ndio_fmt_t
  { _ndio__fmt_name_t name;       ///< \returns a string with the format's name.
    _ndio__is_fmt_t   is_fmt;     ///< \returns 1 if the format supports reading the file at \a path or writing a file with the extension indicated by \path.  Returns 0 otherwise.
    _ndio__open_t     open;       ///< Opens the file at \a path according to the \a mode. \returns a the address of the file context on success, otherwise 0.
    _ndio__close_t    close;      ///< Closes the file and releases any internally acquired resources.  Should always succeed.
    _ndio__shape_t    shape;      ///< \returns 0 on failure, otherwise returns the shape, dimensions, and type of the data in a dummy nd_t object.  Caller is responsible for freeing the returned object.
    _ndio__read_t     read;       ///< \returns 0 on failure, otherwise returns 1.  \param[out] dst Destination container for the file's contents. Should have been allocated with enough storage to hold the data.  Should always read the whole file (rewind if necessecary).
    _ndio__write_t    write;      ///< \returns 0 on failure, otherwise returns 1.  \param[in]  src The data to save to the file.  Should be treated as an append operation.  Append vs overwrite behavior should be handled on opening the file.
    _ndio__set_t      set;        ///< Set format specific data.  \returns 0 on failure, otherwise returns 1.
    _ndio__get_t      get;        ///< Get format specific data.  \returns 0 on failure, otherwise returns 1.
    _ndio__canseek_t  canseek;    ///< \returns 1 if \a idim is seekable, otherwise return 0.  The function does not need to handle bounds checking for \a idim.
    _ndio__seek_t     seek;       ///< Fills \a dst with a seek volume from position \a pos.  The read volume has full shape along a non-seekable dimension and unit shape on seekable dimensions.
    _ndio__subarray_t subarray;   ///< Reads a \a dst shaped subvolume starting at \a ori with step \a step.  For formats that support it, use this instead of seek/can_seek.
    _ndio_finalize_format_t close_fmt; ///< called when reference count goes to zero.  Do _not_ free \a fmt.
    _ndio__add_plugin_t add_plugin;
    void*             lib;        ///< Context.  Used internally. Should be NULL when returned from a ndio_get_format_api() call.
    int               ref;        ///< Reference count.  Used internally.  Initialize to zero.
  };

  typedef const ndio_fmt_t* (*ndio_get_format_api_t)(void); ///< Produces a plugin's interface.  \returns the api used to implement the format.  The caller will not free the returned pointer.  It should be statically allocated somewhere.
#ifdef __cplusplus
} //extern "C" {
#endif
