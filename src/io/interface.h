/** \file
    Description of protocol required for implementations of a ndio compatible
    file format.

    Implementations of a file format must export a function with the signiture:
    \code
     const ndio_fmt_t ndio_get_format_api();
    \endcode

    \author Nathan Clack
    \date   June 2012

    \todo Add some facility to set paramters. e.g. Tiff Compression, or bitrate for a lossy compression.
          Should be some key-value store, or each plugin should expose a special header that exposes a
          parameters struct.
    */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif
  typedef struct _ndio_t* ndio_t;
  typedef struct _nd_t*   nd_t;

  typedef const char* (*_ndio__fmt_name_t)(void);
  typedef unsigned    (*_ndio__is_fmt_t  )(const char* path, const char *mode);
  typedef void*       (*_ndio__open_t    )(const char* path, const char *mode);
  typedef void        (*_ndio__close_t   )(ndio_t file);
  typedef nd_t        (*_ndio__shape_t   )(ndio_t file);
  typedef unsigned    (*_ndio__read_t    )(ndio_t file, nd_t dst);
  typedef unsigned    (*_ndio__write_t   )(ndio_t file, nd_t src);

  typedef struct _ndio_fmt_t
  { _ndio__fmt_name_t name;       ///< \returns a string with the format's name.
    _ndio__is_fmt_t   is_fmt;     ///< \returns 1 if the format supports reading the file at \a path or writing a file with the extension indicated by \path.  Returns 0 otherwise.
    _ndio__open_t     open;       ///< Opens the file at \a path according to the \a mode. \returns a the address of the file context on success, otherwise 0.
    _ndio__close_t    close;      ///< Closes the file and releases any internally acquired resources.  Should always succeed.
    _ndio__shape_t    shape;      ///< \returns 0 on failure, otherwise returns the shape, dimensions, and type of the data in a dummy nd_t object.  Caller is responsible for freeing the returned object.
    _ndio__read_t     read;       ///< \returns 0 on failure, otherwise returns \a self.  \param[out] dst Destination container for the file's contents. Should have been allocated with enough storage to hold the data.
    _ndio__write_t    write;      ///< \returns 0 on failure, otherwise returns \a self.  \param[in]  src The data to save to the file.  Should be treated as an append operation.  Append vs overwrite behavior should be handled on opening the file.
    void*             lib;        ///< Context.  Used internally. Should be NULL when returned from a ndio_get_format_api() call.
  } ndio_fmt_t;

  typedef const ndio_fmt_t* (*ndio_get_format_api_t)(void); ///< \returns the api used to implement the format.  The caller will not free the returned pointer.  It should be statically allocated somewhere.
#ifdef __cplusplus
} //extern "C" {
#endif
