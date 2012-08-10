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
 *  \todo Add some facility to set paramters. e.g. Tiff Compression, or bitrate for a lossy compression.
 *        Should be some key-value store, or each plugin should expose a special header that exposes a
 *        parameters struct.
 *  \todo Describe the contract for each interface function in more detail.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif
  //Required: include io.h and core.h before this file
  //typedef struct _ndio_t* ndio_t;
  //typedef struct _nd_t*   nd_t;

  typedef const char* (*_ndio__fmt_name_t)(void);                               ///< Plugin name. \returns a string with the format's name.
  typedef unsigned    (*_ndio__is_fmt_t)(const char* path, const char *mode);   ///< Determines format support for file at \a path. \returns 1 if the format supports reading the file at \a path or writing a file with the extension indicated by \path.  Returns 0 otherwise.
  typedef void*       (*_ndio__open_t  )(const char* path, const char *mode);   ///< Opens the file at \a path according to the \a mode. \returns a the address of the file context on success, otherwise 0.
  typedef void        (*_ndio__close_t )(ndio_t file);                          ///< Closes the file and releases any internally acquired resources.  Should always succeed.
  typedef nd_t        (*_ndio__shape_t )(ndio_t file);                          ///< Gets the shape of the data in \a file. \returns 0 on failure, otherwise returns the shape, dimensions, and type of the data in a dummy nd_t object.  Caller is responsible for freeing the returned object.
  typedef unsigned    (*_ndio__read_t  )(ndio_t file, nd_t dst);                ///< Reads the data in \a file to \a a. \returns 0 on failure, otherwise returns \a self.  \param[out] dst Destination container for the file's contents. Should have been allocated with enough storage to hold the data.
  typedef unsigned    (*_ndio__write_t )(ndio_t file, nd_t src);                ///< Writes the data in \a a to \a file. \returns 0 on failure, otherwise returns \a self.  \param[in]  src The data to save to the file.  Should be treated as an append operation.  Append vs overwrite behavior should be handled on opening the file.
  typedef unsigned    (*_ndio__set_t)(ndio_t file, void *param, size_t nbytes); ///< (optional) Set format specific options or metadata.
  typedef void*       (*_ndio__get_t)(ndio_t file);                             ///< (optional) Get format specific options or metadata.

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
    _ndio__read_t     read;       ///< \returns 0 on failure, otherwise returns 1.  \param[out] dst Destination container for the file's contents. Should have been allocated with enough storage to hold the data.
    _ndio__write_t    write;      ///< \returns 0 on failure, otherwise returns 1.  \param[in]  src The data to save to the file.  Should be treated as an append operation.  Append vs overwrite behavior should be handled on opening the file.
    _ndio__set_t      set;        ///< Set format specific data.  \returns 0 on failure, otherwise returns 1.
    _ndio__get_t      get;        ///< Get format specific data.  \returns 0 on failure, otherwise returns 1.
    void*             lib;        ///< Context.  Used internally. Should be NULL when returned from a ndio_get_format_api() call.
  };

  typedef const ndio_fmt_t* (*ndio_get_format_api_t)(void); ///< Produces a plugin's interface.  \returns the api used to implement the format.  The caller will not free the returned pointer.  It should be statically allocated somewhere.
#ifdef __cplusplus
} //extern "C" {
#endif
