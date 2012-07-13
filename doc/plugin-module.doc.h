/** \defgroup ndioplugins Plugins
 * File IO handled through the \a ndio_t interface is mediated by a
 * plugin-based system.
 *
 * Plugins are shared libraries that live in a specific location (so they may
 * be found at run-time) and that implement a specific interface.  The
 * interface expected here consists of a single function named
 * ndio_get_format_api() with the following type:
 *
 * \code
 * typedef const ndio_fmt_t* (*ndio_get_format_api_t)(void);
 * \endcode
 *
 * See io/interface.h for more.
 *
 * A plugin system is used for file IO because supporting a wide variety of
 * useful formats requires incorporating existing third-party libraries.  These
 * third-party libraries are dependencies that aren't related to the core
 * functionality provided by this library.  Using a plugin interface helps
 * decouple concerns about IO from the main library.
 */
