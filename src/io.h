/** \file
    File IO for nD-arrays.

    \author Nathan Clack
    \date   June 2012

    \section ndio-fmt Formats

    This gets a bit tricky because I'd like to write useful formats like tiff,
    mp4, etc... for arrays with appropriate dimensionality.  For arrays with
    more than 2 spatial dimensions, 3 colors, and time nice portable formats
    don't exist.  There's hdf5, matlab, npz, and custom formats.

    Another complication is how to deal with tilings, like the one I'm using
    for the microscope.  The tile id is kind-of a dimension.  There's data
    associated with the dimension such as the tile position.

    Color is another dimension that could also have associated metadata, such
    as the color name (red, blue, GFP, etc...), colormap, and so on.

    Maybe it's enough to provide a metadata facitily for a dimension.


    \section ndio-plugins Extension formats

    protocol?
    more trouble than it's worth?


    \section ndio-special-arrays Faux Arrays

    It's possible for ndioSeek, the primary read function, to return an array
    that holds a ndio_t reference in it's data field.  This ~could~ be used 
    to implement algorithms that are part of the nd library where the full array
    is stored on disk, and only required chunks are read in at any particular time.



    */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* An idea:

   Use a dimensions descriptor to reorder dimensions before
   writing so they are best matched to the output format.

   typedef enum _nd_dim_desc_t
  { nd_dim_nodesc=0,
    nd_x,
    nd_y,
    nd_z,
    nd_t,
    nd_color,
    nd_tile,
    nd_dim_max_desc
  } nd_dim_desc_t;
*/

typedef struct _ndio_t* ndio_t;
typedef struct _nd_t*   nd_t;

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

#ifdef __cplusplus
} //extern "C" {
#endif
