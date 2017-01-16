cmake_minimum_required(VERSION 2.8.9)

project(ScatteredTransform)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/ScatteredTransform")
set(EXTENSION_CATEGORY "Transforms")
set(EXTENSION_CONTRIBUTORS "Grand Joldes (The University of Western Australia, ISML)")
set(EXTENSION_DESCRIPTION "Creates a BSpline transform from a displacement field defined at scattered points by using the Multi-level BSpline interpolation algorithm.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/grandwork2/ScatteredTransform/master/ScatteredTransform.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/grandwork2/ScatteredTransform/master/Screenshot.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ScatteredTransform)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_CPACK})
