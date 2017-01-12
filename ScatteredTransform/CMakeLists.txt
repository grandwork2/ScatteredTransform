
#-----------------------------------------------------------------------------
set(MODULE_NAME ScatteredTransform)

#-----------------------------------------------------------------------------

#
# SlicerExecutionModel
#

find_package(Boost REQUIRED)



#-----------------------------------------------------------------------------
set(MODULE_INCLUDE_DIRECTORIES
  ${vtkITK_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  )

set(MODULE_SRCS
  mba.hpp
  )

set(MODULE_TARGET_LIBRARIES
  ${ITK_LIBRARIES}
  )

#-----------------------------------------------------------------------------
SEMMacroBuildCLI(
  NAME ${MODULE_NAME}
  TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
  INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
  ADDITIONAL_SRCS ${MODULE_SRCS}
  )

#-----------------------------------------------------------------------------

