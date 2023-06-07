
#-----------------------------------------------------------------------------
set(MODULE_NAME ScatteredTransform)

#-----------------------------------------------------------------------------

#
# SlicerExecutionModel
#

#-----------------------------------------------------------------------------
set(MODULE_INCLUDE_DIRECTORIES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MRMLCore_INCLUDE_DIRS}
  )

set(MODULE_SRCS
  mba.hpp
  )

set(MODULE_TARGET_LIBRARIES
  ${ITK_LIBRARIES}
  MRMLCore
  )

#-----------------------------------------------------------------------------
SEMMacroBuildCLI(
  NAME ${MODULE_NAME}
  TARGET_LIBRARIES ${MODULE_TARGET_LIBRARIES}
  INCLUDE_DIRECTORIES ${MODULE_INCLUDE_DIRECTORIES}
  ADDITIONAL_SRCS ${MODULE_SRCS}
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  add_subdirectory(Testing)
endif()
