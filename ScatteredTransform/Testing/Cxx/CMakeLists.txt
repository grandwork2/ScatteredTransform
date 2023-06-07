
#-----------------------------------------------------------------------------
set(INPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../Data/Input)
set(TEMP "${CMAKE_BINARY_DIR}/Testing/Temporary")

set(CLP ${MODULE_NAME})

#-----------------------------------------------------------------------------
add_executable(${CLP}Test ${CLP}Test.cxx)
target_link_libraries(${CLP}Test ${CLP}Lib ${SlicerExecutionModel_EXTRA_EXECUTABLE_TARGET_LIBRARIES})
set_target_properties(${CLP}Test PROPERTIES LABELS ${CLP})

#-----------------------------------------------------------------------------
set(testname ${CLP}Test)
add_test(NAME ${testname} COMMAND ${SEM_LAUNCH_COMMAND} $<TARGET_FILE:${CLP}Test>
  ModuleEntryPoint
  --initialPointsFile ${INPUT}/points.vtp --displacedPointsFile ${INPUT}/points_displaced.vtp
  --bsplineTransformFile ${TEMP}/${CLP}Test.tfm --transformSpaceDimension 3D  
  --invertTransform --useLinearApproximation --splineGridSpacing 100,100,30 --domainFromInputPoints 
  --minCoordinates 0,0,0 --maxCoordinates 100,100,100 --tolerance 0.1 --minGridSpacing 0.1 --maxNumLevels 3 
  )

set_property(TEST ${testname} PROPERTY LABELS ${CLP})

#-----------------------------------------------------------------------------
