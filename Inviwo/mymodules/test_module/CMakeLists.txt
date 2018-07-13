#--------------------------------------------------------------------
# Inviwo test_module Module
ivw_module(test_module)

#--------------------------------------------------------------------
# Add header files
set(HEADER_FILES
    #${CMAKE_CURRENT_SOURCE_DIR}/test_moduleprocessor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/my_test_class.h
    ${CMAKE_CURRENT_SOURCE_DIR}/my_test_class_w_skeleton.h
    ${CMAKE_CURRENT_SOURCE_DIR}/processor.h
)
ivw_group("Header Files" ${HEADER_FILES})

#--------------------------------------------------------------------
# Add source files
set(SOURCE_FILES
    #${CMAKE_CURRENT_SOURCE_DIR}/test_moduleprocessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/my_test_class.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/my_test_class_w_skeleton.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/processor.cpp
)
ivw_group("Source Files" ${SOURCE_FILES})


#--------------------------------------------------------------------
# Add shaders
set(SHADER_FILES
    #${CMAKE_CURRENT_SOURCE_DIR}/glsl/test_moduleprocessor.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl/my_test_class.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl/my_test_class.geom
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl/my_test_class.vert
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl/my_test_class_w_skeleton.frag
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl/my_test_class_w_skeleton.geom
    ${CMAKE_CURRENT_SOURCE_DIR}/glsl/my_test_class_w_skeleton.vert
)
ivw_group("Shader Files" ${SHADER_FILES})


#--------------------------------------------------------------------
# Add Unittests
set(TEST_FILES
	#${CMAKE_CURRENT_SOURCE_DIR}/tests/unittests/test_module-test.cpp
)
ivw_add_unittest(${TEST_FILES})

#--------------------------------------------------------------------
# Create module
ivw_create_module(${SOURCE_FILES} ${HEADER_FILES} ${SHADER_FILES})

#--------------------------------------------------------------------
# Add shader directory to pack
# ivw_add_to_module_pack(${CMAKE_CURRENT_SOURCE_DIR}/glsl)