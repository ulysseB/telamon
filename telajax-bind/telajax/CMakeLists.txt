cmake_minimum_required(VERSION 2.8)

project(Telajax)
set (TELAJAX_VERSION_MAJOR 1)
set (TELAJAX_VERSION_MINOR 0)
set (TELAJAX_VERSION_PATCH 0)

set(TELAJAX_VERSION_FULL
	${TELAJAX_VERSION_MAJOR}.${TELAJAX_VERSION_MINOR}.${TELAJAX_VERSION_PATCH})


option(BUILD_STATIC "BUILD_STATIC" OFF)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})

find_package(OpenCL REQUIRED)

link_directories(${OpenCL_LIBRARIES})

include_directories(./include ${OpenCL_INCLUDE_DIR})

set (CMAKE_C_FLAGS " -g -std=gnu99 ${CMAKE_C_FLAGS} ")

add_subdirectory(src)
add_subdirectory(tests)

install(FILES ./include/telajax.h DESTINATION include)

# uninstall target
configure_file(
	"${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
	"${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	IMMEDIATE @ONLY)
add_custom_target(uninstall
	COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)

