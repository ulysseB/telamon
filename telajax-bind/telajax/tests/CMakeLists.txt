enable_testing()

set(TESTS_LIST
	simple_telajax
	hello_world
)

# # Build and add test
foreach(test ${TESTS_LIST})
	add_executable( ${test} ${test}.c )
	target_link_libraries( ${test} telajax ${OpenCL_LIBRARIES} )
	add_test( ${test} ${test} )
	set_tests_properties ( ${test} PROPERTIES TIMEOUT 10 )
endforeach(test)

# add 'test' target
add_custom_target(check
	COMMAND ${CMAKE_CTEST_COMMAND} -V
	DEPENDS
		${TESTS_LIST}
)
