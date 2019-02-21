set (TELAJAX_SRC
	device_finalize.c
	device_init.c
	device_mem_alloc.c
	device_mem_read.c
	device_mem_release.c
	device_mem_write.c
	device_waitall.c
  handling_ocl_error.c
	kernel_build.c
	kernel_enqueue.c
	kernel_release.c
	kernel_set_dim.c
	kernel_set_args.c
	wrapper.c
	event_set_callback.c
	event_wait.c
	event_release.c
)

add_library(telajax SHARED ${TELAJAX_SRC})

set_target_properties(
	telajax
	PROPERTIES
		SOVERSION 0
		VERSION  ${TELAJAX_VERSION_FULL}
)

install(TARGETS telajax
  LIBRARY DESTINATION lib64
  ARCHIVE DESTINATION lib64
)

if(${BUILD_STATIC})
	add_library(telajax_static STATIC ${TELAJAX_SRC})
	install(TARGETS telajax_static
	  LIBRARY DESTINATION lib64
	  ARCHIVE DESTINATION lib64
	)
endif(${BUILD_STATIC})
