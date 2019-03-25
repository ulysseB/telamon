#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
// FIXME: THREAD = 1 => invalid C
{candidate_id}

// #include <utask.h> // Provides <pthread.h> TODO(pthread)

#define N_THREADS {num_threads}
{def_time}

// Store arguments values.
static struct {{
  {arg_struct_fields}
#ifdef TIMING
  uint64_t* __time_ptr;
#endif
}} arguments;

// Barrier.
// pthread_barrier_t barrier; TODO(pthread)

static void* {name}_thread(void *arg) {{
  {var_decls}
  uint32_t code_id = (uint32_t) arg; // FIXME: Use mppa API instead ?
  {thread_idxs}
  // FIXME: memory privatisation
#ifdef TIMING
  uint64_t time;
  // pthread_barrier_wait(&barrier); TODO(pthread)
  if (code_id == 0) {{
    time = __k1_read_dsu_timestamp();
  }}
#endif // TIMING
  {body}
#ifdef TIMING
  // pthread_barrier_wait(&barrier); TODO(pthread)
  if (core_id == 0) {{
    time = __k1_read_dsu_timestamp() - time;
    *arguments.__time_ptr = time; // TODO(async): use async API
  }}
#endif
  return NULL;
}}

void {name}({kernel_args}) {{
  {arg_struct_build}
#ifdef TIMING
  arguments.__time_ptr = __time_ptr;
#endif
  // Initialize threads TODO(pthread)
  /*assert(!pthread_barrier_init(&barrier, NULL, N_THREADS));
  pthread_t threads[N_THREADS-1];
  for(int i=1; i<N_THREADS; ++i) {{
    assert(!pthread_create(&threads[i-1], NULL, {name}_thread, (void*)i));
  }}*/
  // Do the work
  {name}_thread(0);
  // Wait for spwanned threads TODO(pthread)
  /*for(int i=1; i<N_THREADS; ++i) {{
    assert(!pthread_join(threads[i-1], NULL));
  }}
  assert(!pthread_barrier_destroy(&barrier));*/
}}
