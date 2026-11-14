// Linker guards catch even uncontended pthread locks and allocator calls that
// would otherwise evade syscall-only tracing. Used only by the Linux tests.
#include <cstddef>
#include <cstdlib>
#include <pthread.h>
extern bool cv_daw_test_forbid_runtime_calls;
extern "C" {
void* __real_malloc(std::size_t);void* __real_calloc(std::size_t,std::size_t);
void* __real_realloc(void*,std::size_t);void __real_free(void*);
int __real_pthread_mutex_lock(pthread_mutex_t*);
int __real_pthread_mutex_trylock(pthread_mutex_t*);
int __real_pthread_mutex_unlock(pthread_mutex_t*);
void* __wrap_malloc(std::size_t n){if(cv_daw_test_forbid_runtime_calls)std::abort();return __real_malloc(n);}
void* __wrap_calloc(std::size_t n,std::size_t s){if(cv_daw_test_forbid_runtime_calls)std::abort();return __real_calloc(n,s);}
void* __wrap_realloc(void* p,std::size_t n){if(cv_daw_test_forbid_runtime_calls)std::abort();return __real_realloc(p,n);}
void __wrap_free(void* p){if(cv_daw_test_forbid_runtime_calls)std::abort();__real_free(p);}
int __wrap_pthread_mutex_lock(pthread_mutex_t* p){if(cv_daw_test_forbid_runtime_calls)std::abort();return __real_pthread_mutex_lock(p);}
int __wrap_pthread_mutex_trylock(pthread_mutex_t* p){if(cv_daw_test_forbid_runtime_calls)std::abort();return __real_pthread_mutex_trylock(p);}
int __wrap_pthread_mutex_unlock(pthread_mutex_t* p){if(cv_daw_test_forbid_runtime_calls)std::abort();return __real_pthread_mutex_unlock(p);}
}
