// LD_PRELOAD heap counter: tracks live bytes held by malloc-family allocations
// (malloc_usable_size of each live chunk). The benchmark reads it through the
// weak symbol gef_heap_live_bytes() to measure the memory a compressed
// representation holds once built. Build: cc -O2 -shared -fPIC heapcount.c -o libheapcount.so
#define _GNU_SOURCE
#include <errno.h>
#include <malloc.h>
#include <stdatomic.h>
#include <stddef.h>
extern void *__libc_malloc(size_t);
extern void __libc_free(void *);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void *__libc_memalign(size_t, size_t);
static _Atomic long long live;
static inline void *track(void *p) { if (p) atomic_fetch_add_explicit(&live, (long long)malloc_usable_size(p), memory_order_relaxed); return p; }
static inline void untrack(void *p) { if (p) atomic_fetch_sub_explicit(&live, (long long)malloc_usable_size(p), memory_order_relaxed); }
size_t gef_heap_live_bytes(void) { long long v = atomic_load(&live); return v > 0 ? (size_t)v : 0; }
void *malloc(size_t n) { return track(__libc_malloc(n)); }
void free(void *p) { untrack(p); __libc_free(p); }
void *calloc(size_t a, size_t b) { return track(__libc_calloc(a, b)); }
void *realloc(void *p, size_t n) {
    if (!p) return malloc(n);
    if (!n) { free(p); return NULL; }
    size_t old = malloc_usable_size(p);
    void *q = __libc_realloc(p, n);
    if (q) { atomic_fetch_sub_explicit(&live, (long long)old, memory_order_relaxed); track(q); }
    return q;
}
void *memalign(size_t a, size_t n) { return track(__libc_memalign(a, n)); }
void *aligned_alloc(size_t a, size_t n) { return memalign(a, n); }
int posix_memalign(void **out, size_t a, size_t n) {
    void *p = memalign(a, n);
    if (!p) return ENOMEM;
    *out = p;
    return 0;
}
void *valloc(size_t n) { return memalign(4096, n); }
void *pvalloc(size_t n) { return memalign(4096, (n + 4095) & ~(size_t)4095); }
