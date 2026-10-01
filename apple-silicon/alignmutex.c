#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

/* Rosetta for Linux translates x86 `lock cmpxchg` into ARM atomics, which
   fault (SIGBUS) when the operand crosses a 16-byte boundary; x86 does not.
   SC2 4.10 keeps pthread mutexes at 2-byte-aligned addresses, so glibc's
   pthread_mutex_lock dies at map load. Route every misaligned mutex to an
   aligned shadow mutex; aligned mutexes pass straight through. */

#define TABLE (1 << 16)
struct slot { uintptr_t key; pthread_mutex_t *shadow; };
static struct slot table[TABLE];
static pthread_mutex_t table_lock = PTHREAD_MUTEX_INITIALIZER;

static int (*r_init)(pthread_mutex_t *, const pthread_mutexattr_t *);
static int (*r_destroy)(pthread_mutex_t *);
static int (*r_lock)(pthread_mutex_t *);
static int (*r_trylock)(pthread_mutex_t *);
static int (*r_timedlock)(pthread_mutex_t *, const struct timespec *);
static int (*r_unlock)(pthread_mutex_t *);
static int (*r_cwait)(pthread_cond_t *, pthread_mutex_t *);
static int (*r_ctimedwait)(pthread_cond_t *, pthread_mutex_t *, const struct timespec *);
static int log_on;

__attribute__((constructor)) static void setup(void) {
    r_init = dlsym(RTLD_NEXT, "pthread_mutex_init");
    r_destroy = dlsym(RTLD_NEXT, "pthread_mutex_destroy");
    r_lock = dlsym(RTLD_NEXT, "pthread_mutex_lock");
    r_trylock = dlsym(RTLD_NEXT, "pthread_mutex_trylock");
    r_timedlock = dlsym(RTLD_NEXT, "pthread_mutex_timedlock");
    r_unlock = dlsym(RTLD_NEXT, "pthread_mutex_unlock");
    r_cwait = dlvsym(RTLD_NEXT, "pthread_cond_wait", "GLIBC_2.3.2");
    if (!r_cwait) r_cwait = dlsym(RTLD_NEXT, "pthread_cond_wait");
    r_ctimedwait = dlvsym(RTLD_NEXT, "pthread_cond_timedwait", "GLIBC_2.3.2");
    if (!r_ctimedwait) r_ctimedwait = dlsym(RTLD_NEXT, "pthread_cond_timedwait");
    log_on = getenv("ALIGNMUTEX_LOG") != NULL;
}

static inline int misaligned(const void *m) { return ((uintptr_t)m & 3) != 0; }

/* The shadow for misaligned mutex m, created on first use with m's kind. */
static pthread_mutex_t *shadow(pthread_mutex_t *m, int create) {
    uintptr_t key = (uintptr_t)m;
    size_t i = (key >> 1) & (TABLE - 1);
    pthread_mutex_t *s = NULL;
    r_lock(&table_lock);
    for (size_t n = 0; n < TABLE; n++, i = (i + 1) & (TABLE - 1)) {
        if (table[i].key == key) { s = table[i].shadow; break; }
        if (table[i].key == 0) {
            if (!create) break;
            int kind;
            memcpy(&kind, (char *)m + 16, sizeof kind);      /* __data.__kind */
            pthread_mutexattr_t a;
            pthread_mutexattr_init(&a);
            pthread_mutexattr_settype(&a, kind & 3);
            s = aligned_alloc(64, 64);
            r_init(s, &a);
            pthread_mutexattr_destroy(&a);
            table[i].key = key; table[i].shadow = s;
            if (log_on) fprintf(stderr, "alignmutex: shadow for %p kind %d\n", (void *)m, kind & 3);
            break;
        }
    }
    r_unlock(&table_lock);
    return s;
}

int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) {
    if (!r_init) setup();
    int rc = r_init(m, a);   /* plain stores; fine at any alignment */
    if (misaligned(m)) {     /* drop a stale shadow at this address */
        uintptr_t key = (uintptr_t)m; size_t i = (key >> 1) & (TABLE - 1);
        r_lock(&table_lock);
        for (size_t n = 0; n < TABLE && table[i].key; n++, i = (i + 1) & (TABLE - 1))
            if (table[i].key == key) { r_destroy(table[i].shadow); free(table[i].shadow); table[i].key = 1; break; }
        r_unlock(&table_lock);
    }
    return rc;
}
int pthread_mutex_destroy(pthread_mutex_t *m) {
    if (!r_destroy) setup();
    return misaligned(m) ? 0 : r_destroy(m);
}
int pthread_mutex_lock(pthread_mutex_t *m) {
    if (!r_lock) setup();
    return misaligned(m) ? r_lock(shadow(m, 1)) : r_lock(m);
}
int pthread_mutex_trylock(pthread_mutex_t *m) {
    if (!r_trylock) setup();
    return misaligned(m) ? r_trylock(shadow(m, 1)) : r_trylock(m);
}
int pthread_mutex_timedlock(pthread_mutex_t *m, const struct timespec *t) {
    if (!r_timedlock) setup();
    return misaligned(m) ? r_timedlock(shadow(m, 1), t) : r_timedlock(m, t);
}
int pthread_mutex_unlock(pthread_mutex_t *m) {
    if (!r_unlock) setup();
    return misaligned(m) ? r_unlock(shadow(m, 1)) : r_unlock(m);
}
int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    if (!r_cwait) setup();
    return r_cwait(c, misaligned(m) ? shadow(m, 1) : m);
}
int pthread_cond_timedwait(pthread_cond_t *c, pthread_mutex_t *m, const struct timespec *t) {
    if (!r_ctimedwait) setup();
    return r_ctimedwait(c, misaligned(m) ? shadow(m, 1) : m, t);
}
