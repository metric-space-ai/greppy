/* Test-only fault at response pooling, not at connect/request setup. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static _Atomic int response_socket = -1;

static void received_response(int fd, const void *buffer, ssize_t length) {
    if (length >= 12 && memcmp(buffer, "HTTP/1.1 502", 12) == 0) {
        atomic_store(&response_socket, fd);
    }
}

ssize_t recv(int fd, void *buffer, size_t length, int flags) {
    ssize_t (*next_recv)(int, void *, size_t, int) = dlsym(RTLD_NEXT, "recv");
    if (next_recv == NULL) { errno = ENOSYS; return -1; }
    ssize_t result = next_recv(fd, buffer, length, flags);
    received_response(fd, buffer, result);
    return result;
}

ssize_t read(int fd, void *buffer, size_t length) {
    ssize_t (*next_read)(int, void *, size_t) = dlsym(RTLD_NEXT, "read");
    if (next_read == NULL) { errno = ENOSYS; return -1; }
    ssize_t result = next_read(fd, buffer, length);
    received_response(fd, buffer, result);
    return result;
}

int setsockopt(int fd, int level, int name, const void *value, socklen_t length) {
    if (fd == atomic_load(&response_socket) && level == SOL_SOCKET &&
        (name == SO_RCVTIMEO || name == SO_SNDTIMEO) &&
        value != NULL && length >= sizeof(struct timeval)) {
        const struct timeval *timeout = value;
        if (timeout->tv_sec == 0 && timeout->tv_usec == 0) {
            errno = EINVAL;
            return -1;
        }
    }
    int (*next_setsockopt)(int, int, int, const void *, socklen_t) =
        dlsym(RTLD_NEXT, "setsockopt");
    if (next_setsockopt == NULL) { errno = ENOSYS; return -1; }
    return next_setsockopt(fd, level, name, value, length);
}
