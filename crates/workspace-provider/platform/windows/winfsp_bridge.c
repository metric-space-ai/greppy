#define FUSE_USE_VERSION 31

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fuse.h>

typedef struct GreppyWindowsStat
{
    uint32_t mode;
    uint64_t size;
    uint64_t inode;
    uint32_t nlink;
    int64_t accessed_unix_ns;
    int64_t modified_unix_ns;
    int64_t changed_unix_ns;
} GreppyWindowsStat;

typedef int (*GreppyDirectoryEmitter)(void *, const char *, const GreppyWindowsStat *, uint64_t);

extern int greppy_windows_getattr(void *, const char *, GreppyWindowsStat *);
extern int greppy_windows_getattr_handle(void *, uint64_t, GreppyWindowsStat *);
extern int greppy_windows_open(void *, const char *, int, uint64_t *);
extern int greppy_windows_release(void *, uint64_t);
extern int greppy_windows_readdir(void *, const char *, uint64_t, void *, GreppyDirectoryEmitter);
extern int greppy_windows_create(void *, const char *, uint32_t, int);
extern int greppy_windows_unlink(void *, const char *, int);
extern int greppy_windows_rename(void *, const char *, const char *, uint32_t);
extern int greppy_windows_chmod(void *, const char *, uint32_t);
extern int greppy_windows_truncate(void *, const char *, uint64_t);
extern int greppy_windows_truncate_handle(void *, uint64_t, uint64_t);
extern int greppy_windows_read(void *, const char *, uint64_t, uint8_t *, size_t);
extern int greppy_windows_read_handle(void *, uint64_t, uint64_t, uint8_t *, size_t);
extern int greppy_windows_write(void *, const char *, uint64_t, const uint8_t *, size_t);
extern int greppy_windows_write_handle(void *, uint64_t, uint64_t, const uint8_t *, size_t);
extern int greppy_windows_symlink(void *, const char *, const char *);
extern int greppy_windows_hardlink(void *, const char *, const char *);
extern int greppy_windows_readlink(void *, const char *, uint8_t *, size_t);
extern int greppy_windows_set_times(void *, const char *, int64_t, int64_t);

static void *greppy_context(void)
{
    return fuse_get_context()->private_data;
}

static void copy_stat(struct fuse_stat *destination, const GreppyWindowsStat *source)
{
    memset(destination, 0, sizeof(*destination));
    destination->st_mode = source->mode;
    destination->st_size = source->size;
    destination->st_ino = source->inode;
    destination->st_nlink = source->nlink;
    destination->st_atim.tv_sec = source->accessed_unix_ns / 1000000000;
    destination->st_atim.tv_nsec = source->accessed_unix_ns % 1000000000;
    destination->st_mtim.tv_sec = source->modified_unix_ns / 1000000000;
    destination->st_mtim.tv_nsec = source->modified_unix_ns % 1000000000;
    destination->st_ctim.tv_sec = source->changed_unix_ns / 1000000000;
    destination->st_ctim.tv_nsec = source->changed_unix_ns % 1000000000;
}

static int greppy_getattr(const char *path, struct fuse_stat *value, struct fuse_file_info *file)
{
    GreppyWindowsStat portable;
    int result = 0 != file && 0 != file->fh
        ? greppy_windows_getattr_handle(greppy_context(), file->fh, &portable)
        : greppy_windows_getattr(greppy_context(), path, &portable);
    if (0 == result)
        copy_stat(value, &portable);
    return result;
}

typedef struct GreppyEmitContext
{
    void *buffer;
    fuse_fill_dir_t filler;
} GreppyEmitContext;

static int greppy_emit_directory(void *opaque, const char *name,
    const GreppyWindowsStat *portable, uint64_t next_offset)
{
    GreppyEmitContext *context = opaque;
    struct fuse_stat value;
    copy_stat(&value, portable);
    return context->filler(context->buffer, name, &value, (fuse_off_t)next_offset,
        FUSE_FILL_DIR_PLUS);
}

static int greppy_readdir(const char *path, void *buffer, fuse_fill_dir_t filler,
    fuse_off_t offset, struct fuse_file_info *file, enum fuse_readdir_flags flags)
{
    GreppyEmitContext context = {buffer, filler};
    (void)file;
    (void)flags;
    if (offset < 0)
        return -EINVAL;
    return greppy_windows_readdir(greppy_context(), path, (uint64_t)offset,
        &context, greppy_emit_directory);
}

static int greppy_mkdir(const char *path, fuse_mode_t mode)
{
    return greppy_windows_create(greppy_context(), path, mode, 1);
}

static int greppy_create(const char *path, fuse_mode_t mode, struct fuse_file_info *file)
{
    int result = greppy_windows_create(greppy_context(), path, mode, 0);
    if (0 != result)
        return result;
    return greppy_windows_open(greppy_context(), path,
        (file->flags & O_ACCMODE) == O_RDONLY, &file->fh);
}

static int greppy_open(const char *path, struct fuse_file_info *file)
{
    return greppy_windows_open(greppy_context(), path,
        (file->flags & O_ACCMODE) == O_RDONLY, &file->fh);
}

static int greppy_unlink(const char *path)
{
    return greppy_windows_unlink(greppy_context(), path, 0);
}

static int greppy_rmdir(const char *path)
{
    return greppy_windows_unlink(greppy_context(), path, 1);
}

static int greppy_rename(const char *source, const char *destination, unsigned int flags)
{
    return greppy_windows_rename(greppy_context(), source, destination, flags);
}

static int greppy_chmod(const char *path, fuse_mode_t mode, struct fuse_file_info *file)
{
    (void)file;
    return greppy_windows_chmod(greppy_context(), path, mode);
}

static int greppy_truncate(const char *path, fuse_off_t size, struct fuse_file_info *file)
{
    if (size < 0)
        return -EINVAL;
    return 0 != file && 0 != file->fh
        ? greppy_windows_truncate_handle(greppy_context(), file->fh, (uint64_t)size)
        : greppy_windows_truncate(greppy_context(), path, (uint64_t)size);
}

static int greppy_read(const char *path, char *buffer, size_t size, fuse_off_t offset,
    struct fuse_file_info *file)
{
    if (offset < 0)
        return -EINVAL;
    return 0 != file && 0 != file->fh
        ? greppy_windows_read_handle(greppy_context(), file->fh, (uint64_t)offset,
            (uint8_t *)buffer, size)
        : greppy_windows_read(greppy_context(), path, (uint64_t)offset,
            (uint8_t *)buffer, size);
}

static int greppy_write(const char *path, const char *buffer, size_t size, fuse_off_t offset,
    struct fuse_file_info *file)
{
    if (offset < 0)
        return -EINVAL;
    return 0 != file && 0 != file->fh
        ? greppy_windows_write_handle(greppy_context(), file->fh, (uint64_t)offset,
            (const uint8_t *)buffer, size)
        : greppy_windows_write(greppy_context(), path, (uint64_t)offset,
            (const uint8_t *)buffer, size);
}

static int greppy_symlink(const char *target, const char *path)
{
    return greppy_windows_symlink(greppy_context(), path, target);
}

static int greppy_link(const char *source, const char *destination)
{
    return greppy_windows_hardlink(greppy_context(), source, destination);
}

static int greppy_readlink(const char *path, char *buffer, size_t size)
{
    if (0 == size)
        return -ERANGE;
    int result = greppy_windows_readlink(greppy_context(), path, (uint8_t *)buffer, size - 1);
    if (result < 0)
        return result;
    buffer[result] = '\0';
    return 0;
}

static int greppy_utimens(const char *path, const struct fuse_timespec times[2],
    struct fuse_file_info *file)
{
    (void)file;
    return greppy_windows_set_times(greppy_context(), path,
        (int64_t)times[0].tv_sec * 1000000000 + times[0].tv_nsec,
        (int64_t)times[1].tv_sec * 1000000000 + times[1].tv_nsec);
}

static int greppy_statfs(const char *path, struct fuse_statvfs *value)
{
    (void)path;
    memset(value, 0, sizeof(*value));
    value->f_bsize = 1048576;
    value->f_frsize = 1048576;
    value->f_blocks = UINT64_MAX / 1048576;
    value->f_bfree = value->f_blocks;
    value->f_bavail = value->f_blocks;
    value->f_namemax = 255;
    return 0;
}

static int greppy_release(const char *path, struct fuse_file_info *file)
{
    (void)path;
    return 0 != file && 0 != file->fh
        ? greppy_windows_release(greppy_context(), file->fh)
        : 0;
}

static int greppy_fsync(const char *path, int data_only, struct fuse_file_info *file)
{
    (void)path;
    (void)data_only;
    (void)file;
    return 0;
}

static struct fuse_operations greppy_operations =
{
    .getattr = greppy_getattr,
    .readlink = greppy_readlink,
    .mkdir = greppy_mkdir,
    .unlink = greppy_unlink,
    .rmdir = greppy_rmdir,
    .symlink = greppy_symlink,
    .rename = greppy_rename,
    .link = greppy_link,
    .chmod = greppy_chmod,
    .truncate = greppy_truncate,
    .open = greppy_open,
    .read = greppy_read,
    .write = greppy_write,
    .statfs = greppy_statfs,
    .release = greppy_release,
    .fsync = greppy_fsync,
    .readdir = greppy_readdir,
    .create = greppy_create,
    .utimens = greppy_utimens,
};

int greppy_winfsp_mount(void *context, const char *mountpoint)
{
    char program[] = "greppy-workspace-provider";
    char foreground[] = "-f";
    char debug[] = "-d";
    char option[] = "-o";
    char filesystem_name[] = "FileSystemName=greppy-cow,volname=Greppy Workspaces,uid=-1,gid=-1";
    char *regular_arguments[] = {program, foreground, option, filesystem_name, (char *)mountpoint, 0};
    char *debug_arguments[] = {program, foreground, debug, option, filesystem_name, (char *)mountpoint, 0};
    if (0 != getenv("GREPPY_WINFSP_DEBUG"))
        return fuse_main(6, debug_arguments, &greppy_operations, context);
    return fuse_main(5, regular_arguments, &greppy_operations, context);
}
