#ifndef GREPPY_WORKSPACE_H
#define GREPPY_WORKSPACE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GreppyWorkspaceCore GreppyWorkspaceCore;

typedef struct GreppyWorkspaceMetadata {
    uint8_t kind; /* 1=file, 2=directory, 3=symlink */
    uint32_t mode;
    uint64_t size;
    uint64_t inode;
    uint32_t nlink;
    int64_t accessed_unix_ns;
    int64_t modified_unix_ns;
    int64_t changed_unix_ns;
} GreppyWorkspaceMetadata;

GreppyWorkspaceCore *greppy_workspace_core_open(const char *absolute_data_root);
void greppy_workspace_core_close(GreppyWorkspaceCore *core);
int32_t greppy_workspace_create(GreppyWorkspaceCore *core, const char *workspace_id,
                                const char *absolute_repository);
int32_t greppy_workspace_remove(GreppyWorkspaceCore *core, const char *workspace_id);
int32_t greppy_workspace_metadata(GreppyWorkspaceCore *core, const char *workspace_id,
                                  const char *path, GreppyWorkspaceMetadata *out);
int64_t greppy_workspace_read(GreppyWorkspaceCore *core, const char *workspace_id,
                              const char *path, uint64_t offset, uint8_t *out,
                              size_t capacity);
int64_t greppy_workspace_read_symlink(GreppyWorkspaceCore *core, const char *workspace_id,
                                      const char *path, uint8_t *out, size_t capacity);
int64_t greppy_workspace_write(GreppyWorkspaceCore *core, const char *workspace_id,
                               const char *path, uint64_t offset, const uint8_t *bytes,
                               size_t length);
int32_t greppy_workspace_truncate(GreppyWorkspaceCore *core, const char *workspace_id,
                                  const char *path, uint64_t size);
/* valid bit 0=mode, bit 1=atime, bit 2=mtime */
int32_t greppy_workspace_set_metadata(GreppyWorkspaceCore *core, const char *workspace_id,
                                      const char *path, uint32_t valid, uint32_t mode,
                                      int64_t accessed_unix_ns, int64_t modified_unix_ns);
int32_t greppy_workspace_create_file(GreppyWorkspaceCore *core, const char *workspace_id,
                                     const char *path, uint32_t mode);
int32_t greppy_workspace_mkdir(GreppyWorkspaceCore *core, const char *workspace_id,
                               const char *path, uint32_t mode);
int32_t greppy_workspace_unlink(GreppyWorkspaceCore *core, const char *workspace_id,
                                const char *path);
int32_t greppy_workspace_rename(GreppyWorkspaceCore *core, const char *workspace_id,
                                const char *source, const char *destination);
int32_t greppy_workspace_hard_link(GreppyWorkspaceCore *core, const char *workspace_id,
                                   const char *source, const char *destination);
int32_t greppy_workspace_symlink(GreppyWorkspaceCore *core, const char *workspace_id,
                                 const char *path, const uint8_t *target, size_t target_len);
char *greppy_workspace_list_json(GreppyWorkspaceCore *core, const char *workspace_id,
                                 const char *path);
char *greppy_workspace_list_workspaces_json(GreppyWorkspaceCore *core);
char *greppy_workspace_last_error(void);
void greppy_workspace_string_free(char *value);

#ifdef __cplusplus
}
#endif
#endif
