CREATE TABLE definition_identity_overrides (
    project TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    PRIMARY KEY (project, qualified_name)
);
CREATE TABLE js_ts_reference_override_files (
    project TEXT NOT NULL,
    file_path TEXT NOT NULL,
    PRIMARY KEY (project, file_path)
);
