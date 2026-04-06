# GitHub MCP Tool Reference

## Available Tools

### create_or_update_file

Creates or updates a file in a GitHub repository.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `owner` | string | ✓ | Repository owner |
| `repo` | string | ✓ | Repository name |
| `path` | string | ✓ | File path within the repo |
| `content` | string | ✓ | File content (auto base64-encoded) |
| `message` | string | ✓ | Commit message |
| `branch` | string | ✓ | Target branch |
| `sha` | string | | Current file SHA (required for updates) |

**Usage for NEW files:**
```
create_or_update_file(
  owner="org",
  repo="terraform-modules",
  path="modules/{service}/main.tf",
  content="resource \"aws_...\" ...",
  message="feat({service}): add main.tf",
  branch="main"
)
```

**Usage for EXISTING files:**
```
create_or_update_file(
  owner="org",
  repo="terraform-modules",
  path="modules/{service}/main.tf",
  content="updated content...",
  message="fix({service}): update main.tf",
  branch="main",
  sha="abc123def456"  # From get_file_contents
)
```

### get_file_contents

Retrieves file content and metadata from a GitHub repository.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `owner` | string | ✓ | Repository owner |
| `repo` | string | ✓ | Repository name |
| `path` | string | ✓ | File path within the repo |
| `branch` | string | | Branch name (default: repo default branch) |

**Returns:**
- `content` — file content (base64-decoded)
- `sha` — current file SHA (needed for updates)
- `size` — file size in bytes
- `encoding` — content encoding

**Error cases:**
- 404: File does not exist → treat as new file, skip SHA

### list_directory_contents

Lists files and directories in a GitHub repository path.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `owner` | string | ✓ | Repository owner |
| `repo` | string | ✓ | Repository name |
| `path` | string | ✓ | Directory path within the repo |
| `branch` | string | | Branch name (default: repo default branch) |

**Returns:**
Array of entries, each with:
- `name` — file or directory name
- `path` — full path
- `type` — "file" or "dir"
- `sha` — object SHA

## Common Patterns

### Commit a New Module (Multiple Files)

```python
# 1. List local files to commit
files = ls("/workspace/terraform_modules/{service}/")

# 2. For each file, read and commit
for file in files:
    content = read_file(f"/workspace/terraform_modules/{service}/{file}")
    create_or_update_file(
        owner="org",
        repo="terraform-modules",
        path=f"modules/{service}/{file}",
        content=content,
        message=f"feat({service}): add {file}",
        branch="main"
    )
```

### Update an Existing Module

```python
# 1. Get current SHAs for all files
existing = list_directory_contents(owner="org", repo="terraform-modules", path="modules/{service}/")

# 2. Build SHA lookup
sha_map = {entry["name"]: entry["sha"] for entry in existing if entry["type"] == "file"}

# 3. For each modified file, commit with SHA
for file in modified_files:
    content = read_file(f"/workspace/terraform_modules/{service}/{file}")
    sha = sha_map.get(file)  # None for new files

    kwargs = {
        "owner": "org", "repo": "terraform-modules",
        "path": f"modules/{service}/{file}",
        "content": content,
        "message": f"fix({service}): update {file}",
        "branch": "main"
    }
    if sha:
        kwargs["sha"] = sha

    create_or_update_file(**kwargs)
```
