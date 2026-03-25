use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const HOOKS: [&str; 5] = [
    "pre-commit",
    "commit-msg",
    "pre-push",
    "post-checkout",
    "post-merge",
];

fn git_output(args: &[&str], repo_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo_root)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn git_dir_path(repo_root: &Path) -> Option<PathBuf> {
    git_output(&["rev-parse", "--git-dir"], repo_root).map(|path| {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            repo_root.join(path)
        }
    })
}

fn install_warning(message: impl AsRef<str>) {
    println!("cargo:warning={}", message.as_ref());
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let entry_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry_path, &dst_path)?;
        } else {
            fs::copy(&entry_path, &dst_path)?;
        }
    }
    Ok(())
}

fn file_contents(path: &Path) -> Option<Vec<u8>> {
    fs::read(path).ok()
}

#[cfg(unix)]
fn make_executable(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

fn resolve_commithooks_source() -> Option<PathBuf> {
    if let Ok(path) = env::var("COMMITHOOKS_DIR") {
        let candidate = PathBuf::from(path);
        if candidate.join("lib/common.sh").is_file() {
            return Some(candidate);
        }
    }

    let home = env::var_os("HOME")?;
    let candidate = PathBuf::from(home).join("Documents/commithooks");
    candidate
        .join("lib/common.sh")
        .is_file()
        .then_some(candidate)
}

fn hook_install_is_blocked(git_dir: &Path) -> bool {
    git_dir.join("rebase-merge").exists()
        || git_dir.join("rebase-apply").exists()
        || git_dir.join("MERGE_HEAD").exists()
}

fn install_commithooks(repo_root: &Path) {
    let Some(source) = resolve_commithooks_source() else {
        install_warning("commithooks source not found; skipping hook install");
        return;
    };

    let Some(git_dir) = git_dir_path(repo_root) else {
        install_warning("not in a git repository; skipping hook install");
        return;
    };

    if hook_install_is_blocked(&git_dir) {
        install_warning("active rebase/merge detected; skipping hook install");
        return;
    }

    let hooks_dir = git_dir.join("hooks");
    if let Err(err) = fs::create_dir_all(&hooks_dir) {
        install_warning(format!(
            "failed to create hooks dir {}: {err}",
            hooks_dir.display()
        ));
        return;
    }

    for hook in HOOKS {
        let src = source.join(hook);
        if !src.is_file() {
            continue;
        }

        let dst = hooks_dir.join(hook);
        let sample = hooks_dir.join(format!("{hook}.sample"));
        let dst_bytes = file_contents(&dst);
        let sample_bytes = file_contents(&sample);
        let src_bytes = file_contents(&src);

        let is_custom = match (&dst_bytes, &sample_bytes, &src_bytes) {
            (Some(dst_bytes), Some(sample_bytes), Some(src_bytes)) => {
                dst_bytes != sample_bytes && dst_bytes != src_bytes
            }
            (Some(dst_bytes), None, Some(src_bytes)) => dst_bytes != src_bytes,
            _ => false,
        };
        if is_custom {
            install_warning(format!(
                "preserving custom hook {}; not overwriting {}",
                hook,
                dst.display()
            ));
            continue;
        }

        if let Err(err) = fs::copy(&src, &dst).and_then(|_| make_executable(&dst)) {
            install_warning(format!("failed to install hook {}: {err}", dst.display()));
        }
    }

    let source_lib = source.join("lib");
    if !source_lib.is_dir() {
        install_warning(format!(
            "commithooks lib missing at {}; skipping hook install",
            source_lib.display()
        ));
        return;
    }

    let lib_dst = git_dir.join("lib");
    if lib_dst.exists() {
        if let Err(err) = fs::remove_dir_all(&lib_dst) {
            install_warning(format!(
                "failed to refresh hook lib {}: {err}",
                lib_dst.display()
            ));
            return;
        }
    }

    if let Err(err) = copy_dir_recursive(&source_lib, &lib_dst) {
        install_warning(format!(
            "failed to copy hook library into {}: {err}",
            lib_dst.display()
        ));
        return;
    }

    let _ = Command::new("git")
        .args(["config", "--unset", "core.hooksPath"])
        .current_dir(repo_root)
        .output();
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=.githooks/pre-commit");
    println!("cargo:rerun-if-changed=.githooks/commit-msg");
    println!("cargo:rerun-if-changed=.githooks/pre-push");
    println!("cargo:rerun-if-changed=.githooks/post-checkout");
    println!("cargo:rerun-if-changed=.githooks/post-merge");
    println!("cargo:rerun-if-env-changed=COMMITHOOKS_DIR");

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    if manifest_dir.is_empty() {
        return;
    }

    let repo_root = Path::new(&manifest_dir);
    install_commithooks(repo_root);
}
