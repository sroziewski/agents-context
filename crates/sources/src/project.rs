//! Project identity for isolating sessions: queries from one project must not see another's.

use std::path::Path;

/// Normalised project key for a working directory: the enclosing git root if there is one,
/// otherwise the directory itself; forward slashes, no trailing slash, lower-cased.
///
/// The git root makes `D:\dev\teams` and `D:\dev\teams\src` one project, while separate repos
/// nested in a plain folder stay separate. Lower-casing is right because Windows paths are
/// case-insensitive and agents report the same directory with different drive-letter casing
/// (`D:\dev` vs `d:/dev`); this tool only runs on Windows.
pub fn key(cwd: &str) -> String {
    let path = Path::new(cwd);
    let root = git_root(path).unwrap_or(path);
    normalise(&root.to_string_lossy())
}

fn git_root(start: &Path) -> Option<&Path> {
    // `.git` is a directory in a normal clone and a file in worktrees and submodules.
    start.ancestors().find(|dir| dir.join(".git").exists())
}

pub fn normalise(path: &str) -> String {
    let p = path.replace('\\', "/").to_lowercase();
    let trimmed = p.trim_end_matches('/');
    // Keep "c:/" rather than collapsing a drive root to "c:".
    if trimmed.ends_with(':') { format!("{trimmed}/") } else { trimmed.to_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn normalise_unifies_separators_case_and_trailing_slash() {
        assert_eq!(normalise(r"D:\dev\My\Teams\"), "d:/dev/my/teams");
        assert_eq!(normalise("d:/dev/my/teams"), "d:/dev/my/teams");
        assert_eq!(normalise("D:/dev/my/teams//"), "d:/dev/my/teams");
    }

    #[test]
    fn normalise_keeps_drive_root() {
        assert_eq!(normalise(r"C:\"), "c:/");
        assert_eq!(normalise("C:"), "c:/");
    }

    #[test]
    fn subdirectory_of_git_repo_maps_to_repo_root() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        fs::create_dir_all(repo.join(".git")).unwrap();
        fs::create_dir_all(repo.join("src/deep")).unwrap();
        let from_sub = key(&repo.join("src/deep").to_string_lossy());
        assert_eq!(from_sub, key(&repo.to_string_lossy()));
        assert_eq!(from_sub, normalise(&repo.to_string_lossy()));
    }

    #[test]
    fn git_file_marks_worktree_root() {
        let tmp = tempfile::tempdir().unwrap();
        let wt = tmp.path().join("wt");
        fs::create_dir_all(wt.join("pkg")).unwrap();
        fs::write(wt.join(".git"), "gitdir: elsewhere").unwrap();
        assert_eq!(key(&wt.join("pkg").to_string_lossy()), normalise(&wt.to_string_lossy()));
    }

    #[test]
    fn sibling_repos_in_plain_folder_stay_separate() {
        let tmp = tempfile::tempdir().unwrap();
        for r in ["a", "b"] {
            fs::create_dir_all(tmp.path().join(r).join(".git")).unwrap();
        }
        let a = key(&tmp.path().join("a").to_string_lossy());
        let b = key(&tmp.path().join("b").to_string_lossy());
        assert_ne!(a, b);
    }

    #[test]
    fn directory_without_repo_is_its_own_project() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("plain");
        fs::create_dir_all(&dir).unwrap();
        // The temp dir itself must not sit inside a repo for this to be meaningful.
        assert!(git_root(&dir).is_none());
        assert_eq!(key(&dir.to_string_lossy()), normalise(&dir.to_string_lossy()));
    }

    #[test]
    fn missing_directory_falls_back_to_normalised_path() {
        assert_eq!(key(r"Z:\gone\project"), "z:/gone/project");
    }
}
