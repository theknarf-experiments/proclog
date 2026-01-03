use internment::Intern;
use std::{
    fmt,
    path::{Path, PathBuf},
};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SrcId(Intern<Vec<String>>);

impl SrcId {
    pub fn empty() -> Self {
        Self(Intern::new(Vec::new()))
    }

    pub fn repl() -> Self {
        Self(Intern::new(vec!["repl".to_string()]))
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        Self(Intern::new(
            path.as_ref()
                .iter()
                .map(|segment| segment.to_string_lossy().into_owned())
                .collect(),
        ))
    }

    pub fn to_path(&self) -> PathBuf {
        self.0.iter().map(|segment| segment.to_string()).collect()
    }
}

impl fmt::Display for SrcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, "?")
        } else {
            write!(f, "{}", self.0.join("/"))
        }
    }
}

impl fmt::Debug for SrcId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}
