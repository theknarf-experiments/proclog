use crate::SrcId;
use std::{
    fmt,
    ops::Range,
};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    src: SrcId,
    range: (usize, usize),
}

impl Span {
    pub fn new(src: SrcId, range: Range<usize>) -> Self {
        assert!(range.start <= range.end);
        Self {
            src,
            range: (range.start, range.end),
        }
    }

    pub fn src(&self) -> SrcId {
        self.src
    }

    pub fn start(&self) -> usize {
        self.range.0
    }

    pub fn end(&self) -> usize {
        self.range.1
    }

    pub fn range(&self) -> Range<usize> {
        self.start()..self.end()
    }

    pub fn union(self, other: Self) -> Self {
        assert_eq!(self.src, other.src, "span source ids must match");
        Self {
            src: self.src,
            range: (self.start().min(other.start()), self.end().max(other.end())),
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{:?}", self.src, self.range())
    }
}

impl chumsky::Span for Span {
    type Context = SrcId;
    type Offset = usize;

    fn new(src: SrcId, range: Range<usize>) -> Self {
        Span::new(src, range)
    }

    fn context(&self) -> Self::Context {
        self.src
    }

    fn start(&self) -> Self::Offset {
        self.range.0
    }

    fn end(&self) -> Self::Offset {
        self.range.1
    }
}

impl ariadne::Span for Span {
    type SourceId = SrcId;

    fn source(&self) -> &Self::SourceId {
        &self.src
    }

    fn start(&self) -> usize {
        self.range.0
    }

    fn end(&self) -> usize {
        self.range.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ariadne::{Label, Report, ReportKind};

    #[test]
    fn ariadne_renders_span_with_line_numbers() {
        let src_id = SrcId::from_path("input");
        let source = "first\nsecond";
        let span = Span::new(src_id, 6..7);

        let report = Report::build(ReportKind::Error, src_id, span.start())
            .with_message("parse error")
            .with_label(Label::new(span).with_message("unexpected token"))
            .finish();

        let mut output = Vec::new();
        report
            .write(ariadne::sources([(src_id, source)]), &mut output)
            .expect("ariadne report should render");

        let rendered = String::from_utf8(output).expect("rendered output should be utf-8");
        let cleaned = strip_ansi(&rendered);
        assert!(
            cleaned.contains("input:2:1"),
            "rendered output:\n{}",
            cleaned
        );
        assert!(
            cleaned.contains("second"),
            "rendered output:\n{}",
            cleaned
        );
    }

    fn strip_ansi(input: &str) -> String {
        let mut output = String::new();
        let mut chars = input.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if let Some('[') = chars.peek().copied() {
                    let _ = chars.next();
                    while let Some(next) = chars.next() {
                        if next == 'm' {
                            break;
                        }
                    }
                    continue;
                }
            }
            output.push(ch);
        }
        output
    }
}
