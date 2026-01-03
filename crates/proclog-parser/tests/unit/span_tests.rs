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
        assert!(cleaned.contains("second"), "rendered output:\n{}", cleaned);
    }

    fn strip_ansi(input: &str) -> String {
        let mut output = String::new();
        let mut chars = input.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '\x1b' {
                if let Some('[') = chars.peek().copied() {
                    let _ = chars.next();
                    for next in chars.by_ref() {
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
