type CrosstermResult<T> = std::io::Result<T>;

pub mod style {
    #[derive(Clone, Copy)]
    #[allow(dead_code)]
    pub enum Color {
        DarkGray,
        Rgb(u8, u8, u8),
    }

    #[derive(Clone, Copy)]
    pub struct Style {
        pub(crate) bg: Option<Color>,
    }

    impl Style {
        pub fn default() -> Self {
            Self { bg: None }
        }

        pub fn bg(mut self, color: Color) -> Self {
            self.bg = Some(color);
            self
        }
    }
}

pub mod backend {
    use std::io::{Stdout, Write};

    use crossterm::{
        cursor,
        execute,
        terminal::{self, Clear, ClearType},
    };

    pub trait Backend {
        fn size(&self) -> super::CrosstermResult<(u16, u16)>;
        fn clear(&mut self) -> super::CrosstermResult<()>;
        fn write_at(&mut self, x: u16, y: u16, content: &str) -> super::CrosstermResult<()>;
        fn flush(&mut self) -> super::CrosstermResult<()>;
        fn set_cursor(&mut self, x: u16, y: u16) -> super::CrosstermResult<()>;
        fn show_cursor(&mut self) -> super::CrosstermResult<()>;
    }

    pub struct CrosstermBackend<W: Write> {
        writer: W,
    }

    impl CrosstermBackend<Stdout> {
        pub fn new(writer: Stdout) -> Self {
            Self { writer }
        }
    }

    impl<W: Write> std::io::Write for CrosstermBackend<W> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.writer.write(buf)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.writer.flush()
        }
    }

    impl<W: Write> Backend for CrosstermBackend<W> {
        fn size(&self) -> super::CrosstermResult<(u16, u16)> {
            terminal::size()
        }

        fn clear(&mut self) -> super::CrosstermResult<()> {
            execute!(
                self.writer,
                Clear(ClearType::All),
                cursor::MoveTo(0, 0)
            )
        }

        fn write_at(&mut self, x: u16, y: u16, content: &str) -> super::CrosstermResult<()> {
            execute!(
                self.writer,
                cursor::MoveTo(x, y),
                crossterm::style::Print(content)
            )
        }

        fn flush(&mut self) -> super::CrosstermResult<()> {
            self.writer.flush()
        }

        fn set_cursor(&mut self, x: u16, y: u16) -> super::CrosstermResult<()> {
            execute!(self.writer, cursor::MoveTo(x, y))
        }

        fn show_cursor(&mut self) -> super::CrosstermResult<()> {
            execute!(self.writer, cursor::Show)
        }
    }
}

pub mod layout {
    #[derive(Clone, Copy, Debug)]
    pub enum Constraint {
        Length(u16),
        Min(u16),
    }

    #[derive(Clone, Copy, Debug)]
    pub enum Direction {
        Horizontal,
        Vertical,
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Rect {
        pub x: u16,
        pub y: u16,
        pub width: u16,
        pub height: u16,
    }

    impl Rect {
        pub fn new(x: u16, y: u16, width: u16, height: u16) -> Self {
            Self {
                x,
                y,
                width,
                height,
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct Layout {
        direction: Direction,
        constraints: Vec<Constraint>,
    }

    impl Default for Layout {
        fn default() -> Self {
            Self {
                direction: Direction::Horizontal,
                constraints: Vec::new(),
            }
        }
    }

    impl Layout {
        pub fn direction(mut self, direction: Direction) -> Self {
            self.direction = direction;
            self
        }

        pub fn constraints(mut self, constraints: &[Constraint]) -> Self {
            self.constraints = constraints.to_vec();
            self
        }

        pub fn split(&self, area: Rect) -> Vec<Rect> {
            if self.constraints.is_empty() {
                return vec![area];
            }

            match self.direction {
                Direction::Vertical => self.split_vertical(area),
                Direction::Horizontal => self.split_horizontal(area),
            }
        }

        fn split_vertical(&self, area: Rect) -> Vec<Rect> {
            let available = area.height;
            let mut rects = Vec::new();
            let mut fixed_sum: u16 = 0;
            let mut min_count = 0;
            for constraint in &self.constraints {
                match constraint {
                    Constraint::Length(len) => fixed_sum = fixed_sum.saturating_add(*len),
                    Constraint::Min(min) => {
                        fixed_sum = fixed_sum.saturating_add(*min);
                        min_count += 1;
                    }
                }
            }
            let mut y = area.y;
            let mut mins_left = min_count as u16;
            let mut extra = available.saturating_sub(fixed_sum);
            let mut remaining_height = available;
            for (idx, constraint) in self.constraints.iter().enumerate() {
                if remaining_height == 0 {
                    rects.push(Rect {
                        x: area.x,
                        y,
                        width: area.width,
                        height: 0,
                    });
                    continue;
                }
                let mut height = match constraint {
                    Constraint::Length(len) => (*len).min(remaining_height),
                    Constraint::Min(min) => {
                        if mins_left <= 1 {
                            let value = (*min).saturating_add(extra).min(remaining_height);
                            extra = 0;
                            if mins_left > 0 {
                                mins_left -= 1;
                            }
                            value
                        } else {
                            mins_left -= 1;
                            (*min).min(remaining_height)
                        }
                    }
                };
                if min_count == 0 && idx == self.constraints.len() - 1 {
                    let expanded = height.saturating_add(extra);
                    height = expanded.min(remaining_height);
                    extra = 0;
                }
                if idx == self.constraints.len() - 1 {
                    height = remaining_height;
                }
                let rect = Rect {
                    x: area.x,
                    y,
                    width: area.width,
                    height,
                };
                y = y.saturating_add(height);
                remaining_height = remaining_height.saturating_sub(height);
                rects.push(rect);
            }
            rects
        }

        fn split_horizontal(&self, area: Rect) -> Vec<Rect> {
            let available = area.width;
            let mut rects = Vec::new();
            let mut fixed_sum: u16 = 0;
            let mut min_count = 0;
            for constraint in &self.constraints {
                match constraint {
                    Constraint::Length(len) => fixed_sum = fixed_sum.saturating_add(*len),
                    Constraint::Min(min) => {
                        fixed_sum = fixed_sum.saturating_add(*min);
                        min_count += 1;
                    }
                }
            }
            let mut x = area.x;
            let mut mins_left = min_count as u16;
            let mut extra = available.saturating_sub(fixed_sum);
            let mut remaining_width = available;
            for (idx, constraint) in self.constraints.iter().enumerate() {
                if remaining_width == 0 {
                    rects.push(Rect {
                        x,
                        y: area.y,
                        width: 0,
                        height: area.height,
                    });
                    continue;
                }
                let mut width = match constraint {
                    Constraint::Length(len) => (*len).min(remaining_width),
                    Constraint::Min(min) => {
                        if mins_left <= 1 {
                            let value = (*min).saturating_add(extra).min(remaining_width);
                            extra = 0;
                            if mins_left > 0 {
                                mins_left -= 1;
                            }
                            value
                        } else {
                            mins_left -= 1;
                            (*min).min(remaining_width)
                        }
                    }
                };
                if min_count == 0 && idx == self.constraints.len() - 1 {
                    let expanded = width.saturating_add(extra);
                    width = expanded.min(remaining_width);
                    extra = 0;
                }
                if idx == self.constraints.len() - 1 {
                    width = remaining_width;
                }
                let rect = Rect {
                    x,
                    y: area.y,
                    width,
                    height: area.height,
                };
                x = x.saturating_add(width);
                remaining_width = remaining_width.saturating_sub(width);
                rects.push(rect);
            }
            rects
        }
    }
}

pub mod text {
    #[derive(Clone, Debug)]
    pub struct Line {
        pub content: String,
    }

    impl Line {
        pub fn from<S: Into<String>>(s: S) -> Self {
            Self { content: s.into() }
        }
    }
}

pub mod widgets {
    use super::{backend::Backend, layout::Rect, style, Frame, Widget};
    use crate::ratatui::text::Line;

    #[derive(Clone, Copy, Default)]
    pub struct Borders {
        pub top: bool,
        pub bottom: bool,
        pub left: bool,
        pub right: bool,
    }

    impl Borders {
        pub const NONE: Borders = Borders {
            top: false,
            bottom: false,
            left: false,
            right: false,
        };
        pub const ALL: Borders = Borders {
            top: true,
            bottom: true,
            left: true,
            right: true,
        };
        pub fn any(&self) -> bool {
            self.top || self.bottom || self.left || self.right
        }
    }

    #[derive(Clone, Copy)]
    pub struct Wrap {
        pub trim: bool,
    }

    #[derive(Clone, Default)]
    pub struct Block {
        title: Option<String>,
        borders: Borders,
    }

    impl Block {
        pub fn default() -> Self {
            Self {
                title: None,
                borders: Borders::NONE,
            }
        }

        pub fn title(mut self, title: impl Into<String>) -> Self {
            self.title = Some(title.into());
            self
        }

        pub fn borders(mut self, borders: Borders) -> Self {
            self.borders = borders;
            self
        }

        fn render<B: Backend>(&self, area: Rect, frame: &mut Frame<'_, B>) -> Rect {
            if !self.borders.any() {
                return area;
            }
            if area.width < 2 || area.height < 2 {
                return area;
            }
            let horizontal = "-".repeat(area.width.saturating_sub(2) as usize);
            let top = format!("+{}+", horizontal);
            let bottom = top.clone();
            let _ = frame.backend_mut().write_at(area.x, area.y, &top);
            let _ = frame
                .backend_mut()
                .write_at(area.x, area.y + area.height.saturating_sub(1), &bottom);
            for row in 1..area.height.saturating_sub(1) {
                let mut line = String::from("|");
                line.push_str(&" ".repeat(area.width.saturating_sub(2) as usize));
                line.push('|');
                let _ = frame.backend_mut().write_at(area.x, area.y + row, &line);
            }
            if let Some(title) = &self.title {
                let mut title_str = String::from(" ");
                title_str.push_str(title);
                let max_width = area.width.saturating_sub(2) as usize;
                if max_width > 0 {
                    let truncated = if title_str.len() > max_width {
                        &title_str[..max_width]
                    } else {
                        &title_str
                    };
                    let _ = frame
                        .backend_mut()
                        .write_at(area.x + 1, area.y, truncated);
                }
            }
            Rect {
                x: area.x + 1,
                y: area.y + 1,
                width: area.width.saturating_sub(2),
                height: area.height.saturating_sub(2),
            }
        }
    }

    pub struct Paragraph {
        lines: Vec<Line>,
        block: Option<Block>,
        trim: bool,
        style: style::Style,
    }

    impl Paragraph {
        pub fn new(lines: Vec<Line>) -> Self {
            Self {
                lines,
                block: None,
                trim: false,
                style: style::Style::default(),
            }
        }

        pub fn block(mut self, block: Block) -> Self {
            self.block = Some(block);
            self
        }

        pub fn wrap(mut self, wrap: Wrap) -> Self {
            self.trim = wrap.trim;
            self
        }

        pub fn style(mut self, style: style::Style) -> Self {
            self.style = style;
            self
        }
    }

    impl Widget for Paragraph {
        fn render<B: Backend>(&self, area: Rect, frame: &mut Frame<'_, B>) {
            let mut inner = area;
            if let Some(block) = &self.block {
                inner = block.render(area, frame);
            }
            if inner.width == 0 || inner.height == 0 {
                return;
            }
            let blank = apply_style(&" ".repeat(inner.width as usize), &self.style);
            for row in 0..inner.height {
                let _ = frame
                    .backend_mut()
                    .write_at(inner.x, inner.y + row, &blank);
            }
            let mut y = inner.y;
            for line in &self.lines {
                if y >= inner.y + inner.height {
                    break;
                }
                let source = if self.trim {
                    line.content.trim().to_string()
                } else {
                    line.content.clone()
                };
                if source.is_empty() {
                    y += 1;
                    continue;
                }
                let mut remaining = source.as_str();
                while !remaining.is_empty() && y < inner.y + inner.height {
                    let max_width = inner.width as usize;
                    let take = remaining.len().min(max_width);
                    let chunk = &remaining[..take];
                    let padded = if take < max_width {
                        let mut owned = chunk.to_string();
                        owned.push_str(&" ".repeat(max_width - take));
                        owned
                    } else {
                        chunk.to_string()
                    };
                    let styled = apply_style(&padded, &self.style);
                    let _ = frame.backend_mut().write_at(inner.x, y, &styled);
                    remaining = &remaining[take..];
                    y += 1;
                }
            }
        }
    }

    fn apply_style(content: &str, style: &style::Style) -> String {
        match style.bg {
            Some(style::Color::DarkGray) => {
                format!("\x1b[48;2;40;40;40m{}\x1b[0m", content)
            }
            Some(style::Color::Rgb(r, g, b)) => {
                format!("\x1b[48;2;{};{};{}m{}\x1b[0m", r, g, b, content)
            }
            None => content.to_string(),
        }
    }
}

use backend::Backend;
use layout::Rect;

pub struct Terminal<B: Backend> {
    backend: B,
}

impl<B: Backend> Terminal<B> {
    pub fn new(backend: B) -> CrosstermResult<Self> {
        Ok(Self { backend })
    }

    pub fn draw<F>(&mut self, f: F) -> CrosstermResult<()>
    where
        F: FnOnce(&mut Frame<'_, B>),
    {
        let (width, height) = self.backend.size()?;
        self.backend.clear()?;
        let mut frame = Frame::new(&mut self.backend, Rect::new(0, 0, width, height));
        f(&mut frame);
        self.backend.flush()
    }

    pub fn show_cursor(&mut self) -> CrosstermResult<()> {
        self.backend.show_cursor()
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}

pub struct Frame<'a, B: Backend> {
    backend: &'a mut B,
    size: Rect,
}

impl<'a, B: Backend> Frame<'a, B> {
    fn new(backend: &'a mut B, size: Rect) -> Self {
        Self { backend, size }
    }

    pub fn size(&self) -> Rect {
        self.size
    }

    pub fn render_widget<W>(&mut self, widget: W, area: Rect)
    where
        W: Widget,
    {
        widget.render(area, self);
    }

    pub fn backend_mut(&mut self) -> &mut B {
        self.backend
    }

    pub fn set_cursor(&mut self, x: u16, y: u16) -> CrosstermResult<()> {
        self.backend.set_cursor(x, y)
    }
}

pub trait Widget {
    fn render<B: Backend>(&self, area: Rect, frame: &mut Frame<'_, B>);
}
