use std::borrow::Borrow;
use std::fs::DirEntry;

// use unicode_width::UnicodeWidthStr;

use tui::buffer::Buffer;
use tui::layout::Rect;
use tui::style::Style;
use tui::widgets::{Block, List, Text, Widget};

pub struct SelectableList<'b, L> {
    block: Block<'b>,
    items: L,
    selected: Option<usize>,
    highlight_symbol: &'b str,
    style: Style,
    highlight_style: Style,
}

impl<'b, B, L> Default for SelectableList<'b, L>
where
    L: Iterator<Item = B> + Default,
    B: Borrow<str>,
{
    fn default() -> Self {
        SelectableList::new(L::default())
    }
}

impl<'b, B, L> SelectableList<'b, L>
where
    L: Iterator<Item = B>,
    B: Borrow<str>,
{
    pub fn new<I>(items: I) -> Self
    where
        I: IntoIterator<Item = B, IntoIter = L>,
    {
        SelectableList {
            block: Block::default(),
            items: items.into_iter(),
            selected: None,
            highlight_symbol: "",
            style: Style::default(),
            highlight_style: Style::default(),
        }
    }

    pub fn block(mut self, block: Block<'b>) -> Self {
        self.block = block;
        self
    }

    pub fn items<I>(mut self, items: I) -> Self
    where
        I: IntoIterator<Item = B, IntoIter = L>,
    {
        self.items = items.into_iter();
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }

    pub fn highlight_symbol(mut self, highlight_symbol: &'b str) -> Self {
        self.highlight_symbol = highlight_symbol;
        self
    }

    pub fn highlight_style(mut self, highlight_style: Style) -> Self {
        self.highlight_style = highlight_style;
        self
    }

    pub fn select(mut self, index: Option<usize>) -> Self {
        self.selected = index;
        self
    }
}

impl<'b, B, L> Widget for SelectableList<'b, L>
where
    L: Iterator<Item = B>,
    B: Borrow<str>,
{
    fn draw(&mut self, area: Rect, buf: &mut Buffer) {
        let list_area = self.block.inner(area);
        let list_height = list_area.height as usize;

        let highlight_symbol = self.highlight_symbol;
        // TODO: unicode_width
        //let blank_symbol = " ".repeat(highlight_symbol.width());
        let blank_symbol = " ".repeat(highlight_symbol.len());

        // Make sure that the list shows the selected item
        // TODO: Make the bottom offset configurable.
        let selected = self.selected;
        let offset = if let Some(selected) = selected {
            if selected >= list_height {
                selected - list_height + 1
            } else {
                0
            }
        } else {
            0
        };

        let items = self.items.by_ref().enumerate().skip(offset).map({
            let highlight_style = self.highlight_style;
            let style = self.style;
            move |(i, item)| {
                if Some(i) == selected {
                    Text::styled(
                        format!("{} {}", highlight_symbol, item.borrow()),
                        highlight_style,
                    )
                } else {
                    Text::styled(format!("{} {}", blank_symbol, item.borrow()), style)
                }
            }
        });

        List::new(items)
            .block(self.block)
            .style(self.style)
            .draw(area, buf);
    }
}

pub struct FileSelector<'b, B> {
    block: Block<'b>,
    items: Vec<B>,
    style: Style,
    selected: Option<usize>,
    scroll: usize,
}

impl<'b, B> Default for FileSelector<'b, B>
where
    B: Borrow<DirEntry>,
{
    fn default() -> Self {
        FileSelector {
            block: Block::default(),
            items: Vec::new(),
            style: Default::default(),
            selected: None,
            scroll: 0,
        }
    }
}

impl<'b, B> FileSelector<'b, B>
where
    B: Borrow<DirEntry>,
{
    pub fn with_capacity(capacity: usize) -> Self {
        FileSelector {
            block: Block::default(),
            items: Vec::with_capacity(capacity),
            style: Default::default(),
            selected: None,
            scroll: 0,
        }
    }

    pub fn block(mut self, block: Block<'b>) -> Self {
        self.block = block;
        self
    }

    pub fn style(mut self, style: Style) -> Self {
        self.style = style;
        self
    }
}

impl<'b, B> Widget for FileSelector<'b, B>
where
    B: Borrow<DirEntry>,
{
    fn draw(&mut self, area: Rect, buf: &mut Buffer) {
        SelectableList::new(self.items.iter().skip(self.scroll).map(|item| {
            item.borrow()
                .file_name()
                .into_string()
                .unwrap_or_else(|badstr| badstr.to_string_lossy().into_owned())
        }))
        .select(self.selected.map(|selected| selected - self.scroll))
        .block(self.block)
        .style(self.style)
        .draw(area, buf);
    }
}
