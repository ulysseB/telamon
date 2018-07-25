//! Table with named columns.
#![allow(dead_code)]
use prettytable;
use std;

pub struct Table<T: std::fmt::Display> {
    header: Vec<String>,
    data: Vec<Vec<T>>,
}

impl<T: std::fmt::Display> Table<T> {
    /// Creates an empty table with the given headers.
    pub fn new(header: Vec<String>) -> Self {
        Table { header, data: vec![] }
    }

    /// Inserts an entry into the table.
    pub fn add_entry(&mut self, entry: Vec<T>) {
        assert_eq!(entry.len(), self.header.len());
        self.data.push(entry);
    }

    /// Prepare the table for pretty printing.
    pub fn pretty(&self) -> prettytable::Table {
        let mut table = prettytable::Table::new();
        table.add_row(self.header.iter().collect());
        for row in &self.data {
            table.add_row(row.iter().collect());
        }
        table
    }

    /// Returns a column of the table.
    pub fn column(&self, id: usize) -> impl Iterator<Item=&T> {
        self.data.iter().map(move |x| &x[id])
    }

    /// Returns the rows of the table.
    pub fn rows(&self) -> std::slice::Iter<Vec<T>> {
        self.data.iter()
    }
}

impl<T: std::fmt::Display> IntoIterator for Table<T> {
    type Item = Vec<T>;
    type IntoIter = std::vec::IntoIter<Vec<T>>;

    fn into_iter(self) -> Self::IntoIter { self.data.into_iter() }
}

impl<'a, T: std::fmt::Display> IntoIterator for &'a Table<T> {
    type Item = &'a Vec<T>;
    type IntoIter = std::slice::Iter<'a, Vec<T>>;

    fn into_iter(self) -> Self::IntoIter { self.data.iter() }
}
