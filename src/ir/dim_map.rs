use ir::dim::Id;
use linked_list;
use linked_list::LinkedList;
use utils::*;

/// Represents a mapping between dimenions.
#[derive(Clone, Debug)]
// TODO(cleanup): once merge is handled exclusively from the domain, we can use a `Vec`
// instead.
pub struct DimMap {
    map: LinkedList<(Id, Id)>,
}

// TODO(cleanup): Send should be derived for LinkedList.
unsafe impl Send for DimMap {}
unsafe impl Sync for DimMap {}

impl DimMap {
    /// Create a new `DimMap`.
    pub fn new<IT>(dims: IT) -> Self
            where IT: IntoIterator<Item=(Id, Id)> {
        DimMap { map: dims.into_iter().collect() }
    }

    /// Returns an empty `DimMap`.
    pub fn empty() -> DimMap { DimMap { map: LinkedList::new() } }

    /// Renames a basic block into an other. Indicates if some mapping were removed.
    pub fn merge_dims(&mut self, lhs: Id, rhs: Id) -> bool {
        self.filter(|&mut pair| pair == (lhs, rhs) || pair == (rhs, lhs)).count() > 0
    }

    /// Iterates over the DimMap.
    pub fn iter(&self) -> linked_list::Iter<(Id, Id)> { self.map.iter() }

    /// Filters the DimMap.
    pub fn filter<F>(&mut self, f: F) -> FilterList<(Id, Id), F>
            where F: FnMut(&mut (Id, Id)) -> bool {
        filter_list(&mut self.map, f)
    }

    /// Returns true if the `DimMap` is empty.
    pub fn is_empty(&self) -> bool { self.map.is_empty() }
}

impl IntoIterator for DimMap {
    type Item = (Id, Id);
    type IntoIter = linked_list::IntoIter<(Id, Id)>;

    fn into_iter(self) -> Self::IntoIter { self.map.into_iter() }
}


impl<'a> IntoIterator for &'a DimMap {
    type Item = &'a (Id, Id);
    type IntoIter = linked_list::Iter<'a, (Id, Id)>;

    fn into_iter(self) -> Self::IntoIter { self.map.iter() }
}
