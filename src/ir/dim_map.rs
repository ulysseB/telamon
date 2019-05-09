use std::fmt;

use crate::ir;
use itertools::Itertools;
use linked_list;
use linked_list::LinkedList;
use serde::de::{self, Deserialize, Deserializer, SeqAccess};
use serde::ser::{Serialize, SerializeSeq, Serializer};

use utils::*;

/// Represents a mapping between dimenions.
#[derive(Clone, Debug)]
pub struct DimMap {
    // TODO(cleanup): once merge is handled exclusively from the domain, we can use
    // a `Vec` instead.
    map: LinkedList<(ir::DimId, ir::DimId)>,
}

impl Serialize for DimMap {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.map.len()))?;
        for e in &self.map {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for DimMap {
    fn deserialize<D>(deserializer: D) -> Result<DimMap, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct DimMapVisitor;

        impl<'de> de::Visitor<'de> for DimMapVisitor {
            type Value = DimMap;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of DimId pairs")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<DimMap, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let mut map = LinkedList::new();
                while let Some(elt) = seq.next_element()? {
                    map.push_back(elt);
                }
                Ok(DimMap { map })
            }
        }

        deserializer.deserialize_seq(DimMapVisitor)
    }
}

// TODO(cleanup): Send should be derived for LinkedList.
unsafe impl Send for DimMap {}
unsafe impl Sync for DimMap {}

impl DimMap {
    /// Create a new `DimMap`.
    pub fn new<IT>(dims: IT) -> Self
    where
        IT: IntoIterator<Item = (ir::DimId, ir::DimId)>,
    {
        DimMap {
            map: dims.into_iter().collect(),
        }
    }

    /// Returns an empty `DimMap`.
    pub fn empty() -> Self {
        DimMap {
            map: LinkedList::new(),
        }
    }

    /// Renames a basic block into an other. Indicates if some mapping were
    /// removed.
    pub fn merge_dims(&mut self, lhs: ir::DimId, rhs: ir::DimId) -> bool {
        self.filter_remove(|&mut pair| pair == (lhs, rhs) || pair == (rhs, lhs))
            .count()
            > 0
    }

    /// Iterates over the DimMap.
    pub fn iter(&self) -> linked_list::Iter<(ir::DimId, ir::DimId)> {
        self.map.iter()
    }

    /// Filters the DimMap, removing the filtered elements in-place.
    pub fn filter_remove<F>(&mut self, f: F) -> FilterList<(ir::DimId, ir::DimId), F>
    where
        F: FnMut(&mut (ir::DimId, ir::DimId)) -> bool,
    {
        filter_list(&mut self.map, f)
    }

    /// Returns true if the `DimMap` is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl IntoIterator for DimMap {
    type Item = (ir::DimId, ir::DimId);
    type IntoIter = linked_list::IntoIter<(ir::DimId, ir::DimId)>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

impl<'a> IntoIterator for &'a DimMap {
    type Item = &'a (ir::DimId, ir::DimId);
    type IntoIter = linked_list::Iter<'a, (ir::DimId, ir::DimId)>;

    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl fmt::Display for DimMap {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(
            fmt,
            "[{}]",
            self.map
                .iter()
                .map(|(lhs, rhs)| format!("{} = {}", lhs, rhs))
                .format(", ")
        )
    }
}
