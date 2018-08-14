extern crate rpds;

use self::rpds::List;
use serde::de::{Deserialize, Deserializer, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeSeq, Serializer};
use std::fmt;
use std::marker::PhantomData;

/// A type representing a sequence of values.
///
/// Can be implemented as either a persistent list or as a native
/// vector. This is handy for serialization: we usually work with
/// persistent lists but those share part of their structure and make
/// them bad candidates for "direct" serialization. Instead, we
/// serialize the persistent list as a sequence, and always
/// deserialize it as a vector. By duplicating the objects when
/// serializing we sidestep the problem of having pointers in the
/// serialized stream.
pub enum Sequence<T> {
    List(List<T>),
    Vec(Vec<T>),
}

impl<T> Sequence<T> {
    pub fn to_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.into()
    }
}

impl<T: Clone> Into<Vec<T>> for Sequence<T> {
    fn into(self) -> Vec<T> {
        match self {
            Sequence::List(list) => {
                list.into_iter().map(::std::clone::Clone::clone).collect()
            }
            Sequence::Vec(vec) => vec,
        }
    }
}

impl<T: Serialize + Clone> Serialize for Sequence<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Sequence::List(list) => {
                let mut seq = serializer.serialize_seq(Some(list.len()))?;
                for arc in list {
                    seq.serialize_element((&arc).clone())?;
                }
                seq.end()
            }
            Sequence::Vec(vec) => serializer.collect_seq(vec.iter()),
        }
    }
}

struct SequenceVisitor<T> {
    _marker: PhantomData<T>,
}

impl<T> SequenceVisitor<T> {
    fn new() -> SequenceVisitor<T> {
        SequenceVisitor {
            _marker: PhantomData,
        }
    }
}

impl<'de, T: Deserialize<'de>> Visitor<'de> for SequenceVisitor<T> {
    type Value = Sequence<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a sequence")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut vec = if let Some(size) = seq.size_hint() {
            Vec::with_capacity(size)
        } else {
            Vec::new()
        };

        while let Some(element) = seq.next_element()? {
            vec.push(element);
        }

        Ok(Sequence::Vec(vec))
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Sequence<T> {
    fn deserialize<D>(deserializer: D) -> Result<Sequence<T>, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(SequenceVisitor::new())
    }
}
