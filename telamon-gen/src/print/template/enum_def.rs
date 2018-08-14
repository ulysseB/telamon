{doc_comment}
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct {type_name} {{ bits: {bits_type} }}

impl {type_name} {{
    {value_defs}{alias_defs}

    pub const ALL: {type_name} = {type_name} {{ bits: {all_bits} }};

    /// Returns the empty domain.
    pub const FAILED: {type_name} = {type_name} {{ bits: 0 }};

    /// Returns the full domain.
    pub fn all() -> Self {{ Self::ALL }}

    /// Insert values in the domain.
    pub fn insert(&mut self, alternatives: Self) {{
        self.bits |= alternatives.bits
    }}

    /// Lists the alternatives contained in the domain.
    pub fn list<'a>(&self) -> impl Iterator<Item=Self> + 'static {{
        let bits = self.bits;
        (0..{num_values}).map(|x| 1 << x)
            .filter(move |x| (bits & x) != 0)
            .map(|x| {type_name} {{ bits: x }})
    }}

    /// Indicates if two choices will have the same value.
    pub fn eq(&self, other: Self) -> bool {{
        self.is_constrained() && *self == other
    }}

    /// Indicates if two choices cannot be equal.
    pub fn neq(&self, other: Self) -> bool {{
        !self.intersects(other)
    }}

    {inverse}
}}

impl Domain for {type_name} {{
    fn is_failed(&self) -> bool {{ self.bits == 0 }}

    fn is_constrained(&self) -> bool {{ self.bits.count_ones() <= 1 }}

    fn contains(&self, other: Self) -> bool {{
        (self.bits & other.bits) == other.bits
    }}

    fn intersects(&self, other: Self) -> bool {{
        (self.bits & other.bits) != 0
    }}

    fn restrict(&mut self, alternatives: Self) {{
        self.bits &= alternatives.bits
    }}
}}

impl std::ops::BitAnd for {type_name} {{
    type Output = {type_name};

    fn bitand(self, other: Self) -> Self {{
        {type_name} {{ bits: self.bits & other.bits }}
    }}
}}

impl std::ops::BitOr for {type_name} {{
    type Output = {type_name};

    fn bitor(self, other: Self) -> Self {{
        {type_name} {{ bits: self.bits | other.bits }}
    }}
}}

impl std::ops::BitXor for {type_name} {{
    type Output = {type_name};

    fn bitxor(self, other: Self) -> Self {{
        {type_name} {{ bits: self.bits ^ other.bits }}
    }}
}}

impl std::ops::Not for {type_name} {{
    type Output = {type_name};

    fn not(self) -> Self {{
        {type_name} {{ bits: self.bits ^ {all_bits} }}
    }}
}}

impl std::ops::BitOrAssign for {type_name} {{
    fn bitor_assign(&mut self, other: Self) {{
        self.bits |= other.bits;
    }}
}}

impl std::ops::BitAndAssign for {type_name} {{
    fn bitand_assign(&mut self, other: Self) {{
        self.bits &= other.bits;
    }}
}}

impl std::fmt::Debug for {type_name} {{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {{
        let mut values = vec![];
        {printers}
        if values.is_empty() {{
            write!(f, "--")
        }} else {{
            write!(f, "{{}}", values[0])?;
            for value in &values[1..] {{ write!(f, " | {{}}", value)?; }}
            Ok(())
        }}
    }}
}}
