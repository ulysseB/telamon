pub fn inverse(self) -> Self {{
    let high_bits = (self.bits & ({low_bits})) << 1;
    let low_bits = (self.bits & ({high_bits})) >> 1;
    let same_bits = self.bits & ({same_bits});
    {type_name} {{ bits: low_bits | high_bits | same_bits}}
}}
