use super::VarDef;
use super::Condition;

#[derive(Debug)]
pub struct TriggerDef {
    pub foralls: Vec<VarDef>,
    pub conditions: Vec<Condition>,
    pub code: String,
}
