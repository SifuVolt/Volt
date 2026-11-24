/// Rust representation of VirxERLU's routine stack.
///
/// Python Volt appends routines, executes the last entry, but uses the first
/// entry as the root routine when deciding whether a committed action may be
/// replaced. Keeping both accessors is important for nested routines such as
/// speedflip_kickoff -> speed_flip.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutineStack<R> {
    entries: Vec<R>,
}

impl<R> Default for RoutineStack<R> {
    fn default() -> Self {
        Self { entries: Vec::new() }
    }
}

impl<R> RoutineStack<R> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Equivalent to Python agent.push(): append a child routine.
    pub fn push(&mut self, routine: R) {
        self.entries.push(routine);
    }

    /// Equivalent to Python agent.pop(): remove the currently executing
    /// routine (the last entry), revealing its parent if one exists.
    pub fn pop(&mut self) -> Option<R> {
        self.entries.pop()
    }

    /// Equivalent to Python agent.clear(): abandon the entire routine chain.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Python run_logic commonly inspects stack[0] to protect the root action.
    pub fn root(&self) -> Option<&R> {
        self.entries.first()
    }

    pub fn root_mut(&mut self) -> Option<&mut R> {
        self.entries.first_mut()
    }

    /// Python get_output executes stack[-1], the active child routine.
    pub fn active(&self) -> Option<&R> {
        self.entries.last()
    }

    pub fn active_mut(&mut self) -> Option<&mut R> {
        self.entries.last_mut()
    }

    pub fn is_clear(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &R> {
        self.entries.iter()
    }
}

/// Explicit state transitions returned by a routine tick. The runtime applies
/// these after the routine borrow ends, avoiding aliasing mutable stack state.
pub enum RoutineTransition<R> {
    Keep,
    Pop,
    Push(R),
    Replace(R),
    Clear,
    ClearAndPush(R),
}

impl<R> RoutineStack<R> {
    pub fn apply(&mut self, transition: RoutineTransition<R>) {
        match transition {
            RoutineTransition::Keep => {}
            RoutineTransition::Pop => {
                self.pop();
            }
            RoutineTransition::Push(routine) => self.push(routine),
            RoutineTransition::Replace(routine) => {
                self.pop();
                self.push(routine);
            }
            RoutineTransition::Clear => self.clear(),
            RoutineTransition::ClearAndPush(routine) => {
                self.clear();
                self.push(routine);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_routines_preserve_root_and_execute_last_entry() {
        let mut stack = RoutineStack::new();
        stack.push("speedflip_kickoff");
        stack.push("speed_flip");

        assert_eq!(stack.root(), Some(&"speedflip_kickoff"));
        assert_eq!(stack.active(), Some(&"speed_flip"));
        assert_eq!(stack.len(), 2);

        assert_eq!(stack.pop(), Some("speed_flip"));
        assert_eq!(stack.root(), Some(&"speedflip_kickoff"));
        assert_eq!(stack.active(), Some(&"speedflip_kickoff"));
    }

    #[test]
    fn clear_removes_the_entire_routine_chain() {
        let mut stack = RoutineStack::new();
        stack.push("jump_shot");
        stack.push("recovery");
        stack.clear();

        assert!(stack.is_clear());
        assert_eq!(stack.root(), None);
        assert_eq!(stack.active(), None);
    }

    #[test]
    fn transitions_match_python_push_pop_and_replace_semantics() {
        let mut stack = RoutineStack::new();
        stack.push("ground_shot");
        stack.apply(RoutineTransition::Push("flip"));
        assert_eq!(stack.active(), Some(&"flip"));
        stack.apply(RoutineTransition::Pop);
        assert_eq!(stack.active(), Some(&"ground_shot"));
        stack.apply(RoutineTransition::Replace("jump_shot"));
        assert_eq!(stack.root(), Some(&"jump_shot"));
        assert_eq!(stack.active(), Some(&"jump_shot"));
    }
}
