pub mod asp_sat;
pub mod sat_solver;

pub use asp_sat::*;
pub use sat_solver::*;

#[cfg(test)]
#[path = "../tests/unit/asp_sat_tests.rs"]
mod asp_sat_tests;
#[cfg(test)]
#[path = "../tests/unit/splr_api_test.rs"]
mod splr_api_test;
