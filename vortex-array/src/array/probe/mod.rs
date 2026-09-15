// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod array;
pub use array::*;
mod validity;
pub use validity::*;

/// Whether scalar access should retain preparation for subsequent lookups.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProbeUsage {
    /// Use temporary resources only, without initializing retained probe state.
    Once,
    /// Allow the encoding to retain preparation and decoded data between lookups.
    Repeated,
}
