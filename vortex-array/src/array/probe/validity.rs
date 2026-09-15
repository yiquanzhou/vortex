// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::probe::ArrayProbe;

/// A validity accessor built by [`Validity::probe`](crate::validity::Validity::probe).
///
/// Uniform validity retains nothing; array-backed validity delegates to an [`ArrayProbe`]
/// over the boolean array with the requested retention policy.
pub enum ProbeValidity {
    /// Validity is uniform, so no lookup is needed.
    Constant(bool),
    /// Validity backed by a boolean array.
    Array(ArrayProbe<'static>),
}

impl ProbeValidity {
    /// Returns whether the row at `index` is valid.
    pub fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        match self {
            Self::Constant(valid) => Ok(*valid),
            Self::Array(probe) => probe
                .execute_scalar(index, ctx)?
                .as_bool()
                .value()
                .ok_or_else(|| vortex_err!("validity value at index {index} is null")),
        }
    }

    /// Returns whether the row at `index` is null.
    pub fn execute_is_invalid(
        &mut self,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<bool> {
        Ok(!self.execute_is_valid(index, ctx)?)
    }
}
