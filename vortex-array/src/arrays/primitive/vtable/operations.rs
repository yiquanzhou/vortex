// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::ProbeAccess;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Primitive;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;

impl OperationsVTable<Primitive> for Primitive {
    type ProbeState = ();

    fn probe_scalar(
        array: ArrayView<'_, Primitive>,
        index: usize,
        _probe: ProbeAccess<'_, Self::ProbeState>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        value_at(array, index)
    }

    fn scalar_at(
        array: ArrayView<'_, Primitive>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        value_at(array, index)
    }
}

fn value_at(array: ArrayView<'_, Primitive>, index: usize) -> VortexResult<Scalar> {
    Ok(match_each_native_ptype!(array.ptype(), |T| {
        Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
    }))
}
