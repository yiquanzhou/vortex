// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;

use smallvec::SmallVec;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::probe::ProbeUsage;
use crate::array::probe::ProbeValidity;
use crate::scalar::Scalar;

/// A scalar accessor over a borrowed or owned array that optionally retains preparation
/// between lookups.
///
/// Array handles share their buffers. [`ProbeUsage::Repeated`] initializes encoding state and
/// child probes lazily and reuses them for following lookups; [`ProbeUsage::Once`] retains
/// nothing. An owned probe can outlive the original array handle. Probes are local to a thread.
pub struct ArrayProbe<'a> {
    array: Cow<'a, ArrayRef>,
    retained: Option<Retained>,
}

/// Everything a repeated-access probe keeps between lookups.
#[derive(Default)]
pub(crate) struct Retained {
    state: ProbeStorage,
    // Boxed because `ProbeValidity` itself contains an `ArrayProbe`.
    validity: Option<Box<ProbeValidity>>,
}

impl ArrayProbe<'static> {
    /// Own an array and choose whether to retain preparation.
    pub fn new(array: ArrayRef, usage: ProbeUsage) -> Self {
        Self {
            array: Cow::Owned(array),
            retained: Retained::for_usage(usage),
        }
    }
}

impl<'a> ArrayProbe<'a> {
    /// Borrow an array and choose whether to retain preparation.
    pub(crate) fn borrowed(array: &'a ArrayRef, usage: ProbeUsage) -> Self {
        Self {
            array: Cow::Borrowed(array),
            retained: Retained::for_usage(usage),
        }
    }

    /// The array this probe reads from.
    pub fn array(&self) -> &ArrayRef {
        &self.array
    }

    /// Read a scalar, including its nullness, preparing and reusing state as appropriate.
    pub fn execute_scalar(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<Scalar> {
        if !self.execute_is_valid(index, ctx)? {
            return Ok(Scalar::null(self.array.dtype().clone()));
        }
        let array: &ArrayRef = &self.array;
        let storage = self.retained.as_mut().map(|retained| &mut retained.state);
        let scalar = array.dyn_array().probe_scalar(array, index, storage, ctx)?;
        debug_assert_eq!(scalar.dtype(), array.dtype(), "Scalar dtype mismatch");
        Ok(scalar)
    }

    /// Check bounds and read validity using this probe's retention policy.
    pub fn execute_is_valid(&mut self, index: usize, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
        let array: &ArrayRef = &self.array;
        vortex_ensure!(index < array.len(), OutOfBounds: index, 0, array.len());
        if !array.dtype().is_nullable() {
            return Ok(true);
        }
        let Some(retained) = &mut self.retained else {
            return array
                .validity()?
                .probe(ProbeUsage::Once)
                .execute_is_valid(index, ctx);
        };
        if retained.validity.is_none() {
            let validity = array.validity()?.probe(ProbeUsage::Repeated);
            retained.validity = Some(Box::new(validity));
        }
        retained
            .validity
            .as_mut()
            .ok_or_else(|| vortex_err!("validity probe was just initialized"))?
            .execute_is_valid(index, ctx)
    }
}

impl Retained {
    fn for_usage(usage: ProbeUsage) -> Option<Self> {
        match usage {
            ProbeUsage::Once => None,
            ProbeUsage::Repeated => Some(Self::default()),
        }
    }
}

/// Encoding state and lazy child probes for one source array.
///
/// Passed to [`OperationsVTable::probe_scalar`](crate::vtable::OperationsVTable::probe_scalar).
/// For repeated access the framework boxes this once per probe and reuses it; for one-off
/// access it lives on the stack for the duration of the call, so `S::default()` must be cheap.
pub struct ProbeState<S> {
    /// The encoding's own retained state.
    pub state: S,
    /// Lazily created probes over the source's child slots.
    pub children: ProbeChildren,
}

impl<S: Default> ProbeState<S> {
    /// Create empty state with the given retention policy.
    ///
    /// The framework builds this for every probe; encodings only need it to run their
    /// `probe_scalar` path from `scalar_at` with [`ProbeUsage::Once`].
    pub fn new(usage: ProbeUsage) -> Self {
        Self {
            state: S::default(),
            children: ProbeChildren {
                usage,
                slots: SmallVec::new(),
            },
        }
    }
}

impl<S> ProbeState<S> {
    /// Whether this state is discarded after one lookup or reused across many.
    ///
    /// Encodings whose preparation is expensive should skip it for [`ProbeUsage::Once`].
    pub fn usage(&self) -> ProbeUsage {
        self.children.usage
    }
}

/// Lazy child probes over the slots of one source array.
///
/// Child probes inherit the parent's retention policy. The slot table is created on the first
/// valid request; slots that are never requested stay empty. Dropping the parent state drops
/// every child probe.
pub struct ProbeChildren {
    usage: ProbeUsage,
    slots: SmallVec<[Option<ArrayProbe<'static>>; 4]>,
}

impl ProbeChildren {
    /// Get or create a probe over `parent`'s given child slot.
    ///
    /// `parent` must be the array this state was created for. Returns `None` for an absent
    /// slot and an error for an out-of-bounds slot; neither case creates the slot table.
    pub fn slot(
        &mut self,
        parent: &ArrayRef,
        slot: usize,
    ) -> VortexResult<Option<&mut ArrayProbe<'static>>> {
        let slots = parent.slots();
        let child = slots
            .get(slot)
            .ok_or_else(|| vortex_err!("Probe slot {slot} is out of bounds"))?;
        let Some(child) = child else {
            return Ok(None);
        };
        if self.slots.is_empty() {
            self.slots.resize_with(slots.len(), || None);
        }
        let usage = self.usage;
        Ok(Some(self.slots[slot].get_or_insert_with(|| {
            ArrayProbe::new(child.clone(), usage)
        })))
    }
}

/// Type-erased, lazily initialized [`ProbeState`] for a retained probe.
#[derive(Default)]
pub(crate) struct ProbeStorage(Option<Box<dyn Any>>);

impl ProbeStorage {
    pub(crate) fn get_or_init<S: Default + 'static>(&mut self) -> VortexResult<&mut ProbeState<S>> {
        self.0
            .get_or_insert_with(|| Box::new(ProbeState::<S>::new(ProbeUsage::Repeated)))
            .downcast_mut::<ProbeState<S>>()
            .ok_or_else(|| vortex_err!("Probe state type mismatch"))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array::IntoArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;

    fn nullable_ints() -> ArrayRef {
        PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array()
    }

    #[rstest]
    fn checks_bounds_and_nulls(
        #[values(ProbeUsage::Once, ProbeUsage::Repeated)] usage: ProbeUsage,
    ) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let mut probe = nullable_ints().probe(usage);
        assert!(probe.execute_scalar(3, &mut ctx).is_err());
        assert!(probe.execute_is_valid(3, &mut ctx).is_err());
        assert!(probe.execute_scalar(1, &mut ctx)?.is_null());
        assert!(!probe.execute_is_valid(1, &mut ctx)?);
        assert_eq!(
            probe.execute_scalar(2, &mut ctx)?,
            Scalar::from(Some(30i32))
        );
        assert_eq!(
            probe.execute_scalar(0, &mut ctx)?,
            Scalar::from(Some(10i32))
        );
        Ok(())
    }

    #[test]
    fn once_retains_nothing() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let mut probe = nullable_ints().probe(ProbeUsage::Once);
        probe.execute_scalar(0, &mut ctx)?;
        assert!(probe.retained.is_none());
        Ok(())
    }

    #[test]
    fn repeated_initializes_lazily_and_outlives_handle() -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = nullable_ints();
        let mut probe = array.probe(ProbeUsage::Repeated);
        drop(array);

        let retained = probe
            .retained
            .as_ref()
            .ok_or_else(|| vortex_err!("missing retained state"))?;
        assert!(retained.state.0.is_none());
        assert!(retained.validity.is_none());

        assert!(probe.execute_scalar(3, &mut ctx).is_err());
        assert!(probe.execute_scalar(1, &mut ctx)?.is_null());
        assert_eq!(
            probe.execute_scalar(0, &mut ctx)?,
            Scalar::from(Some(10i32))
        );

        let retained = probe
            .retained
            .as_ref()
            .ok_or_else(|| vortex_err!("missing retained state"))?;
        assert!(retained.state.0.is_some());
        assert!(retained.validity.is_some());
        Ok(())
    }

    #[rstest]
    fn children_are_created_on_demand(
        #[values(ProbeUsage::Once, ProbeUsage::Repeated)] usage: ProbeUsage,
    ) -> VortexResult<()> {
        let mut ctx = crate::array_session().create_execution_ctx();
        let array = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter([1i32, 2]).into_array()),
            ("b", PrimitiveArray::from_iter([3i64, 4]).into_array()),
        ])?
        .into_array();
        let mut state = ProbeState::<()>::new(usage);

        assert!(state.children.slot(&array, 5).is_err());
        // Slot 0 is the struct's absent validity.
        assert!(state.children.slot(&array, 0)?.is_none());
        assert!(state.children.slots.is_empty());

        let child = state
            .children
            .slot(&array, 2)?
            .ok_or_else(|| vortex_err!("missing slot"))?;
        assert_eq!(child.execute_scalar(1, &mut ctx)?, Scalar::from(4i64));
        assert_eq!(child.retained.is_some(), usage == ProbeUsage::Repeated);

        assert_eq!(state.children.slots.len(), array.slots().len());
        assert!(state.children.slots[1].is_none());
        assert!(state.children.slots[2].is_some());
        Ok(())
    }

    #[test]
    fn storage_rejects_mismatched_state_type() -> VortexResult<()> {
        let mut storage = ProbeStorage::default();
        storage.get_or_init::<()>()?;
        assert!(storage.get_or_init::<u8>().is_err());
        Ok(())
    }
}
