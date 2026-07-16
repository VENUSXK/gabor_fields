# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
Bounds-aware Adam optimizer (BoundedAdam) for Dr.Jit / Mitsuba training.
Image losses / metrics (L1, L2, PSNR, SSIM) now live in gfields.losses.
'''

from __future__ import annotations # Delayed parsing of type annotations

import os

import drjit as dr
import mitsuba as mi
import numpy as np


# @dr.freeze
def _apply_bounds(new_value, original_value, upper, lower):
    '''Frozen bounds handling for optimizer step'''
    Bool = dr.mask_t(dr.leaf_t(new_value))
    over_boundary = Bool(False)
    
    if upper is not None:
        over_upper = new_value >= upper
        over_boundary |= over_upper
        clamped = dr.select(original_value >= upper, upper, original_value)
        new_value = dr.select(over_upper, clamped + 0.5 * (upper - clamped), new_value)
    
    if lower is not None:
        over_lower = new_value <= lower
        over_boundary |= over_lower
        clamped = dr.select(original_value <= lower, lower, original_value)
        new_value = dr.select(over_lower, clamped + 0.5 * (lower - clamped), new_value)
    
    return new_value, over_boundary


class BoundedAdam(dr.opt.Adam):
    '''
    Implements a bounds-aware Adam optimizer.

    The BoundedAdam class allows for setting bounds on each parameter. If a
    gradient step reaches one of the bounds, the optimized value will be moved
    by half of the distance towards the bound instead. This ensures that the
    optimization process can approach the bound as closely as possible without
    ever stepping over it. This usually works better than clamping as this
    process will also reset the optimizer state for this parameter.
    '''
    def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                 mask_updates=False, uniform=False, params=None, promote_fp16=True, noise_init=5e2, noise_decay=0.):
        '''
        Parameter ``lr``:
            learning rate

        Parameter ``beta_1``:
            controls the exponential averaging of first order gradient moments

        Parameter ``beta_2``:
            controls the exponential averaging of second order gradient moments

        Parameter ``mask_updates``:
            if enabled, parameters and state variables will only be updated in a
            given iteration if it received nonzero gradients in that iteration

        Parameter ``uniform``:
            if enabled, the optimizer will use the 'UniformAdam' variant of Adam
            [Nicolet et al. 2021], where the update rule uses the *maximum* of
            the second moment estimates at the current step instead of the
            per-element second moments.

        Parameter ``params`` (:py:class:`dict`):
            Optional dictionary-like object containing parameters to optimize.
        '''
        self.noise_lr = noise_init
        self.noise_decay = noise_decay
        # Opt-in per-step NaN diagnostics (2 device syncs per parameter); the
        # fused NaN scrub in step() is always on regardless.
        self.debug_nan = os.environ.get('VOLPRIM_DEBUG_NAN', '0') == '1'
        self.bounds = {}
        super().__init__(
            lr,
            params,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            mask_updates=mask_updates,
            promote_fp16=promote_fp16,
            uniform=uniform
        )

    def set_bounds(self, key, upper=None, lower=None):
        '''
        Set bounds for a parameter
        '''
        # Equal bounds are allowed: they freeze the parameter at that value.
        assert lower is None or upper is None or lower <= upper, 'Upper bound should be higher than lower bound! Did you mix the argument order?'
        self.bounds[key] = (upper, lower)

    @property
    def variables(self):
        '''
        Backwards compatibility: return dict-like access to parameter values.
        In the new DrJit API, use opt[key] or iterate with opt.keys()/values()/items().
        '''
        return {k: self[k] for k in self.keys()}

    def step(self, *, eval: bool = True, grad_scale=None, active=None):
        '''
        Take a gradient step with bounds handling.
        '''
        from drjit.opt import _LRCache

        with dr.profile_range('BoundedAdam.step()'):
            cache = _LRCache()

            def sigmoid(x):
                return 1 / (1 + dr.exp(-x))

            # MCMC relocation noise on the centers. Skip the whole block once the
            # noise has decayed to (or was configured as) ~zero: the host-side
            # np.random.randn(3, N) draw + transfer and the cov matmul chain are
            # pure per-step overhead when the resulting displacement is negligible
            # (lr * noise_lr < 1e-8 of the [-1,1] volume).
            noises = {}
            if abs(self.noise_lr) > 1e-6:
                with dr.suspend_grad():
                    for key in list(self.keys()):
                        if 'centers' in key:
                            centers = self[key]
                            lvl = int(key.split('_')[-1])
                            t = 1e-6
                            N01 = mi.Vector3f(np.random.randn(*centers.shape))
                            SS = dr.diag(self[f'scales_{lvl}']**2)
                            R = dr.quat_to_matrix(mi.Matrix3f, self[f'quaternions_{lvl}'])
                            cov = R @ SS @ R.T
                            noise = self.noise_lr * sigmoid(-100*(t - dr.abs(self[f'opacities_{lvl}']))) * cov @ N01
                            noises[key] = noise
            self.noise_lr *= self.noise_decay

            for key, (value, promoted, lr, extra) in self.state.items():
                # Fetch gradient and apply scaling
                grad = value.grad.array
                if grad_scale is not None:
                    grad *= grad_scale

                # NaN diagnostics are opt-in (debug_nan / VOLPRIM_DEBUG_NAN=1):
                # each `if dr.any(...)` forces a kernel eval + host sync, i.e.
                # ~2 syncs per parameter per step. The scrub below stays fused in
                # the step kernel, so silent NaN gradients are still neutralized.
                if self.debug_nan:
                    if dr.any(dr.any(dr.isnan(grad))):
                        print('Grad has nan:', key)
                    if dr.any(dr.any(dr.isnan(value.array))):
                        print('Param has nan:', key)
                grad = dr.select(dr.isnan(grad), 0, grad)

                # Use default or parameter-specific learning rate
                lr_v = lr if lr is not None else self.lr

                # Get original value for bounds computation
                value_flat = dr.detach(value).array
                original_value = value_flat

                # Adam step (uses parent's _step with LR cache)
                new_value, new_extra = self._step(cache, value_flat, grad, lr_v, extra)

                if key in noises.keys():
                    new_value -= lr_v * noises[key]


                # Bounds handling (using frozen helper)
                if key in self.bounds:
                    upper, lower = self.bounds[key]
                    new_value, over_boundary = _apply_bounds(new_value, original_value, upper, lower)
                    
                    # Reset momentum for elements that hit boundaries.
                    # drjit >= 1.3 carries a fourth optional v_max entry
                    # (amsgrad); reset it on the same lanes when present.
                    t, m_t, v_t, *rest = new_extra
                    v_max = rest[0] if rest else None
                    if v_max is not None:
                        v_max = dr.select(over_boundary, 0, v_max)
                    new_extra = (
                        t,
                        dr.select(over_boundary, 0, m_t),
                        dr.select(over_boundary, 0, v_t),
                        *( (v_max,) if rest else () ),
                    )

                # Optional masking (from base class logic)
                mask = False
                if self.mask_updates:
                    mask |= grad == 0
                if active is not None:
                    mask |= ~active

                if mask is not False:
                    new_value = dr.select(mask, value_flat, new_value)
                    new_extra = self._select(mask, extra, new_extra)

                # Reconstruct value type if needed
                value_tp = type(value)
                if type(new_value) is not value_tp:
                    if dr.is_tensor_v(value_tp):
                        new_value = value_tp(new_value, value.shape)
                    else:
                        new_value = value_tp(new_value)
                dr.enable_grad(new_value)

                # Update state
                new_state = new_value, promoted, lr, new_extra
                dr.schedule(new_state)
                self.state[key] = new_state

            if eval:
                dr.eval()

    def __repr__(self):
        return ('BoundedAdam[\n'
                '  parameters = %s,\n'
                '  lr = %s,\n'
                '  betas = (%g, %g),\n'
                '  eps = %g,\n'
                '  bounds = %s\n'
                ']' % (list(self.keys()), self.lr,
                       self.beta_1, self.beta_2, self.epsilon, self.bounds))
