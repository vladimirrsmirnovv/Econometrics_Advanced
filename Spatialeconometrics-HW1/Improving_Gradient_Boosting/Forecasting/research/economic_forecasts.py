"""Expanding, forecast-only SLX/SDM and local ridge models at market level."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def expanding_forecasts(log_monthly, static, weights, local_weights):
    """All predictions at t are estimated on realised outcomes strictly before t.

    SDM uses pooled historical spatial 2SLS and a reduced-form joint forecast.
    It is a predictive experiment; short panels do not justify causal inference.
    Unstable IV estimates fall back to expanding SLX, explicitly recorded.
    """
    n, periods = log_monthly.shape
    growth = np.column_stack([np.full(n, np.nan), np.diff(log_monthly, axis=1)])
    covariates = []
    for t in range(periods):
        last = growth[:, t-1] if t >= 2 else np.zeros(n)
        mean3 = np.nanmean(growth[:, max(1,t-3):t], axis=1) if t >= 2 else np.zeros(n)
        seasonal = growth[:, t-12] if t >= 13 else np.zeros(n)
        own = np.column_stack([last, mean3, seasonal, static[:, :3],
             np.full(n, np.sin(2*np.pi*(t % 12 + 1)/12)),
             np.full(n, np.cos(2*np.pi*(t % 12 + 1)/12))])
        own = np.nan_to_num(own)
        peer = weights @ own[:, :6]
        covariates.append((own, np.column_stack([own, peer])))
    predictions = {name: np.full((n,periods),np.nan) for name in ['slx','sdm','gwr']}
    metadata = []
    local_a = np.zeros((n,5,5))
    local_b = np.zeros((n,5))
    local_graph = (sparse.eye(n,format='csr') + local_weights) * 0.5
    eye = sparse.eye(n,format='csr')
    for t in range(3,periods):
        x = np.vstack([covariates[s][1] for s in range(2,t)])
        y = np.concatenate([growth[:,s] for s in range(2,t)])
        finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
        center=x[finite].mean(axis=0); scale=x[finite].std(axis=0)
        scale[scale < 1e-10] = 1
        z = np.column_stack([np.ones(finite.sum()),(x[finite]-center)/scale])
        zz = z.T @ z
        penalty = np.eye(z.shape[1])*10
        penalty[0,0]=0
        coefficients=np.linalg.solve(zz+penalty,z.T@y[finite])
        current=np.column_stack([np.ones(n),(covariates[t][1]-center)/scale])
        predictions['slx'][:,t]=current@coefficients

        # Estimate endogenous current Wy only on historical months, then solve
        # the future spatial system without using realised y_t anywhere.
        wy=np.concatenate([weights@growth[:,s] for s in range(2,t)])[finite]
        instrument_extras=np.vstack([weights@(weights@covariates[s][0][:,:6]) for s in range(2,t)])[finite]
        instrument_center=instrument_extras.mean(axis=0)
        instrument_scale=instrument_extras.std(axis=0)
        instrument_scale[instrument_scale<1e-10]=1
        h=np.column_stack([z,(instrument_extras-instrument_center)/instrument_scale])
        endogenous=np.column_stack([z,wy])
        stage1,_,instrument_rank,_=np.linalg.lstsq(h,endogenous,rcond=1e-8)
        fitted=h@stage1
        iv,_,second_rank,_=np.linalg.lstsq(fitted,y[finite],rcond=1e-8)
        rho=float(iv[-1])
        stable=bool(np.isfinite(rho) and abs(rho)<0.95 and second_rank==endogenous.shape[1])
        if stable:
            predictions['sdm'][:,t]=spsolve(eye-rho*weights,current@iv[:-1])
        else:
            predictions['sdm'][:,t]=predictions['slx'][:,t]
        metadata.append(dict(time=t,training_times=list(range(2,t)),rho=rho,
            instrument_rank=int(instrument_rank),second_stage_rank=int(second_rank),
            stable_joint_forecast=stable,unstable_policy='fallback_to_expanding_SLX'))

        # Local ridge fits: cumulative historical moments, never current labels.
        def local_x(s):
            own=covariates[s][0]
            return np.column_stack([np.ones(n),10*own[:,0],10*own[:,1],own[:,-2:]])
        previous=t-1
        lx=local_x(previous)
        ly=np.nan_to_num(growth[:,previous])
        local_a+=np.einsum('ni,nj->nij',lx,lx)
        local_b+=lx*ly[:,None]
        a=(local_graph@local_a.reshape(n,-1)).reshape(n,5,5)
        b=local_graph@local_b
        a+=np.eye(5)[None,:,:]*(0.1*(t-2))
        local_coef=np.linalg.solve(a,b[...,None])[...,0]
        predictions['gwr'][:,t]=np.einsum('ni,ni->n',local_x(t),local_coef)
    for value in predictions.values():
        value[:]=np.clip(value,-0.7,0.7)
    return predictions,metadata
