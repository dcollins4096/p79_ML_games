import numpy as np

try:
    from scipy.ndimage import uniform_filter
except ImportError:
    raise ImportError("This code requires scipy: pip install scipy")


def _local_mean(arr, size):
    """Local mean with a square top-hat kernel."""
    return uniform_filter(arr.astype(np.float64), size=size, mode="nearest")


def polarization_angle_from_qu(q, u):
    """
    Polarization angle from Stokes Q,U in radians.
    Angle is in [-pi/2, pi/2).
    """
    return 0.5 * np.arctan2(u, q)


def local_angle_dispersion_from_qu(q, u, window=9, min_pfrac=None):
    """
    Estimate local angular dispersion sigma_phi from Stokes Q,U
    using circular statistics on doubled angles.

    Parameters
    ----------
    q, u : 2D arrays
        Stokes Q and U maps.
    window : int
        Size of local square window.
    min_pfrac : float or None
        Optional minimum polarized intensity threshold. If given,
        pixels with sqrt(Q^2+U^2) below this are masked out.

    Returns
    -------
    sigma_phi : 2D array
        Local polarization-angle dispersion in radians.
    psi : 2D array
        Polarization angle map in radians.
    """
    psi = polarization_angle_from_qu(q, u)

    p = np.sqrt(q**2 + u**2)
    if min_pfrac is not None:
        mask = p >= min_pfrac
    else:
        mask = np.ones_like(p, dtype=bool)

    # Orientation statistics should use 2*psi because polarization
    # angles are defined modulo pi, not 2*pi.
    c2 = np.cos(2.0 * psi) * mask
    s2 = np.sin(2.0 * psi) * mask
    w  = mask.astype(np.float64)

    c2_bar = _local_mean(c2, window)
    s2_bar = _local_mean(s2, window)
    w_bar  = _local_mean(w,  window)

    # Avoid dividing by very small weights near masked regions.
    good = w_bar > 1e-6
    c2_bar[good] /= w_bar[good]
    s2_bar[good] /= w_bar[good]

    # Mean resultant length for orientation data
    R = np.sqrt(c2_bar**2 + s2_bar**2)
    R = np.clip(R, 1e-12, 1.0)

    # Circular std for doubled angles, then divide by 2
    sigma_phi = 0.5 * np.sqrt(-2.0 * np.log(R))

    # Optional safety cap: DCF assumes small-angle regime
    # You may want to ignore pixels above ~25 deg = 0.44 rad
    sigma_phi[~good] = np.nan

    return sigma_phi, psi


def dcf_bpos_map(
    projected_density,
    vel_variance,
    q_stokes,
    u_stokes,
    los_depth,
    window=9,
    q_correction=0.5,
    unit_system="cgs",
    density_is_column=True,
    sigma_phi_max=np.deg2rad(25.0),
    min_p=None,
):
    """
    Compute a DCF estimate of plane-of-sky magnetic field strength.

    Parameters
    ----------
    projected_density : 2D array
        If density_is_column=True, this is column density Sigma [g/cm^2 in cgs
        or kg/m^2 in SI]. Otherwise it is volume density rho.
    vel_variance : 2D array
        LOS velocity variance map. Units should be (cm/s)^2 in cgs or (m/s)^2 in SI.
        The code uses sigma_v = sqrt(local mean(vel_variance)).
    q_stokes, u_stokes : 2D arrays
        Stokes Q and U maps.
    los_depth : float
        Assumed line-of-sight depth used to convert column density to effective
        volume density: rho_eff = Sigma / los_depth.
        Ignored if density_is_column=False.
    window : int
        Local patch size for estimating angle dispersion and local averages.
    q_correction : float
        DCF correction factor. Often ~0.5.
    unit_system : {"cgs", "si"}
        cgs uses B = Qc * sqrt(4*pi*rho) * sigma_v / sigma_phi
        si  uses B = Qc * sqrt(mu0*rho)   * sigma_v / sigma_phi
    density_is_column : bool
        Whether projected_density is a column density map.
    sigma_phi_max : float
        Mask pixels where angular dispersion exceeds this limit.
        DCF is most reliable in small-angle regime.
    min_p : float or None
        Minimum polarized intensity threshold for using Q,U.

    Returns
    -------
    result : dict
        Dictionary with:
          - "B_dcf"         : DCF plane-of-sky field estimate
          - "rho_eff"       : effective volume density used
          - "sigma_v"       : local velocity dispersion
          - "sigma_phi"     : local angle dispersion
          - "psi"           : polarization angle
          - "valid_mask"    : mask where DCF estimate is considered valid
    """
    Sigma_or_rho = np.asarray(projected_density, dtype=np.float64)
    vel_variance = np.asarray(vel_variance, dtype=np.float64)
    q_stokes = np.asarray(q_stokes, dtype=np.float64)
    u_stokes = np.asarray(u_stokes, dtype=np.float64)

    if density_is_column:
        rho_eff = Sigma_or_rho / float(los_depth)
    else:
        rho_eff = Sigma_or_rho.copy()

    # Local density and local velocity dispersion
    rho_loc = _local_mean(rho_eff, window)
    var_loc = _local_mean(vel_variance, window)
    var_loc = np.clip(var_loc, 0.0, None)
    sigma_v = np.sqrt(var_loc)

    # Local angle dispersion from Q,U
    sigma_phi, psi = local_angle_dispersion_from_qu(
        q_stokes, u_stokes, window=window, min_pfrac=min_p
    )

    valid = np.isfinite(sigma_phi)
    valid &= sigma_phi > 0.0
    valid &= sigma_phi < sigma_phi_max
    valid &= np.isfinite(rho_loc) & (rho_loc > 0.0)
    valid &= np.isfinite(sigma_v)

    B = np.full_like(rho_loc, np.nan, dtype=np.float64)

    if unit_system.lower() == "cgs":
        prefac = np.sqrt(4.0 * np.pi * rho_loc)
    elif unit_system.lower() == "si":
        mu0 = 4.0e-7 * np.pi
        prefac = np.sqrt(mu0 * rho_loc)
    else:
        raise ValueError("unit_system must be 'cgs' or 'si'")

    B[valid] = q_correction * prefac[valid] * sigma_v[valid] / sigma_phi[valid]

    return {
        "B_dcf": B,
        "rho_eff": rho_loc,
        "sigma_v": sigma_v,
        "sigma_phi": sigma_phi,
        "psi": psi,
        "valid_mask": valid,
    }


def compare_to_target(predicted_B, target_B, mask=None, eps=1e-30):
    """
    Simple metrics for comparing DCF or ML predictions to target field strength.
    """
    pred = np.asarray(predicted_B, dtype=np.float64)
    targ = np.asarray(target_B, dtype=np.float64)

    good = np.isfinite(pred) & np.isfinite(targ) & (targ > 0)
    if mask is not None:
        good &= mask

    if good.sum() == 0:
        return {"n": 0}

    x = pred[good]
    y = targ[good]

    frac = x / (y + eps)
    logerr = np.log10((x + eps) / (y + eps))

    rmse = np.sqrt(np.mean((x - y) ** 2))
    mae = np.mean(np.abs(x - y))
    mean_log10_ratio = np.mean(logerr)
    scatter_log10_ratio = np.std(logerr)

    # Pearson correlation
    xc = x - x.mean()
    yc = y - y.mean()
    corr = np.sum(xc * yc) / np.sqrt(np.sum(xc**2) * np.sum(yc**2) + eps)

    return {
        "n": int(good.sum()),
        "rmse": rmse,
        "mae": mae,
        "mean_pred_over_true": np.mean(frac),
        "median_pred_over_true": np.median(frac),
        "mean_log10_pred_over_true": mean_log10_ratio,
        "std_log10_pred_over_true": scatter_log10_ratio,
        "pearson_r": corr,
    }

# Example inputs
# projected_density : 2D map
# mean_velocity     : 2D map  (not directly used below, but you could use it later)
# vel_variance      : 2D map
# Q_map, U_map      : 2D maps
# B_target          : 2D map of true/projected magnetic field strength

def do_it(projected_density, vel_variance, Q_map, U_map, B_target):
    los_depth = 1.0         # put this in your simulation length units
    window = 9              # local patch for angle dispersion and local averaging
    q_correction = 0.5

    dcf = dcf_bpos_map(
            projected_density=projected_density,
            vel_variance=vel_variance,
            q_stokes=Q_map,
            u_stokes=U_map,
            los_depth=los_depth,
            window=window,
            q_correction=q_correction,
            unit_system="cgs",          # or "si"
            density_is_column=True,     # likely true for projected density
            sigma_phi_max=np.deg2rad(25.0),
    )

    B_dcf = dcf["B_dcf"]
    valid = dcf["valid_mask"]

    metrics = compare_to_target(B_dcf, B_target, mask=valid)
    print(metrics)
    return B_dcf, valid

