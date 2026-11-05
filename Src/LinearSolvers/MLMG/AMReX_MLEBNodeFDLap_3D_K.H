#ifndef AMREX_MLEBNODEFDLAP_3D_K_H_
#define AMREX_MLEBNODEFDLAP_3D_K_H_

namespace amrex {

// See the 2D header for the EB position convention.

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_scale_rhs (int i, int j, int k, Array4<Real> const& rhs,
                            Array4<int const> const& dmsk, Array4<Real const> const& ebpx,
                            Array4<Real const> const& ebpy, Array4<Real const> const& ebpz) noexcept
{
    if (!dmsk(i,j,k)) {
        Real const s = amrex::min(mlebndfdlap_hm(ebpx(i-1,j,k)), ebpx(i,j,k),
                                  mlebndfdlap_hm(ebpy(i,j-1,k)), ebpy(i,j,k),
                                  mlebndfdlap_hm(ebpz(i,j,k-1)), ebpz(i,j,k));
        rhs(i,j,k) *= s;
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real mlebndfdlap_scaled_hinv (Real scale, Real h) noexcept
{
    return (h == Real(0.0)) ? Real(1.0) : scale / h;
}

template <typename S>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_adotx (int i, int j, int k, Array4<Real> const& y,
                        Array4<Real const> const& x, Array4<int const> const& dmsk,
                        S const& sig, Real bx, Real by, Real bz) noexcept
{
    if (dmsk(i,j,k)) {
        y(i,j,k) = Real(0.0);
    } else {
        Real const sigxm = sig.x(i-1,j,k);
        Real const sigxp = sig.x(i  ,j,k);
        Real const sigym = sig.y(i,j-1,k);
        Real const sigyp = sig.y(i,j  ,k);
        Real const sigzm = sig.z(i,j,k-1);
        Real const sigzp = sig.z(i,j,k  );
        y(i,j,k) = bx * (sigxm*x(i-1,j,k) + sigxp*x(i+1,j,k))
            +      by * (sigym*x(i,j-1,k) + sigyp*x(i,j+1,k))
            +      bz * (sigzm*x(i,j,k-1) + sigzp*x(i,j,k+1))
            - (bx*(sigxm+sigxp) + by*(sigym+sigyp) + bz*(sigzm+sigzp)) * x(i,j,k);
    }
}

template <typename S>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_gsrb (int i, int j, int k, Array4<Real> const& x,
                       Array4<Real const> const& rhs, Array4<int const> const& dmsk,
                       S const& sig, Real bx, Real by, Real bz, int redblack) noexcept
{
    if ((i+j+k+redblack)%2 == 0) {
        if (dmsk(i,j,k)) {
            x(i,j,k) = Real(0.);
        } else {
            Real const sigxm = sig.x(i-1,j,k);
            Real const sigxp = sig.x(i  ,j,k);
            Real const sigym = sig.y(i,j-1,k);
            Real const sigyp = sig.y(i,j  ,k);
            Real const sigzm = sig.z(i,j,k-1);
            Real const sigzp = sig.z(i,j,k  );
            Real const gamma = -(bx*(sigxm+sigxp) + by*(sigym+sigyp) + bz*(sigzm+sigzp));
            Real const Ax = bx * (sigxm*x(i-1,j,k) + sigxp*x(i+1,j,k))
                +           by * (sigym*x(i,j-1,k) + sigyp*x(i,j+1,k))
                +           bz * (sigzm*x(i,j,k-1) + sigzp*x(i,j,k+1))
                + gamma * x(i,j,k);
            constexpr Real omega = Real(1.25);
            x(i,j,k) += (rhs(i,j,k) - Ax) * (omega / gamma);
        }
    }
}

// Symmetric EB stencil: see the 2D header.

//! Open length from node (i,j,k) toward side s (0 low, 1 high) in direction d.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real mlebndfdlap_open_length (int i, int j, int k, int d, int s,
                              Array4<Real const> const& ebpx, Array4<Real const> const& ebpy,
                              Array4<Real const> const& ebpz) noexcept
{
    if (d == 0) {
        return s ? ebpx(i,j,k) : mlebndfdlap_hm(ebpx(i-1,j,k));
    } else if (d == 1) {
        return s ? ebpy(i,j,k) : mlebndfdlap_hm(ebpy(i,j-1,k));
    } else {
        return s ? ebpz(i,j,k) : mlebndfdlap_hm(ebpz(i,j,k-1));
    }
}

/**
 * \brief Link factors of a node, without sigma and the row scale: b_d times
 * Shortley-Weller 2 / (h- + h+), or b_d for the symmetric stencil.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_link_factors (Real const (&h)[3][2], Real const (&b)[3], bool sym,
                               Real (&f)[3][2]) noexcept
{
    for (int d = 0; d < 3; ++d) {
        Real const c = sym ? b[d] : b[d] * Real(2.0) / (h[d][0]+h[d][1]);
        f[d][0] = c;
        f[d][1] = c;
    }
}

//! Open lengths and open links of node (i,j,k), indexed [direction][side].
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_links (int i, int j, int k, Array4<Real const> const& levset,
                        Array4<Real const> const& ebpx, Array4<Real const> const& ebpy,
                        Array4<Real const> const& ebpz,
                        Real (&h)[3][2], bool (&open)[3][2]) noexcept
{
    for (int d = 0; d < 3; ++d) {
        for (int s = 0; s < 2; ++s) {
            int const o = s ? 1 : -1;
            h[d][s] = mlebndfdlap_open_length(i,j,k,d,s,ebpx,ebpy,ebpz);
            open[d][s] = levset(i + (d==0 ? o : 0), j + (d==1 ? o : 0), k + (d==2 ? o : 0))
                < Real(0.0) && h[d][s] == Real(1.0);
        }
    }
}

template <typename S>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real mlebndfdlap_edge_sig (S const& sig, int i, int j, int k, int d, int s) noexcept
{
    if (d == 0) { return s ? sig.x(i,j,k) : sig.x(i-1,j,k); }
    if (d == 1) { return s ? sig.y(i,j,k) : sig.y(i,j-1,k); }
    return s ? sig.z(i,j,k) : sig.z(i,j,k-1);
}

template <typename S, typename F>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_adotx_eb_doit (int i, int j, int k, Array4<Real> const& y,
                                Array4<Real const> const& x, Array4<Real const> const& levset,
                                Array4<int const> const& dmsk,
                                Array4<Real const> const& ebpx, Array4<Real const> const& ebpy,
                                Array4<Real const> const& ebpz,
                                S const& sig, F const& xeb, Real bx, Real by, Real bz,
                                bool sym) noexcept
{
    if (dmsk(i,j,k)) {
        y(i,j,k) = Real(0.0);
    } else {
        Real const xc = x(i,j,k);
        Real h[3][2];
        bool open[3][2];
        mlebndfdlap_links(i,j,k,levset,ebpx,ebpy,ebpz,h,open);
        Real const scale = amrex::min(h[0][0],h[0][1],h[1][0],h[1][1],h[2][0],h[2][1]);
        Real const b[3] = {bx, by, bz};
        Real f[3][2];
        mlebndfdlap_link_factors(h,b,sym,f);

        Real out = Real(0.0);
        for (int d = 0; d < 3; ++d) {
            for (int s = 1; s >= 0; --s) { // high side first, as Shortley-Weller had it
                int const o = s ? 1 : -1;
                int const ii = i + (d==0 ? o : 0);
                int const jj = j + (d==1 ? o : 0);
                int const kk = k + (d==2 ? o : 0);
                Real const c = mlebndfdlap_edge_sig(sig,i,j,k,d,s) * f[d][s];
                if (open[d][s]) {
                    out += c * scale * (x(ii,jj,kk) - xc);
                } else {
                    out += c * mlebndfdlap_scaled_hinv(scale, h[d][s]) * (xeb(ii,jj,kk) - xc);
                }
            }
        }
        y(i,j,k) = out;
    }
}

template <typename S>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_adotx_eb (int i, int j, int k, Array4<Real> const& y,
                           Array4<Real const> const& x, Array4<Real const> const& levset,
                           Array4<int const> const& dmsk,
                           Array4<Real const> const& ebpx, Array4<Real const> const& ebpy,
                           Array4<Real const> const& ebpz,
                           S const& sig, Real xeb, Real bx, Real by, Real bz, bool sym) noexcept
{
    mlebndfdlap_adotx_eb_doit(i, j, k, y, x, levset, dmsk, ebpx, ebpy, ebpz, sig,
                              [=] (int, int, int) -> Real { return xeb; },
                              bx, by, bz, sym);
}

template <typename S>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_adotx_eb (int i, int j, int k, Array4<Real> const& y,
                           Array4<Real const> const& x, Array4<Real const> const& levset,
                           Array4<int const> const& dmsk,
                           Array4<Real const> const& ebpx, Array4<Real const> const& ebpy,
                           Array4<Real const> const& ebpz,
                           S const& sig, Array4<Real const> const& xeb,
                           Real bx, Real by, Real bz, bool sym) noexcept
{
    mlebndfdlap_adotx_eb_doit(i, j, k, y, x, levset, dmsk, ebpx, ebpy, ebpz, sig,
                              [=] (int i1, int i2, int i3) -> Real {
                                  return xeb(i1,i2,i3); },
                              bx, by, bz, sym);
}

template <typename S>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlebndfdlap_gsrb_eb (int i, int j, int k, Array4<Real> const& x,
                          Array4<Real const> const& rhs, Array4<Real const> const& levset,
                          Array4<int const> const& dmsk,
                          Array4<Real const> const& ebpx, Array4<Real const> const& ebpy,
                          Array4<Real const> const& ebpz,
                          S const& sig, Real bx, Real by, Real bz, int redblack,
                          bool sym) noexcept
{
    if ((i+j+k+redblack)%2 == 0) {
        if (dmsk(i,j,k)) {
            x(i,j,k) = Real(0.);
        } else {
            Real h[3][2];
            bool open[3][2];
            mlebndfdlap_links(i,j,k,levset,ebpx,ebpy,ebpz,h,open);
            Real const scale = amrex::min(h[0][0],h[0][1],h[1][0],h[1][1],h[2][0],h[2][1]);
            Real const b[3] = {bx, by, bz};
            Real f[3][2];
            mlebndfdlap_link_factors(h,b,sym,f);

            bool cut = false; // a link to the EB
            Real gamma = Real(0.0);
            Real rho = Real(0.0);
            for (int d = 0; d < 3; ++d) {
                for (int s = 1; s >= 0; --s) {
                    int const o = s ? 1 : -1;
                    Real const c = mlebndfdlap_edge_sig(sig,i,j,k,d,s) * f[d][s];
                    if (open[d][s]) {
                        gamma -= c * scale;
                        rho += c * scale * x(i + (d==0 ? o : 0), j + (d==1 ? o : 0),
                                             k + (d==2 ? o : 0));
                    } else {
                        cut = true;
                        gamma -= c * mlebndfdlap_scaled_hinv(scale, h[d][s]);
                    }
                }
            }

            Real const Ax = rho + gamma*x(i,j,k);
            // Over-relaxing rows cut by the EB can make MG diverge.
            Real const omega = cut ? Real(1.0) : Real(1.25);
            x(i,j,k) += (rhs(i,j,k) - Ax) * (omega / gamma);
        }
    }
}

}

#endif
