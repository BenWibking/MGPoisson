//==============================================================================
// AMRPoisson
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_poisson.cpp
/// \brief Defines a test problem for a cell-centered Poisson solve.
///

#include "AMReX_Array.H"
#include "AMReX_Config.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_Loop.H"
#include "AMReX_MLMG.H"
#include "AMReX_MLPoisson.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>
#include <type_traits>

#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "face_box.hpp"
#include "multipole.hpp"
#include "test_poisson.hpp"

auto problem_main() -> int
{
	// Solve the Poisson equation with free-space boundary conditions using the method of James
	// (1977).
	// Additional reference: Moon, Kim, & Ostriker (2019) [ApJS, 241:24 (20pp), 2019 April].
	// They re-derive the method for the standard 2nd-order cell-centered stencil and emphasize
	// the importance of using the *discrete* Green's function (DGF) for the surface
	// charge-induced potential.

	// initialize geometry
	const int n_cell = 16;
	const int max_grid_size = 8;
	const double Lx = 1.0;
	const double FourPiG = 1.0;
	const int ncomp = 1;  // number of components  [should always be 1]
	const int nghost = 1; // number of ghost cells [should always be 1]
	const int nlev = 1;   // number of AMR levels

	amrex::Box domain(amrex::IntVect{AMREX_D_DECL(0, 0, 0)},
			  amrex::IntVect{AMREX_D_DECL(n_cell - 1, n_cell - 1, n_cell - 1)});
	amrex::RealBox boxSize{{AMREX_D_DECL(amrex::Real(0.0), amrex::Real(0.0), amrex::Real(0.0))},
			       {AMREX_D_DECL(amrex::Real(Lx), amrex::Real(Lx), amrex::Real(Lx))}};

	// set boundary condition type (Dirichlet)
	amrex::Array<int, AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0, 0, 0)};
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_lo;
	amrex::Array<amrex::LinOpBCType, AMREX_SPACEDIM> bc_hi;
	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
		bc_lo[i] = amrex::LinOpBCType::Dirichlet;
		bc_hi[i] = amrex::LinOpBCType::Dirichlet;
	}

	// create single-level Cartesian grids
	amrex::Geometry geom(domain, boxSize, 0, is_periodic);
	amrex::BoxArray grids(domain);
	grids.maxSize(max_grid_size);
	amrex::DistributionMapping dmap{grids};

	// create MLPoisson object
	amrex::MLPoisson poissoneq({geom}, {grids}, {dmap});
	poissoneq.setDomainBC(bc_lo, bc_hi);

	// set order of extrapolation for ghost cells
	// (see	https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html)
	// "It should be emphasized that the data in levelbcdata for Dirichlet or Neumann boundaries
	// are assumed to be exactly on the face of the physical domain; storing these values in the
	// ghost cell of a cell-centered array is a convenience of implementation."
	poissoneq.setMaxOrder(3); // default = 3

	// create MLMG object
	amrex::MLMG mlmg(poissoneq);
	mlmg.setVerbose(1);
	mlmg.setBottomVerbose(0);
	mlmg.setBottomSolver(amrex::MLMG::BottomSolver::bicgstab);
	mlmg.setMaxFmgIter(0); // only helps if problem is very smooth

	// create MultiFabs
	amrex::MultiFab phi(grids, dmap, ncomp, nghost);
	amrex::MultiFab rhs(grids, dmap, ncomp, nghost);
	amrex::Vector<amrex::MultiFab *> phi_levels(nlev);
	amrex::Vector<amrex::MultiFab const *> rhs_levels(nlev);
	phi_levels[0] = &phi;
	rhs_levels[0] = &rhs;

	// get boundary faceBoxes
	auto faceBoxes = getFaceBoxes(geom, phi);

	// set initial guess for phi
	phi.setVal(0);

	// set density field
	amrex::Real const R_sphere = 0.5;
	amrex::Real const rho_sphere = 1.0 / ((4. / 3.) * M_PI * std::pow(R_sphere, 3));

	auto prob_lo = geom.ProbLoArray();
	auto prob_hi = geom.ProbHiArray();
	auto dx = geom.CellSizeArray();
	for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto rho = rhs.array(mfi);
		amrex::Real x0 = 0.5 * (prob_hi[0] + prob_lo[0]);
		amrex::Real y0 = 0.5 * (prob_hi[1] + prob_lo[1]);
		amrex::Real z0 = 0.5 * (prob_hi[2] + prob_lo[2]);
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
			amrex::Real const r =
			    std::pow(x - x0, 2) + std::pow(y - y0, 2) + std::pow(z - z0, 2);

			// TODO(benwibking): integrate over cell volume
			if (r <= R_sphere) {
				rho(i, j, k) = FourPiG * rho_sphere * (dx[0] * dx[1] * dx[2]);
			} else {
				rho(i, j, k) = 0;
			}
		});
	}

	// set boundary values (see above for definition)
	for (int n = 0; n < faceBoxes.size(); ++n) {
		// [N.B. Clang 13 does not allow capture of structured bindings...]
		auto &[arr, facebox, orientation] = faceBoxes[n];
		amrex::ParallelFor(facebox,
				   [=, arr = arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					   arr(i, j, k) = 0;
				   });
	}

	// multigrid solution residual tolerances
	const amrex::Real reltol = 1.0e-12; // doesn't work below ~1e-13...
	const amrex::Real abstol = 0;	    // unused if zero

	/// begin free-space solver

	// Step 1. Solve Poisson equation with \phi_bdry = 0.

	poissoneq.setLevelBC(0,
			     &phi); // set Dirichlet boundary conditions using ghost cells of 'phi'
	amrex::Real residual_linf = mlmg.solve(phi_levels, rhs_levels, reltol, abstol); // solve

	amrex::Print() << "Residual max norm = " << residual_linf << "\n\n";

	// Step 2. Compute surface charge (4 \pi \sigma).
	// 	(This is saved in the ghost cells of 'phi'.)

	for (int n = 0; n < faceBoxes.size(); ++n) {
		auto &[arr, facebox, o] = faceBoxes[n];

		amrex::ParallelFor(
		    facebox, [=, arr = arr, o = o] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			    amrex::GpuArray<int, 3> bdry{};
			    if (o.isLow()) {
				    bdry = domain.loVect3d();
			    } else {
				    bdry = domain.hiVect3d();
			    }

			    // Eq. 33 of Moon et al. See also Figure 1.
			    // 	[== 4piG*rho(i,j,k)]
			    if (o.coordDir() == 0) { // x-face
				    arr(i, j, k) = arr(bdry[0], j, k) / (dx[0] * dx[0]);
			    } else if (o.coordDir() == 1) { // y-face
				    arr(i, j, k) = arr(i, bdry[1], k) / (dx[1] * dx[1]);
			    } else if (o.coordDir() == 2) { // z-face
				    arr(i, j, k) = arr(i, j, bdry[2]) / (dx[2] * dx[2]);
			    }
		    });
	}

	// Step 2b. Compute Cartesian multipoles of surface charge for M local faceBoxes.

	std::vector<Multipole> multipoles(faceBoxes.size());
	for (int n = 0; n < faceBoxes.size(); ++n) {
		auto &[arr, facebox, o] = faceBoxes[n];
		Multipole &mp = multipoles[n];
		auto dx = geom.CellSizeArray();
		auto prob_lo = geom.ProbLoArray();
		auto lo = facebox.loVect3d();
		auto hi = facebox.hiVect3d();
		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> face_lo = {
		    AMREX_D_DECL(lo[0] * dx[0], lo[1] * dx[1], lo[2] * dx[2])};

		amrex::Real x0 =
		    face_lo[0] + amrex::Real(0.5) * dx[0] * (hi[0] - lo[0] + amrex::Real(1));
		amrex::Real y0 =
		    face_lo[1] + amrex::Real(0.5) * dx[1] * (hi[1] - lo[1] + amrex::Real(1));
		amrex::Real z0 =
		    face_lo[2] + amrex::Real(0.5) * dx[2] * (hi[2] - lo[2] + amrex::Real(1));

		amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r0 = {AMREX_D_DECL(x0, y0, z0)};
		for (int l = 0; l < AMREX_SPACEDIM; ++l) {
			mp.r0[l] = r0[l];
		}

		amrex::Loop(
		    facebox, [=, arr = arr, &mp] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			    amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			    amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			    amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];

			    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> r = {
				AMREX_D_DECL(x - x0, y - y0, z - z0)};

			    // monopole
			    mp.q0 += arr(i, j, k); // monopole

			    // dipole
			    for (int l = 0; l < AMREX_SPACEDIM; ++l) {
				    mp.q1[l] += arr(i, j, k) * r[l];
			    }

			    // quadrupole
			    for (int l = 0; l < AMREX_SPACEDIM; ++l) {
				    for (int m = 0; m < AMREX_SPACEDIM; ++m) {
					    mp.q2[l][m] += arr(i, j, k) * r[l] * r[m];
				    }
			    }
		    });
	}

	// Step 2c. MPI_Allgather multipoles. Now each process has multipoles for all N faceBoxes.

	static_assert(std::is_trivially_copyable<Multipole>::value);
	std::vector<Multipole> all_mp(multipoles);

	// Step 3. Compute the potential at each cell in M local faceBoxes, using multipoles of all
	// N faceBoxes. This step has unavoidable complexity O(M*N).

	//#if 0
	amrex::Print() << "x0 y0 z0 mass dipole_x dipole_y\n";
	for (auto &mp : all_mp) {
		amrex::Print() << mp.r0[0] << "\t" << mp.r0[1] << "\t" << mp.r0[2] << "\t";
		amrex::Print() << mp.q0 << "\t";
		amrex::Print() << mp.q1[0] << "\t" << mp.q1[1] << "\t" << mp.q1[2] << "\n";
	}
	//#endif

	// loop over local faceboxes
	for (int n = 0; n < faceBoxes.size(); ++n) {
		auto &[arr, facebox, o] = faceBoxes[n];
		auto dx = geom.CellSizeArray();
		auto prob_lo = geom.ProbLoArray();

		amrex::ParallelFor(facebox,
				   [=, arr = arr] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					   arr(i, j, k) = 0;
				   });

		// loop over all multipoles
		for (auto &mp : all_mp) {
			// multipole expansion is about mp.r0
			amrex::Real x0 = mp.r0[0];
			amrex::Real y0 = mp.r0[1];
			amrex::Real z0 = mp.r0[2];

			amrex::ParallelFor(facebox, [=, arr = arr] AMREX_GPU_DEVICE(
							int i, int j, int k) noexcept {
				// compute cell coordinates
				amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
				amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
				amrex::Real const z = prob_lo[2] + (k + amrex::Real(0.5)) * dx[2];
				amrex::Real r = std::sqrt(AMREX_D_TERM((x - x0) * (x - x0),
								       +(y - y0) * (y - y0),
								       +(z - z0) * (z - z0)));

				// evaluate DGF convolved with each multipole term
				// (N.B. Burkhart [1997] gives an asymptotic expansion of the
				// DGF in (1/r)^n, up to n=5.)
				amrex::Real const u = x / r;
				amrex::Real const v = y / r;
				amrex::Real const w = z / r;

				auto U = [=] AMREX_GPU_DEVICE(int i, int j) {
					amrex::Real Ux = std::pow(dx[0], i) * std::pow(u, j);
					amrex::Real Uy = std::pow(dx[1], i) * std::pow(v, j);
					amrex::Real Uz = std::pow(dx[2], i) * std::pow(w, j);
					return Ux + Uy + Uz;
				};

				amrex::Real K = FourPiG;
				amrex::Real eta3 =
				    (1. / 8.) * (U(2, 0) - 6 * U(2, 2) + 5 * U(2, 4));
				amrex::Real V = (dx[0] * dx[1] * dx[2]);
				amrex::Real dx2 = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);

				if (r == 0.) {
					arr(i, j, k) += K * mp.q0 * (dx2 / V);
				} else {
					arr(i, j, k) += K * mp.q0 * (1.0 / r);
					arr(i, j, k) += K * mp.q0 * (eta3 / (r * r * r));
				}
			});
		}
	}
	amrex::Print() << std::endl;

	// Step 4. Solve for the potential with Dirichlet boundary conditions given by
	// the potential computed in step 3.
	poissoneq.setLevelBC(0, &phi);
	amrex::Real final_resid = mlmg.solve(phi_levels, rhs_levels, reltol, abstol);
	amrex::Print() << "[Final solve] residual max norm = " << final_resid << "\n\n";

	return 0;
}
