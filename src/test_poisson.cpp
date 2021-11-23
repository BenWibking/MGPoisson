//==============================================================================
// AMRPoisson
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file test_poisson.cpp
/// \brief Defines a test problem for a cell-centered Poisson solve.
///

#include "AMReX_Array.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Geometry.H"
#include "AMReX_IntVect.H"
#include "AMReX_MLMG.H"
#include "AMReX_MLPoisson.H"
#include "AMReX_MultiFab.H"
#include <AMReX.H>

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
	const int n_cell = 128;
	const int max_grid_size = 128;
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

	// set density field to a Fourier mode of the box
	// (N.B. when periodic, very slow convergence when kx=ky=1...)
	const int kx = 1;
	const int ky = 1;

	auto prob_lo = geom.ProbLoArray();
	auto dx = geom.CellSizeArray();
	for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto rho = rhs.array(mfi);
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			amrex::Real const x = prob_lo[0] + (i + amrex::Real(0.5)) * dx[0];
			amrex::Real const y = prob_lo[1] + (j + amrex::Real(0.5)) * dx[1];
			rho(i, j, k) =
			    FourPiG * std::sin(2.0 * M_PI * kx * x) * std::sin(2.0 * M_PI * ky * y);
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
	const amrex::Real reltol = 1.0e-13; // doesn't work below ~1e-13...
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

	// Step 2c. MPI_Allgather multipoles. Now each process has multipoles for all N faceBoxes.

	// Step 3. Compute the potential at each cell in M local faceBoxes, using multipoles of all
	// N faceBoxes. This step has unavoidable complexity O(M*N).

	// (N.B. Burkhart [1997] gives an asymptotic expansion of the DGF in (1/r)^n, up to n=5.
	// Alternatively, one can place a unit charge in the center of the box, apply Burkhart's
	// expansion to compute the Dirichlet boundary conditions, and solve numerically for the DGF
	// with multigrid. This can be done with a small box, e.g. 32^3, since the expansion
	// rapidly converges as r >> dx.)

	// Step 4. Solve for the potential with Dirichlet boundary conditions given by
	// the potential computed in step 3.

	return 0;
}
