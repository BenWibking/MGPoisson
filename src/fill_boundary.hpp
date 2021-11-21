#include "AMReX.H"
#include "AMReX_Array.H"
#include "AMReX_Config.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_INT.H"
#include "AMReX_IntVect.H"
#include "AMReX_Orientation.H"

template <typename F> void fillBoundaryCells(amrex::Geometry geom, amrex::MultiFab &phi, F &&user_f)
{
	// create a domain box containing valid + periodic cells following AMReX_PhysBCFunct.H
	// (used to check if a fabbox is at the domain boundary, i.e. if gdomain.contains(fabbox) ==
	// false)
	amrex::Box pdomain = amrex::convert(geom.Domain(), phi.boxArray().ixType());
	for (int i = 0; i < AMREX_SPACEDIM; ++i) {
		if (geom.isPeriodic(i)) {
			pdomain.grow(i, phi.n_grow[i]);
		}
	}

	for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
		amrex::FArrayBox &dest = phi[mfi];
		amrex::Array4<amrex::Real> const &arr = dest.array();

		const amrex::Box &bx = mfi.fabbox();
		const amrex::IntVect &len = phi.n_grow;

		if (!pdomain.contains(bx)) { // check if box overlaps domain boundary
			amrex::Array<amrex::Box, 2 *AMREX_SPACEDIM> domain_face_boxes = {
			    AMREX_D_DECL(amrex::adjCellLo(pdomain, 0, len[0]),
					 amrex::adjCellLo(pdomain, 1, len[1]),
					 amrex::adjCellLo(pdomain, 2, len[2])),
			    AMREX_D_DECL(amrex::adjCellHi(pdomain, 0, len[0]),
					 amrex::adjCellHi(pdomain, 1, len[1]),
					 amrex::adjCellHi(pdomain, 2, len[2]))};

			amrex::Array<amrex::Orientation, 2 *AMREX_SPACEDIM> domain_faces = {
			    AMREX_D_DECL(amrex::Orientation(0, amrex::Orientation::low),
					 amrex::Orientation(1, amrex::Orientation::low),
					 amrex::Orientation(2, amrex::Orientation::low)),
			    AMREX_D_DECL(amrex::Orientation(0, amrex::Orientation::high),
					 amrex::Orientation(1, amrex::Orientation::high),
					 amrex::Orientation(2, amrex::Orientation::high))};

			amrex::Vector<std::pair<amrex::Box, amrex::Orientation>> face_boxes;
			for (int i = 0; i < domain_faces.size(); ++i) {
				amrex::Box &b = domain_face_boxes[i];
				amrex::Orientation &o = domain_faces[i];
				amrex::Box tmp = b & bx;
				if (tmp.ok()) {
					face_boxes.push_back(std::make_pair(tmp, o));
				}
			}

			for (int n = 0; n < face_boxes.size(); ++n) {
				auto [facebox, orientation] = face_boxes[n];
				amrex::ParallelFor(
				    facebox, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
					    user_f(orientation, arr, i, j, k);
				    });
			}
		}
	}
}