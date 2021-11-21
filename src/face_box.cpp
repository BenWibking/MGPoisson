//==============================================================================
// AMRPoisson
// Copyright 2021 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file face_box.cpp
/// \brief Compute the intersection of all boxes with the faces of the domain.
///

#include <AMReX.H>
#include "AMReX_Geometry.H"
#include "AMReX_MultiFab.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_Orientation.H"

#include "face_box.hpp"

auto getFaceBoxes(amrex::Geometry geom, amrex::MultiFab &phi)
    -> amrex::Vector<std::tuple<amrex::Array4<amrex::Real>, amrex::Box, amrex::Orientation>>
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

	const amrex::IntVect &len = phi.n_grow;
	const amrex::Array<amrex::Box const, 2 *AMREX_SPACEDIM> domain_face_boxes = {
	    AMREX_D_DECL(amrex::adjCellLo(pdomain, 0, len[0]), amrex::adjCellLo(pdomain, 1, len[1]),
			 amrex::adjCellLo(pdomain, 2, len[2])),
	    AMREX_D_DECL(amrex::adjCellHi(pdomain, 0, len[0]), amrex::adjCellHi(pdomain, 1, len[1]),
			 amrex::adjCellHi(pdomain, 2, len[2]))};

	const amrex::Array<amrex::Orientation const, 2 *AMREX_SPACEDIM> domain_faces = {
	    AMREX_D_DECL(amrex::Orientation(0, amrex::Orientation::low),
			 amrex::Orientation(1, amrex::Orientation::low),
			 amrex::Orientation(2, amrex::Orientation::low)),
	    AMREX_D_DECL(amrex::Orientation(0, amrex::Orientation::high),
			 amrex::Orientation(1, amrex::Orientation::high),
			 amrex::Orientation(2, amrex::Orientation::high))};

	amrex::Vector<std::tuple<amrex::Array4<amrex::Real>, amrex::Box, amrex::Orientation>>
	    face_boxes;

	for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
		auto arr = phi[mfi].array();
		const amrex::Box &bx = mfi.fabbox();

		if (!pdomain.contains(bx)) { // check if box overlaps domain boundary
			for (int i = 0; i < domain_faces.size(); ++i) {
				amrex::Box const &b = domain_face_boxes[i];
				amrex::Orientation const &o = domain_faces[i];
				amrex::Box const tmp = b & bx;
				if (tmp.ok()) {
					face_boxes.push_back(std::make_tuple(arr, tmp, o));
				}
			}
		}
	}

	return face_boxes; // this is correct due to C++11 move semantics
}