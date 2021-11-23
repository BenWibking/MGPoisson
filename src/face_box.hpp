#ifndef FACE_BOX_HPP_
#define FACE_BOX_HPP_

#include <AMReX.H>
#include "AMReX_MultiFab.H"

auto getFaceBoxes(amrex::Geometry geom, amrex::MultiFab &phi)
    -> amrex::Vector<std::tuple<amrex::Array4<amrex::Real>, amrex::Box, amrex::Orientation>>;

#endif // FACE_BOX_HPP_