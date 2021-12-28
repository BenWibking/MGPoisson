#ifndef MULTIPOLE_HPP_
#define MULTIPOLE_HPP_

#include "AMReX_REAL.H"
#include <AMReX.H>

using Real = amrex::Real;

class Multipole
{
      public:
	Real r0[AMREX_SPACEDIM]{}; // center of expansion

	Real q0{};				   // monopole
	Real q1[AMREX_SPACEDIM]{};		   // dipole
	Real q2[AMREX_SPACEDIM][AMREX_SPACEDIM]{}; // quadrupole
};

#endif // MULTIPOLE_HPP_