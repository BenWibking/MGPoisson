#ifndef MULTIPOLE_HPP_
#define MULTIPOLE_HPP_

#include "AMReX_REAL.H"
#include <AMReX.H>

class Multipole
{
      public:
	amrex::Real r0[AMREX_SPACEDIM]{};    // center of expansion
    
	amrex::Real q0{};				                    // monopole
	amrex::Real q1[AMREX_SPACEDIM]{};		            // dipole
	amrex::Real q2[AMREX_SPACEDIM][AMREX_SPACEDIM]{};   // quadrupole
};

#endif // MULTIPOLE_HPP_