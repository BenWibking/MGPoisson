#ifndef MULTIPOLE_HPP_
#define MULTIPOLE_HPP_

#include "AMReX_REAL.H"
#include <AMReX.H>

class Multipole
{
      public:
	amrex::Real q0 = 0;
	amrex::Real q1[AMREX_SPACEDIM] = {};
	amrex::Real q2[AMREX_SPACEDIM][AMREX_SPACEDIM] = {};
};

#endif // MULTIPOLE_HPP_