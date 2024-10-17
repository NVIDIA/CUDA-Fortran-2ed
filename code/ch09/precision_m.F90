module precision_m
  use, intrinsic :: iso_fortran_env
  use iso_fortran_env, only: real32, real64
  integer, parameter :: singlePrecision = real32
  integer, parameter :: doublePrecision = real64
  
#ifdef DOUBLE
  integer, parameter :: fp_kind = doublePrecision
#else
  integer, parameter :: fp_kind = singlePrecision
#endif

end module precision_m

