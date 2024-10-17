module precision_m
  use, intrinsic :: iso_fortran_env

#ifdef DOUBLE
  integer, parameter :: fp_kind = real64
#else
  integer, parameter :: fp_kind = real32
#endif
end module precision_m

