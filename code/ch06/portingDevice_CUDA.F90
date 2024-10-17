program main
  implicit none
  integer, parameter :: n=8
  real :: a(n), b(n)
#ifdef _CUDA
  attributes(device) :: a, b
  real :: a_h(n), b_h(n)
#endif
  integer :: i

  !$cuf kernel do <<<*,*>>> 
  do i = 1, n  
     a(i) = i+1
  enddo

  !$cuf kernel do <<<*,*>>>
  do i = 1, n
     b(i) = a(i)+1
  enddo

#ifdef _CUDA
  a_h = a; b_h = b
  print *, 'GPU version'
  print *, a_h
  print *, b_h
#else
  print *, a
  print *, b
#endif

end program main


