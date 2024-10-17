program main
  implicit none
  integer, parameter :: n=8
  real :: a(n), b(n)
  !@cuf attributes(device) :: a, b
  !@cuf real :: a_h(n), b_h(n)
  integer :: i

  !$cuf kernel do <<<*,*>>> 
  do i = 1, n  
     a(i) = i+1
  enddo

  !$cuf kernel do <<<*,*>>>
  do i = 1, n
     b(i) = a(i)+1
  enddo

  !@cuf a_h = a; b_h = b
  !@cuf print *, 'GPU version'

#ifdef _CUDA
  print *, a_h
  print *, b_h
#else
  print *, a
  print *, b
#endif

end program main


