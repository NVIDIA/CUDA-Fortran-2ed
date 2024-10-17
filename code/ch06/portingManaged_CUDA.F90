program main
#ifdef _CUDA  
  use cudafor
#endif
  implicit none
  integer, parameter :: n=8
  real :: a(n), b(n)
#ifdef _CUDA
  attributes(managed) :: a, b
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
  i = cudaDeviceSynchronize()
  print *, 'GPU version'
#endif
  
  print *, a
  print *, b
end program main


