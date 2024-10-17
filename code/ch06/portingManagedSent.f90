program main
  !@cuf use cudafor
  implicit none
  integer, parameter :: n=8
  real :: a(n), b(n)
  !@cuf attributes(managed) :: a, b
  integer :: i

  !$cuf kernel do <<<*,*>>> 
  do i = 1, n
     a(i) = i+1
  enddo

  !$cuf kernel do <<<*,*>>>
  do i = 1, n
     b(i) = a(i)+1
  enddo

  !@cuf i = cudaDeviceSynchronize()
  !@cuf print *, 'GPU version'
  
  print *, a
  print *, b
end program main


