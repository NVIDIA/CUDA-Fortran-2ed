program main
  implicit none
  integer, parameter :: n=8
  
  real :: a(n), b(n)
  !@cuf  real, device :: a_d(n), b_d(n)
  integer :: i

  !@cuf associate(a => a_d)
  !$cuf kernel do
  do i = 1, n
     a(i) = i+1
  enddo
  !@cuf associate(b => b_d) 
  !$cuf kernel do
  do i = 1, n
     b(i) = a(i)+1
  enddo
  !@cuf end associate ! b ...
  !@cuf end associate ! a ...
  !@cuf a = a_d; b = b_d

  !@cuf print *, 'GPU run'
  print *, a
  print *, b
end program main


