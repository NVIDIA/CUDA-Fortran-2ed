program main
  implicit none
  integer, parameter :: n=8
  real :: a(n), b(n)
  integer :: i

  do i = 1, n
     a(i) = i+1
  enddo

  do i = 1, n
     b(i) = a(i)+1
  enddo

  print *, a
  print *, b
end program main


