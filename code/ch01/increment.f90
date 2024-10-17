module m
contains
  subroutine increment(a, b)
    implicit none
    integer, intent(inout) :: a(:)
    integer, intent(in) :: b
    integer :: i, n

    n = size(a)
    do i = 1, n
       a(i) = a(i)+b
    enddo

  end subroutine increment
end module m


program incrementCPU
  use m
  implicit none
  integer, parameter :: n = 256
  integer :: a(n), b

  a = 1
  b = 3
  call increment(a, b)

  if (any(a /= 4)) then
     print *, '**** Program Failed ****'
  else
     print *, 'Program Passed'
  endif
end program incrementCPU
