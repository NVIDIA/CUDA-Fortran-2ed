program main
  !@cuf use cudafor
  implicit none
  integer, parameter :: nx = 400, ny = 200
  integer :: i, j
  type rgb
     real :: v(3)
  end type rgb
  type(rgb) :: fb(nx, ny)
  !@cuf type(rgb), device :: fb_d(nx, ny)
  
  !@cuf associate (fb => fb_d)
  !$cuf kernel do (2) <<<*,*>>>
  do j = 1, ny
     do i = 1, nx                
        fb(i,j)%v(1) = real(i)/nx
        fb(i,j)%v(2) = real(j)/ny
        fb(i,j)%v(3) = 0.2
     end do
  end do
  !@cuf end associate

  !@cuf fb = fb_d
  
  ! ppm output

  print "(a2)", 'P3'   ! indicates RGB colors in ASCII, must be flush left
  print *, nx, ny      ! width and height of image
  print *, 255         ! maximum value for each color

  do j = ny, 1, -1
     do i = 1, nx                
        print "(3(1x,i3))", int(255*fb(i,j)%v)
     end do
  end do

end program main
