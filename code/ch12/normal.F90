module rayTracing
#ifdef _CUDA
  use rgbCUDA
#else
  use rgbHost
#endif
  
  real, parameter :: lowerLeftCorner(3) = [-2.0, -1.0, -1.0]
  real, parameter :: horizontal(3) = [4.0, 0.0, 0.0]
  real, parameter :: vertical(3) = [0.0, 2.0, 0.0]
  real, parameter :: origin(3) = [0.0, 0.0, 0.0]

  type ray
     real :: origin(3)
     real :: dir(3)
  end type ray

  interface ray
     module procedure rayConstructor
  end interface ray

  type sphere
     real :: center(3), radius
  end type sphere

contains
  
  !@cuf attributes(device) &
  function normalize(a) result(res)
    implicit none
    real :: a(3), res(3)

    res = a/sqrt(sum(a**2))
  end function normalize

  !@cuf attributes(device) &
  function rayConstructor(origin, dir) result(r)
    implicit none
    !dir$ ignore_tkr (d) origin, (d) dir   
    real :: origin(3), dir(3)
    type(ray) :: r
    r%origin = origin
    r%dir = normalize(dir)
  end function rayConstructor
  
  !@cuf attributes(device) &
  function hitSphere(s, r) result(res)
    implicit none
    type(sphere) :: s
    type(ray) :: r
    real :: oc(3), a, b, c, disc
    real :: res

    oc = r%origin - s%center
    a = dot_product(r%dir, r%dir)
    b = 2.0*dot_product(r%dir, oc)
    c = dot_product(oc, oc) - s%radius**2
    disc = b**2 - 4.0*a*c
    if (disc > 0.0) then
       res = (-b -sqrt(disc))/(2.0*a)
    else
       res = -1.0
    endif
  end function hitSphere

  !@cuf attributes(device) &
  function color(r, s) result(res)
    implicit none
    type(ray) :: r
    type(sphere) :: s
    type(rgb) :: res
    real :: t, n(3)

    t = hitSphere(s, r)
    if (t > 0) then
       n = r%dir*t + r%origin - s%center
       n = normalize(n)
       res = rgb(0.5*(n+1.0))
    else
       t = 0.5*(r%dir(2) + 1.0)  
       res = rgb((1.0-t)*[1.0, 1.0, 1.0] + t*[0.5, 0.7, 1.0])
    endif
  end function color

#ifdef _CUDA  
  attributes(global) subroutine renderKernel(fb, nx, ny, s)
    implicit none
    type(rgb) :: fb(nx,ny)
    integer, value :: nx, ny
    type(sphere) :: s
    type(ray) :: r
    real :: dir(3)
    real :: u, v
    integer :: i, j

    i = threadIdx%x + (blockIdx%x-1)*blockDim%x
    j = threadIdx%y + (blockIdx%y-1)*blockDim%y
    if (i <= nx .and. j <= ny) then
       u = real(i)/nx
       v = real(j)/ny
       dir = lowerLeftCorner + u*horizontal + v*vertical - origin
       r = ray(origin, dir)
       fb(i,j) = color(r, s)
    end if
  end subroutine renderKernel
#endif
    
  subroutine render(fb, s)
    !@cuf use cudafor
    implicit none
    type(rgb) :: fb(:,:)
    type(sphere) :: s
    type(ray) :: r
    real :: u, v
    integer :: nx, ny, i, j
    
    nx = size(fb,1)    
    ny = size(fb,2)

#ifdef _CUDA
    block
      type(rgb), device, allocatable :: fb_d(:,:)
      type(sphere), device :: s_d
      type(dim3) :: tBlock, grid

      allocate(fb_d(nx,ny))
      s_d = s
      tBlock = dim3(32,8,1)
      grid = dim3((nx-1)/tBlock%x+1, (ny-1)/tBlock%y+1, 1)
      call renderKernel<<<grid, tBlock>>>(fb_d, nx, ny, s_d)
      fb = fb_d
      deallocate(fb_d)
    end block
#else    
    do j = 1, ny
       do i = 1, nx                
          u = real(i)/nx
          v = real(j)/ny
          r = ray(origin, &
               lowerLeftCorner + u*horizontal + v*vertical - origin)
          fb(i,j) = color(r, s)
       end do
    end do
#endif
  end subroutine render
  
end module rayTracing

program main
  use rayTracing
  implicit none
  integer, parameter :: nx = 400, ny = 200
  integer :: i, j
  type(rgb) :: fb(nx,ny)
  type(sphere) :: s

  s = sphere([0.0, 0.0, -1.0], 0.5) 
  call render(fb, s)
  
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
