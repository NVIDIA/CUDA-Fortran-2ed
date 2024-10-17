module rayTracing
#ifdef _CUDA
  use rgbCUDA
#else
  use rgbHost
#endif

  ! anti-aliasing parameter
  integer :: nRaysPerPixel
  !@cuf attributes(managed) :: nRaysPerPixel

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

  type environs
     integer :: nSpheres
     type(sphere), &
          !@cuf managed, &
          allocatable :: spheres(:)
  end type environs

  type hitRecord
     real :: t
     real :: p(3)
     real :: normal(3)
  end type hitRecord

  type camera
     real :: lowerLeftCorner(3)
     real :: horizontal(3)
     real :: vertical(3)
     real :: origin(3)
  end type camera
  
contains

  
  subroutine environsAlloc(env)
    type(environs) :: env
    !@cuf attributes(managed) :: env
    allocate(env%spheres(env%nSpheres))
  end subroutine environsAlloc

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
  function hitSphere(s, r, tmin, tmax, rec) result(res)
    implicit none
    type(sphere) :: s
    type(ray) :: r
    real :: tmin, tmax
    type(hitRecord) :: rec
    logical :: res
    
    real :: oc(3), a, b, c, disc, t

    oc = r%origin - s%center
    a = dot_product(r%dir, r%dir)
    b = 2.0*dot_product(r%dir, oc)
    c = dot_product(oc, oc) - s%radius**2
    disc = b**2 - 4.0*a*c
    if (disc >= 0.0) then
       t = (-b -sqrt(disc))/(2.0*a)
       if (t > tmin .and. t < tmax) then
          rec%t = t
          rec%p = r%dir*t + r%origin
          rec%normal = (rec%p - s%center)/s%radius
          res = .true.
          return
       end if
       t = (-b +sqrt(disc))/(2.0*a)
       if (t > tmin .and. t < tmax) then
          rec%t = t
          rec%p = r%dir*t + r%origin
          rec%normal = (rec%p - s%center)/s%radius
          res = .true.
          return
       end if
    end if
    res = .false.
  end function hitSphere

  !@cuf attributes(device) &
  function hitEnvirons(env, r, tmin, tmax, rec) result(res)
    implicit none
    type(environs) :: env
    type(ray) :: r
    real :: tmin, tmax
    type(hitRecord) :: rec
    logical :: res
    
    type(hitRecord) :: recl
    integer :: i
    real :: closest

    closest = tmax
    res = .false.

    do i = 1, env%nSpheres
       if (hitSphere(env%spheres(i), r, tmin, closest, recl)) then
          res = .true.
          rec = recl
          closest = recl%t
       end if
    end do
  end function hitEnvirons

  !@cuf attributes(device) &
  function color(r, env) result(res)
    implicit none    
    type(ray) :: r
    type(environs) :: env
    type(rgb):: res

    type(hitRecord) :: rec
    real :: t, n(3)
    
    if (hitEnvirons(env, r, 0.0, huge(t), rec)) then
       n = normalize(rec%normal)
       res = rgb(0.5*(n+1.0))
    else
       t = 0.5*(r%dir(2) + 1.0)  
       res = rgb((1.0-t)*[1.0, 1.0, 1.0] + t*[0.5, 0.7, 1.0])
    endif
  end function color

#ifdef _CUDA  
  attributes(global) subroutine renderKernel(fb, nx, ny, env, cam)
    use curand_device
    implicit none
    type(rgb) :: fb(nx,ny)
    integer, value :: nx, ny
    type(environs) :: env
    type(camera) :: cam
    
    type(ray) :: r
    real :: dir(3)
    real :: u, v
    integer :: i, j

    integer :: ir
    type(rgb) :: c  

    type(curandStateXORWOW) :: h
    integer(8) :: seed, seq, offset
    
    i = threadIdx%x + (blockIdx%x-1)*blockDim%x
    j = threadIdx%y + (blockIdx%y-1)*blockDim%y

    if (i <= nx .and. j <= ny) then

       ! initialize RNG
       seed = nx*(j-1) + i + 4321
       seq = 0
       offset = 0
       call curand_init(seed, seq, offset, h)
    
       c = [0.0, 0.0, 0.0]
       do ir = 1, nRaysPerPixel
          u = (real(i-1)+curand_uniform(h))/nx
          v = (real(j-1)+curand_uniform(h))/ny
          dir = cam%lowerLeftCorner + &
               u*cam%horizontal + v*cam%vertical - cam%origin
          r = ray(cam%origin, dir)
          c = c + color(r, env)
       end do
       fb(i,j) = c/nRaysPerPixel
    end if
  end subroutine renderKernel
#endif
    
  subroutine render(fb, env, cam)
    !@cuf use cudafor
    implicit none
    type(rgb) :: fb(:,:)
    type(environs) :: env
    !@cuf attributes(managed) :: env
    type(camera) :: cam
    !@cuf attributes(managed) :: cam

    type(ray) :: r
    real :: u, v
    integer :: nx, ny, i, j

    integer :: ir
    real :: rn(2)
    type(rgb) :: c
    
    nx = size(fb,1)    
    ny = size(fb,2)

#ifdef _CUDA
    block
      type(rgb), device, allocatable :: fb_d(:,:)
      type(dim3) :: tBlock, grid
      
      allocate(fb_d(nx,ny))
      tBlock = dim3(32,8,1)
      grid = dim3((nx-1)/tBlock%x+1, (ny-1)/tBlock%y+1, 1)
      call renderKernel<<<grid, tBlock>>>(fb_d, nx, ny, env, cam)
      fb = fb_d
      deallocate(fb_d)
    end block
#else    
    do j = 1, ny
       do i = 1, nx
          c = [0.0, 0.0, 0.0]
          do ir = 1, nRaysPerPixel
             call random_number(rn)
             u = (real(i-1) + rn(1))/nx
             v = (real(j-1) + rn(2))/ny
             r = ray(cam%origin, cam%lowerLeftCorner + &
                  u*cam%horizontal + v*cam%vertical - cam%origin)
             c = c + color(r, env)
          end do
          fb(i,j) = c/nRaysPerPixel
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
  type(environs) :: env
  !@cuf attributes(managed) :: env
  type(camera) :: cam
  !@cuf attributes(managed) :: cam
  
  ! setup environment
  env%nSpheres = 2
  call environsAlloc(env)  
  env%spheres(1) = sphere([0.0, 0.0, -1.0], 0.5)
  env%spheres(2) = sphere([0.0, -100.5, -1.0], 100.0)
  
  ! setup camera
  cam%lowerLeftCorner = [-2.0, -1.0, -1.0]
  cam%horizontal = [4.0, 0.0, 0.0]
  cam%vertical = [0.0, 2.0, 0.0]
  cam%origin = [0.0, 0.0, 0.0]

  ! antialiasing parameter
  nRaysPerPixel = 100
  
  call render(fb, env, cam)
  
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
