module rayTracing
#ifdef _CUDA
  use rgbCUDA
  use curand_device
  use cudafor
#else
  use rgbHost
#endif
  
#ifdef _CUDA
#define c_hostDevPtr c_devptr
#define c_hostDevLoc c_devloc
#else
#define c_hostDevPtr c_ptr
#define c_hostDevLoc c_loc
#endif
  
  use iso_c_binding

  ! anti-aliasing parameter
  integer :: nRaysPerPixel
  !@cuf attributes(managed) :: nRaysPerPixel

  !@cuf type(curandStateXORWOW), device, allocatable :: randState_d(:,:)

  
  type ray
     real :: origin(3)
     real :: dir(3)
  end type ray
  
  interface ray
     module procedure rayConstructor
  end interface ray

  ! material types

  enum, bind(c)
     enumerator :: lambMat = 1, metalMat = 2
  end enum
  
  type lambertian
     type(rgb) :: albedo
  end type lambertian

  type metal
     type(rgb) :: albedo
     real :: fuzz
  end type metal
  
  ! objects

  type sphere
     real :: center(3), radius
     integer :: materialType
     type(c_hostDevPtr) :: matPtr
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
     integer :: materialType
     type(c_hostDevPtr) :: matPtr
  end type hitRecord

  type camera
     real :: lowerLeftCorner(3)
     real :: horizontal(3)
     real :: vertical(3)
     real :: origin(3)
  end type camera

  interface randomNumber
     module procedure :: rngScalar, rngArray
  end interface randomNumber

contains

#ifdef _CUDA
  attributes(device) function rngScalar() result(res)
    implicit none
    real :: res
    integer :: i, j
    type(curandStateXORWOW) :: h
    i = (blockIdx%x-1)*blockDim%x + threadIdx%x
    j = (blockIdx%y-1)*blockDim%y + threadIdx%y
    h = randState_d(i,j)
    res = curand_uniform(h)
    randState_d(i,j) = h
  end function rngScalar

  attributes(device) function rngArray(n) result(res)
    implicit none
    integer, intent(in) :: n
    real :: res(n)
    integer :: i, j, k
    type(curandStateXORWOW) :: h
    i = (blockIdx%x-1)*blockDim%x + threadIdx%x
    j = (blockIdx%y-1)*blockDim%y + threadIdx%y
    h = randState_d(i,j)
    do k = 1, n
       res(k) = curand_uniform(h)
    end do
    randState_d(i,j) = h
  end function rngArray
#else
  function rngScalar() result(res)
    real :: res
    call random_number(res)
  end function rngScalar

  function rngArray(n) result(res)
    integer, intent(in) :: n
    real :: res(n)
    call random_number(res)
  end function rngArray
#endif

  !@cuf attributes(device) &
  function randomPointInUnitSphere() result(res)
    implicit none
    real :: res(3)
    do
       res = randomNumber(3)
       res = 2*res - 1.0
       if (sum(res**2) <= 1.0) exit
    enddo
  end function randomPointInUnitSphere
  
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
  function reflect(rin, n) result(res)
    implicit none
    type(ray) :: rin
    real :: n(3)
    real :: res(3)
    res = rin%dir - 2.0*dot_product(rin%dir,n)*n
  end function reflect

  
  !@cuf attributes(device) &
  function scatter(rin, rec, attenuation, rscattered) result(res)
    implicit none
    type(ray) :: rin
    type(hitRecord) :: rec
    type(rgb) :: attenuation
    type(ray) :: rscattered
    logical :: res

    type(lambertian) :: lambPtr; pointer(lambCrayP, lambPtr)
    type(metal) :: metalPtr;     pointer(metalCrayP, metalPtr)
    real :: n(3), dir(3)

    if (rec%materialType == lambMat) then
       n = normalize(rec%normal)
       dir = n + randomPointInUnitSphere()
       rscattered = ray(rec%p, dir)
       lambCrayP = transfer(rec%matPtr, lambCrayP)
       attenuation = lambPtr%albedo
       res = .true.
    else if (rec%materialType == metalMat) then
       n = normalize(rec%normal)
       metalCrayP = transfer(rec%matPtr, metalCrayP)
       dir = reflect(rin, n) + metalPtr%fuzz*randomPointInUnitSphere()
       rscattered = ray(rec%p, dir)
       attenuation = metalPtr%albedo
       res = (dot_product(rscattered%dir, n) > 0)
    end if
        
  end function scatter

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
          rec%materialType = s%materialType          
          rec%matPtr = s%matPtr
          res = .true.
          return
       end if
       t = (-b +sqrt(disc))/(2.0*a)
       if (t > tmin .and. t < tmax) then
          rec%t = t
          rec%p = r%dir*t + r%origin
          rec%normal = (rec%p - s%center)/s%radius
          rec%materialType = s%materialType
          rec%matPtr = s%matPtr
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
    integer, parameter :: maxDepth = 50
    real, parameter :: shadowAcne = 0.001
    type(ray) :: r
    type(environs) :: env
    type(rgb) :: res

    type(ray) :: lRay, scRay
    type(hitRecord) :: rec
    real :: t, n(3)
    type(rgb) :: attenuation, scAtten
    integer :: depth
    
    attenuation = [1.0, 1.0, 1.0]
    lRay = r

    res = [0.0, 0.0, 0.0]
    do depth = 1, maxDepth
       if (hitEnvirons(env, lRay, shadowAcne, huge(t), rec)) then
          if (scatter(lray, rec, scAtten, scRay)) then
             attenuation = scAtten*attenuation
             lRay = scRay
          else
             res = [0.0, 0.0, 0.0]
             return
          endif
       else
          t = 0.5*(lray%dir(2) + 1.0)  
          res = rgb((1.0-t)*[1.0, 1.0, 1.0] + t*[0.5, 0.7, 1.0])
          res = res * attenuation
          return
       endif
    end do
  end function color

#ifdef _CUDA  
  attributes(global) subroutine renderKernel(fb, nx, ny, env, cam)

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
       call curand_init(seed, seq, offset, randState_d(i,j))
    
       c = [0.0, 0.0, 0.0]
       do ir = 1, nRaysPerPixel
          u = (real(i-1)+randomNumber())/nx
          v = (real(j-1)+randomNumber())/ny
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


  type(lambertian)  :: lambArr(2)
  !@cuf attributes(device) :: lambArr
  type(metal) :: metalArr(2)
  !@cuf attributes(device) :: metalArr
  type(c_hostDevPtr) :: cPtr
  
  ! setup environment
  env%nSpheres = 4
  call environsAlloc(env)

  lambArr(1) = lambertian(rgb([0.8, 0.3, 0.3]))
  cPtr = c_hostDevLoc(lambArr(1))
  env%spheres(1) = sphere([0.0, 0.0, -1.0], 0.5, lambMat, cPtr)

  lambArr(2) = lambertian(rgb([0.8, 0.8, 0.0]))
  cPtr = c_hostDevLoc(lambArr(2))
  env%spheres(2) = sphere([0.0, -100.5, -1.0], 100.0, lambMat, cPtr)

  metalArr(1) = metal(rgb([0.8, 0.6, 0.2]), 1.0)
  cPtr = c_hostDevLoc(metalArr(1))
  env%spheres(3) = sphere([1.0, 0.0, -1.0], 0.5, metalMat, cPtr)
  
  metalArr(2) = metal(rgb([0.8, 0.8, 0.8]), 0.3)
  cPtr = c_hostDevLoc(metalArr(2))
  env%spheres(4) = sphere([-1.0, 0.0, -1.0], 0.5, metalMat, cPtr)
  
  ! setup camera
  cam%lowerLeftCorner = [-2.0, -1.0, -1.0]
  cam%horizontal = [4.0, 0.0, 0.0]
  cam%vertical = [0.0, 2.0, 0.0]
  cam%origin = [0.0, 0.0, 0.0]

  ! antialiasing parameter
  nRaysPerPixel = 100
  
  ! allocate RNG state array, initialization done in kernel
  !@cuf allocate(randState_d(nx,ny))
  
  call render(fb, env, cam)
  
  ! ppm output

  print "(a2)", 'P3'   ! indicates RGB colors in ASCII, must be flush left
  print *, nx, ny      ! width and height of image
  print *, 255         ! maximum value for each color
  do j = ny, 1, -1
     do i = 1, nx                
        print "(3(1x,i3))", int(255*sqrt(fb(i,j)%v))
     end do
  end do

end program main
