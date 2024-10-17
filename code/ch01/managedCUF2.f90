program managedCUF
  !@cuf use cudafor
  implicit none
  integer, parameter :: nx=1024, ny=512
  integer :: a(nx,ny)
  !@cuf	attributes(managed):: a	
  integer :: b, i ,j
    
  a = 1
  b = 3 
  !$cuf kernel do (2) <<<*,*>>>
  do j=1,ny
     do i=1,nx
        a(i,j)=a(i,j)+b
     end do
  end do
  !@cuf	i=cudaDeviceSynchronize()

  !@cuf print *, "Running CUDA version ..."
  if(any(a /= 4)) then
     print *, "**** Program Failed ****"
  else
     print *, "Program Passed"
  end if
end program managedCUF
