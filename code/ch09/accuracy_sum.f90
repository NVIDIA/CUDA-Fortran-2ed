program sum_accuracy
  implicit none
  real, allocatable :: x(:)
  real :: sum_intrinsic,sum_cpu, sum_kahan, sum_pairwise, &
       comp, y, tmp
  double precision :: sum_cpu_dp
  integer :: i,inext,icurrent,  N=10000000

  allocate (x(N))
  x=7.

  ! Summation using intrinsic
  sum_intrinsic=sum(x)

  ! Recursive summation 
  sum_cpu=0.
  sum_cpu_dp=0.d0
  do i=1,N
     ! accumulator in single precision
     sum_cpu=sum_cpu+x(i)
     ! accumulator in double precision
     sum_cpu_dp=sum_cpu_dp+x(i)
  end do

  ! Kahan summation
  sum_kahan=0.
  comp=0. ! running compensation to recover lost low-order bits
  
  do i=1,N
     y    = comp +x(i)
     tmp  = sum_kahan + y     ! low-order bits may be lost
     comp = (sum_kahan-tmp)+y ! (sum-tmp) recover low-order bits
     sum_kahan = tmp
  end do
  sum_kahan=sum_kahan +comp
  
  ! Pairwise summation
  icurrent=N
  inext=ceiling(real(N)/2)
  do while (inext >1)
     do i=1,inext
        if ( 2*i <= icurrent) x(i)=x(i)+x(i+inext)
     end do
     icurrent=inext
     inext=ceiling(real(inext)/2)
  end do
  sum_pairwise=x(1)+x(2)
  
  write(*, "('Summming ',i10, &
       ' elements of magnitude ',f3.1)") N,7.
  write(*, "('Sum with intrinsic function       =',f12.1, &
       '   Error=', f12.1)")  &
       sum_intrinsic, 7.*N-sum_intrinsic
  write(*, "('Recursive sum with SP accumulator =',f12.1, &
       '   Error=', f12.1)")  sum_cpu, 7.*N-sum_cpu
  write(*, "('Recursive sum with DP accumulator =',f12.1, &
       '   Error=', f12.1)")  sum_cpu_dp, 7.*N-sum_cpu_dp
  write(*, "('Pairwise sum in SP                =',f12.1, &
       '   Error=', f12.1)")  sum_pairwise, 7.*N-sum_pairwise
  write(*, "('Compensated sum in SP             =',f12.1, &
       '   Error=', f12.1)")  sum_kahan, 7.*N-sum_kahan
  
  deallocate(x)
end program sum_accuracy
