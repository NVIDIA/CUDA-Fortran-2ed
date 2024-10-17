module timing
  interface wallclock
     function wallclock() result(res) bind(C, name='wallclock')
       use iso_c_binding
       real (c_double) :: res
     end function wallclock
  end interface wallclock
end module timing

