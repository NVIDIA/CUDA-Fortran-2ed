module &
#ifdef _CUDA
  rgbCUDA
#else
  rgbHost
#endif

  type rgb
     real :: v(3)
  end type rgb

  interface assignment (=)
     module procedure rgbEqR3, r3EqRgb
  end interface assignment (=)

  interface operator(*)
     module procedure rgbTimesR3, r3TimesRgb, rgbTimesRgb, &
          rgbTimesR, rTimesRgb
  end interface operator(*)

  interface operator(/)
     module procedure rgbDivR3, r3DivRgb, rgbDivRgb, rgbDivI
  end interface operator(/)

  interface operator(+)
     module procedure rgbPlusR3, r3PlusRgb, rgbPlusRgb
  end interface operator(+)

  interface operator(-)
     module procedure rgbMinusR3, r3MinusRgb, rgbMinusRgb
  end interface operator(-)

contains

  !@cuf attributes(device) &
  subroutine rgbEqR3(rgbout, rin)
    type(rgb), intent(out) :: rgbout
    real(4), intent(in) :: rin(3)
    rgbout%v = rin
  end subroutine rgbEqR3
  
  !@cuf attributes(device) &
  subroutine r3EqRgb(rout, rgbin)
    real(4), intent(out) :: rout(3)
    type(rgb), intent(in) :: rgbin
    rout = rgbin%v
  end subroutine r3EqRgb

  !@cuf attributes(device) &
  function rgbTimesR3(rgbin, rin) result(res)
    type(rgb), intent(in) :: rgbin
    real(4), intent(in) :: rin(3)
    real(4) :: res(3)
    res = rgbin%v * rin
  end function rgbTimesR3

  !@cuf attributes(device) &
  function r3TimesRgb(rin, rgbin) result(res)
    real(4), intent(in) :: rin(3)
    type(rgb), intent(in) :: rgbin
    real(4) :: res(3)
    res = rgbin%v * rin
  end function r3TimesRgb

  !@cuf attributes(device) &
  function rgbTimesRgb(rgb1, rgb2) result(res)
    type(rgb), intent(in) :: rgb1, rgb2
    real(4) :: res(3)
    res = rgb1%v * rgb2%v
  end function rgbTimesRgb

  !@cuf attributes(device) &
  function rgbTimesR(rgbin, rin) result(res)
    type(rgb), intent(in) :: rgbin
    real(4), intent(in) :: rin
    real(4) :: res(3)
    res = rgbin%v * rin
  end function rgbTimesR

  !@cuf attributes(device) &
  function rTimesRgb(rin, rgbin) result(res)
    real(4), intent(in) :: rin
    type(rgb), intent(in) :: rgbin
    real(4) :: res(3)
    res = rgbin%v * rin
  end function rTimesRgb

  !@cuf attributes(device) &
  function rgbDivR3(rgbin, rin) result(res)
    type(rgb), intent(in) :: rgbin
    real(4), intent(in) :: rin(3)
    real(4) :: res(3)
    res = rgbin%v / rin
  end function rgbDivR3

  !@cuf attributes(device) &
  function rgbDivI(rgbin, iin) result(res)
    type(rgb), intent(in) :: rgbin
    integer(4), intent(in) :: iin
    real(4) :: res(3)
    res = rgbin%v / iin
  end function rgbDivI

  !@cuf attributes(device) &
  function r3DivRgb(rin, rgbin) result(res)
    real(4), intent(in) :: rin(3)
    type(rgb), intent(in) :: rgbin
    real(4) :: res(3)
    res = rgbin%v / rin
  end function r3DivRgb

  !@cuf attributes(device) &
  function rgbDivRgb(rgb1, rgb2) result(res)
    type(rgb), intent(in) :: rgb1, rgb2
    real(4) :: res(3)
    res = rgb1%v / rgb2%v
  end function rgbDivRgb

  !@cuf attributes(device) &
  function rgbPlusR3(rgbin, rin) result(res)
    type(rgb), intent(in) :: rgbin
    real(4), intent(in) :: rin(3)
    real(4) :: res(3)
    res = rgbin%v + rin
  end function rgbPlusR3

  !@cuf attributes(device) &
  function r3PlusRgb(rin, rgbin) result(res)
    real(4), intent(in) :: rin(3)
    type(rgb), intent(in) :: rgbin
    real(4) :: res(3)
    res = rgbin%v + rin
  end function r3PlusRgb

  !@cuf attributes(device) &
  function rgbPlusRgb(rgb1, rgb2) result(res)
    type(rgb), intent(in) :: rgb1, rgb2
    real(4) :: res(3)
    res = rgb1%v + rgb2%v
  end function rgbPlusRgb

  !@cuf attributes(device) &
  function rgbMinusR3(rgbin, rin) result(res)
    type(rgb), intent(in) :: rgbin
    real(4), intent(in) :: rin(3)
    real(4) :: res(3)
    res = rgbin%v - rin
  end function rgbMinusR3

  !@cuf attributes(device) &
  function r3MinusRgb(rin, rgbin) result(res)
    real(4), intent(in) :: rin(3)
    type(rgb), intent(in) :: rgbin
    real(4) :: res(3)
    res = rgbin%v - rin
  end function r3MinusRgb

  !@cuf attributes(device) &
  function rgbMinusRgb(rgb1, rgb2) result(res)
    type(rgb), intent(in) :: rgb1, rgb2
    real(4) :: res(3)
    res = rgb1%v - rgb2%v
  end function rgbMinusRgb

end module 
