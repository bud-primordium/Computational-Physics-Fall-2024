program example
  call sub1()
  call sub1()
  call sub2()
  call sub1()
end program example

subroutine sub1()
  implicit none
  integer :: count = 0
  count = count + 1
  print *, "Sub1 Count:", count
end subroutine sub1

subroutine sub2()
  implicit none
  integer :: count = 0
  count = count + 1
  print *, "Sub2 Count:", count
end subroutine sub2
