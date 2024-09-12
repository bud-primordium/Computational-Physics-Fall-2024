program arrays
    implicit none
  
    ! 1D integer array
    integer, dimension(10) :: array1
  
    ! An equivalent array declaration
    integer :: array2(10)
  
    ! 2D real array
    real, dimension(10, 10) :: array3
  
    ! Custom lower and upper index bounds
    real :: array4(0:9)
    real :: array5(-5:5)
    
    !print all of them and their index bounds
    print *, "Array1: ", lbound(array1), ubound(array1)
    print *, "Array2: ", lbound(array2), ubound(array2)
    print *, "Array3: ", lbound(array3, 1, kind=4), ubound(array3, 1, kind=4), lbound(array3, 2), ubound(array3, 2)
    print *, "Array4: ", lbound(array4), ubound(array4)
    print *, "Array5: ", lbound(array5), ubound(array5)

  end program arrays