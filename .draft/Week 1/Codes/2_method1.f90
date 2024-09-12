program twenty_four_game_full
    implicit none
    integer, parameter :: n = 4
    real(8) :: cards(n), result
    integer :: i, j, k, op1, op2, op3
    logical :: found
    character(1) :: ops(4)
    
    ops = ['+', '-', '*', '/']
    
    print *, "Enter four numbers:"
    do i = 1, n
        read (*,*), cards(i)
    end do
    
    found = .false.
    ! 枚举三个操作符的位置和六种运算
    do i = 1, n
        do j = 1, n
            if (i /= j) then
                do k = 1, n
                    if (k /= i .and. k /= j) then
                        do op1 = 1, 6
                            do op2 = 1, 6
                                do op3 = 1, 6
                                    result = perform_operations(cards(i), cards(j), cards(k), cards(10 - i - j - k), op1, op2, op3)
                                    if (abs(result - 24.0) < 1e-6) then
                                        print *, "Solution found:", cards(i), ops(op1), cards(j), ops(op2), cards(k), ops(op3), cards(10-i-j-k), "=", result
                                        found = .true.
                                    end if
                                end do
                            end do
                        end do
                    end if
                end do
            end if
        end do
    end do
    
    if (.not. found) print *, "No solution found"
    
contains

    function perform_operations(a, b, c, d, op1, op2, op3) result(res)
        real(8), intent(in) :: a, b, c, d
        integer, intent(in) :: op1, op2, op3
        real(8) :: res
        real(8) :: temp1, temp2
        
        temp1 = apply_op(a, b, op1)
        temp2 = apply_op(temp1, c, op2)
        res = apply_op(temp2, d, op3)
    end function perform_operations

    function apply_op(x, y, op) result(res)
        real(8), intent(in) :: x, y
        integer, intent(in) :: op
        real(8) :: res
        
        select case (op)
        case (1)
            res = x + y
        case (2)
            res = x - y
        case (3)
            res = y - x
        case (4)
            res = x * y
        case (5)
            res = x / y
        case (6)
            res = y / x
        end select
    end function apply_op

end program twenty_four_game_full
