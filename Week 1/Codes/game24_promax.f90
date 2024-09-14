!----------------------------------------------------------------------------
! Program:     game24_promax.f90
! Author:      Gilbert Young
! Date:        2024-09-14
! Description:
!   This program performs a recursive brute-force search to solve the 24 game.
!   It finds all possible combinations of the given numbers using basic arithmetic
!   operations (addition, subtraction, multiplication, division) to reach the target number 24.
!----------------------------------------------------------------------------

module game24_module
    implicit none
    ! Define constants
    integer, parameter :: max_limit = 8     ! Maximum allowed value for the number of inputs
    integer, parameter :: expr_len = 200   ! Maximum length for expressions

contains

    !-----------------------------------------------------------------------
    ! Subroutine: convert_to_number
    ! Description:
    !   Converts user input (numbers or card values) into numeric values.
    !   Handles card values such as 'A', 'J', 'Q', 'K' and converts them into
    !   corresponding numbers (A=1, J=11, Q=12, K=13).
    ! Arguments:
    !   input_str: Input string representing the number or card.
    !   number:    Output real number after conversion.
    !   ios:       I/O status indicator (0 for success, non-zero for error).
    !-----------------------------------------------------------------------
    subroutine convert_to_number(input_str, number, ios)
        implicit none
        character(len=*), intent(in) :: input_str
        real, intent(out)            :: number
        integer, intent(out)         :: ios
        character(len=1)             :: first_char
        real                         :: temp_number

        ios = 0  ! Reset the I/O status to 0 (valid input by default)
        first_char = input_str(1:1)

        select case (first_char)
        case ('A', 'a')
            number = 1
        case ('J', 'j')
            number = 11
        case ('Q', 'q')
            number = 12
        case ('K', 'k')
            number = 13
        case default
            read (input_str, *, iostat=ios) temp_number  ! Attempt to read a real number

            ! If input is not a valid real number or is not an integer, set ios to 1
            if (ios /= 0 .or. mod(temp_number, 1.0) /= 0.0) then
                ios = 1  ! Invalid input
            else
                number = temp_number  ! Valid integer input
            end if
        end select
    end subroutine convert_to_number

    !-----------------------------------------------------------------------
    ! Subroutine: remove_decimal_zeros
    ! Description:
    !   Removes trailing zeros after the decimal point from a string.
    ! Arguments:
    !   str:    Input string that may contain trailing zeros.
    !   result: Output string with trailing zeros removed.
    !-----------------------------------------------------------------------
    subroutine remove_decimal_zeros(str, result)
        implicit none
        character(len=*), intent(in)  :: str       ! Input: String to remove zeros from
        character(len=*), intent(out) :: result    ! Output: String without trailing zeros
        integer                        :: i, len_str  ! Loop counter and string length

        len_str = len_trim(str)
        result = adjustl(str(1:len_str))

        ! Find the position of the decimal point
        i = index(result, '.')

        ! If there's a decimal point, remove trailing zeros
        if (i > 0) then
            do while (len_str > i .and. result(len_str:len_str) == '0')
                len_str = len_str - 1
            end do
            if (result(len_str:len_str) == '.') len_str = len_str - 1
            result = result(1:len_str)
        end if
    end subroutine remove_decimal_zeros

    !-----------------------------------------------------------------------
    ! Subroutine: create_new_arrays
    ! Description:
    !   Creates new arrays after performing an operation.
    ! Arguments:
    !   nums:      Input array of numbers.
    !   exprs:     Input array of expressions.
    !   idx1:      Index of the first element to remove.
    !   idx2:      Index of the second element to remove.
    !   result:    Result of the operation.
    !   new_expr:  New expression string.
    !   new_nums:  Output array of numbers with elements removed and result added.
    !   new_exprs: Output array of expressions with elements removed and new_expr added.
    !-----------------------------------------------------------------------
    subroutine create_new_arrays(nums, exprs, idx1, idx2, result, new_expr, new_nums, new_exprs)
        implicit none
        real, intent(in)                        :: nums(:)       ! Input: Array of numbers
        character(len=expr_len), intent(in)     :: exprs(:)      ! Input: Array of expressions
        integer, intent(in)                     :: idx1, idx2    ! Input: Indices of elements to remove
        real, intent(in)                        :: result        ! Input: Result of the operation
        character(len=expr_len), intent(in)     :: new_expr      ! Input: New expression
        real, allocatable, intent(out)          :: new_nums(:)   ! Output: New array of numbers
        character(len=expr_len), allocatable, intent(out) :: new_exprs(:) ! Output: New array of expressions
        integer                                 :: i, j, n       ! Loop counters and size of input arrays

        n = size(nums)
        allocate (new_nums(n - 1))
        allocate (new_exprs(n - 1))

        j = 0
        do i = 1, n
            if (i /= idx1 .and. i /= idx2) then
                j = j + 1
                new_nums(j) = nums(i)
                new_exprs(j) = exprs(i)
            end if
        end do

        ! Add the result of the operation to the new arrays
        new_nums(n - 1) = result
        new_exprs(n - 1) = new_expr
    end subroutine create_new_arrays

    !-----------------------------------------------------------------------
    ! Recursive Subroutine: solve_24
    ! Description:
    !   Recursively solves the 24 game by trying all possible operations.
    ! Arguments:
    !   nums:   Array of numbers to use in the game.
    !   exprs:  Array of string expressions representing the numbers.
    !   found:  Logical flag indicating if a solution has been found.
    !-----------------------------------------------------------------------
    recursive subroutine solve_24(nums, exprs, found)
        implicit none
        real, intent(in)                         :: nums(:)       ! Input: Array of numbers
        character(len=expr_len), intent(in)      :: exprs(:)      ! Input: Array of expressions
        logical, intent(inout)                   :: found         ! Input/Output: Flag indicating if a solution is found
        integer                                  :: n             ! Size of the input arrays
        integer                                  :: i, j, op      ! Loop counters
        real                                     :: a, b, result  ! Temporary variables for calculations
        real, allocatable                        :: new_nums(:)   ! Temp array to store numbers after an operation
        character(len=expr_len), allocatable     :: new_exprs(:)  ! Temp array to store expressions after an operation
        character(len=expr_len)                  :: expr_a, expr_b, new_expr ! Temp variables for expressions

        n = size(nums)

        ! If a solution is found, return
        if (found) return

        ! Base case: If only one number is left, check if it is 24
        if (n == 1) then
            if (abs(nums(1) - 24.0) < 1e-4) then
                write (*, '(A, A, A, F10.6, A)') 'Solution found:', trim(exprs(1)), '= 24 (', nums(1), ' )'
                found = .true.
            end if
            return
        end if

        ! Iterate over all pairs of numbers
        do i = 1, n - 1
            do j = i + 1, n
                a = nums(i)
                b = nums(j)
                expr_a = exprs(i)
                expr_b = exprs(j)

                ! Iterate over all operators
                do op = 1, 4
                    ! Avoid division by zero
                    if ((op == 4 .and. abs(b) < 1e-6)) cycle

                    ! Perform the operation and create the new expression
                    select case (op)
                    case (1)
                        result = a + b
                        new_expr = '('//trim(expr_a)//'+'//trim(expr_b)//')'
                    case (2)
                        result = a - b
                        new_expr = '('//trim(expr_a)//'-'//trim(expr_b)//')'
                    case (3)
                        result = a * b
                        new_expr = '('//trim(expr_a)//'*'//trim(expr_b)//')'
                    case (4)
                        result = a / b
                        new_expr = '('//trim(expr_a)//'/'//trim(expr_b)//')'
                    end select

                    ! Create new arrays with the selected numbers removed
                    call create_new_arrays(nums, exprs, i, j, result, new_expr, new_nums, new_exprs)

                    ! Recursively call the solve_24 function with the new arrays
                    call solve_24(new_nums, new_exprs, found)

                    ! If a solution is found, deallocate memory and return
                    if (found) then
                        deallocate (new_nums)
                        deallocate (new_exprs)
                        return
                    end if

                    ! Handle commutative operations only once
                    if (op == 1 .or. op == 3) cycle

                    ! Swap operands for subtraction and division
                    if (op == 2 .or. op == 4) then
                        if (op == 4 .and. abs(a) < 1e-6) cycle  ! Avoid division by zero

                        select case (op)
                        case (2)
                            result = b - a
                            new_expr = '('//trim(expr_b)//'-'//trim(expr_a)//')'
                        case (4)
                            result = b / a
                            new_expr = '('//trim(expr_b)//'/'//trim(expr_a)//')'
                        end select

                        ! Create new arrays with the selected numbers removed
                        call create_new_arrays(nums, exprs, i, j, result, new_expr, new_nums, new_exprs)

                        ! Recursively call the solve_24 function with the new arrays
                        call solve_24(new_nums, new_exprs, found)

                        ! If a solution is found, deallocate memory and return
                        if (found) then
                            deallocate (new_nums)
                            deallocate (new_exprs)
                            return
                        end if
                    end if

                end do  ! End of operator loop
            end do  ! End of j loop
        end do  ! End of i loop
    end subroutine solve_24

end module game24_module

program game24_promax
    use game24_module
    implicit none

    ! Declare variables
    integer                        :: maxn            ! Number of numbers to be entered by the user
    real, allocatable              :: numbers(:)      ! Array to store the numbers entered by the user
    character(len=expr_len), allocatable :: expressions(:)  ! Array to store the expressions
    integer                        :: i, ios          ! Loop counter and I/O status
    logical                        :: found_solution  ! Flag to indicate if a solution was found
    character(len=10)              :: user_input      ! Variable to store user input
    character(len=1)               :: play_again      ! Variable to store the user's decision

    do  ! Game loop to allow restarting the game

        ! Prompt the user for the number of numbers to use in the game
        do
            write (*, '(A,I0,A)', advance='no') 'Enter the number of numbers (1 to ', max_limit, '): '
            read (*, *, iostat=ios) maxn

            ! Check if the input is valid
            if (ios /= 0) then
                write (*, '(A,I0,A)') 'Invalid input. Please enter an integer between 1 and ', max_limit, '.'
                cycle
            end if

            ! Validate the input: Ensure the number of numbers is within the valid range
            if (maxn < 1 .or. maxn > max_limit) then
                write (*, '(A,I0,A)') 'Error: Number of numbers must be between 1 and ', max_limit, '. Try again.'
                cycle
            end if

            exit  ! Exit loop if the input is valid
        end do

        ! Allocate memory for the arrays based on the number of numbers
        allocate (numbers(maxn))
        allocate (expressions(maxn))

        ! Prompt the user to enter the numbers or card values
        write (*, '(A,I0,A)') 'Enter ', maxn, ' numbers or card values (A=1, J=11, Q=12, K=13).'
        do i = 1, maxn
            do
                ! Prompt the user to enter a number or card value
                write (*, '(A,I0,A)', advance='no') 'Enter value for card ', i, ': '
                read (*, '(A)', iostat=ios) user_input

                ! Check if input is an integer or valid card symbol (A, J, Q, K)
                call convert_to_number(user_input, numbers(i), ios)

                ! If the input is valid, exit loop
                if (ios == 0) exit

                ! Invalid input: prompt the user to try again
                print *, 'Invalid input. Please enter an integer or valid card symbol (A, J, Q, K).'
            end do

            ! Convert the number to a string expression and remove trailing zeros
            write (expressions(i), '(F0.2)') numbers(i)
            call remove_decimal_zeros(expressions(i), expressions(i))
        end do

        ! Initialize the solution flag to false
        found_solution = .false.

        ! Call the recursive function to solve the 24 game
        call solve_24(numbers, expressions, found_solution)

        ! If no solution was found, print a message
        if (.not. found_solution) then
            print *, 'No valid solution found.'
        end if

        ! Deallocate the memory used by the arrays
        deallocate (numbers)
        deallocate (expressions)

        ! Ask the user if they want to play again
        write (*, '(A)', advance='no') 'Play again? (Enter y/n to continue or any other key to exit): '
        read (*, '(A)') play_again  ! Read user input

        ! Check if the user wants to exit
        if (play_again /= 'y' .and. play_again /= 'Y') exit

    end do  ! End of game loop

    print *, 'Exiting the game...'

end program game24_promax
