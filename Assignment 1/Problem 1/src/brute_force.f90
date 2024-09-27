! @Author: Gilbert Young
! @Time: 2024/9/13 21:00
! @File_name: brute_force.f90
! @IDE: Vscode
! @Formatter: Fprettify
! @Description: This script performs a brute-force search for solutions to the Diophantine equation a^5 + b^5 + c^5 + d^5 = e^5.

program diophantine_fifths
    implicit none
    integer, parameter :: N = 200  ! Upper bound
    integer(kind=8) :: a, b, c, d, e
    real(kind=8) :: start_time, end_time
    character(len=200) :: output

    ! Start timing
    call cpu_time(start_time)

    ! Brute-force search for solutions
    do a = 1, N
        ! Ensure b starts from a to avoid duplicate solutions (a <= b)
        do b = a, N
            do c = b, N
                do d = c, N
                    do e = d, N
                        ! Check if a^5 + b^5 + c^5 + d^5 == e^5
                        if (a**5 + b**5 + c**5 + d**5 == e**5) then
                            ! Format and store the solution in output string
                            write (output, '(A, I0, A, I0, A, I0, A, I0, A, I0, A)') &
                                'Solution:', a, '^5+', b, '^5+', c, '^5+', d, '^5=', e, '^5'
                            ! Print the formatted solution
                            print *, trim(output)
                        end if
                    end do
                end do
            end do
        end do
    end do

    ! End timing and print elapsed time
    call cpu_time(end_time)
    print *, 'Elapsed time for brute_force:', end_time - start_time, 'seconds'
end program diophantine_fifths
