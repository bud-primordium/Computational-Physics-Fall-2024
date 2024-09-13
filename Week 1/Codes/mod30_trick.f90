! @Author: Gilbert Young
! @Time: 2024/9/13 21:00
! @File_name: mod30_trick.f90
! @IDE: Vscode
! @Formatter: Fprettify
! @Description: This script uses the mod30 trick to efficiently find solutions
!               to the Diophantine equation a^5 + b^5 + c^5 + d^5 = e^5.

program diophantine_fifths
    implicit none
    integer, parameter :: N = 200  ! Upper bound for variables
    integer(kind=8) :: a, b, c, d, e, r_left
    real(kind=8) :: start_time, end_time
    character(len=200) :: output

    ! Start timing for the brute-force search
    call cpu_time(start_time)

    ! Mod30_trick used for solutions
    do a = 1, N
        do b = a, N  ! Ensure b >= a
            do c = b, N  
                do d = c, N  
                    r_left = mod(a + b + c + d, 30)  ! Compute remainder for e
                    do e = d + mod(r_left - d, 30), N, 30 ! Ensure e >= d and e % 30 == r_left
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
    print *, 'Elapsed time for mod30_trick:', end_time - start_time, 'seconds'
end program diophantine_fifths
