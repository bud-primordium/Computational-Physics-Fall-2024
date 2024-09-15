! @Author: Gilbert Young
! @Time: 2024/9/13 21:00
! @File_name: mod30_trick_reverse.f90
! @IDE: Vscode
! @Formatter: Fprettify
! @Description: This script utilizes a reverse approach with the mod30 trick 
!               to find solutions to the Diophantine equation a^5 + b^5 + c^5 + d^5 = e^5. 



program diophantine_fifths
    implicit none
    integer, parameter :: N = 200
    integer(kind=8) :: a, b, c, d, e, a_min
    real(kind=8) :: start_time, end_time
    character(len=200) :: output

    ! Measure the elapsed time for brute_force
    call cpu_time(start_time)
    do e = N, 1, -1
        do d = e, 1, -1
            do c = d, 1, -1
                do b = c, 1, -1
                    a_min = mod(e-d-c-b, 30)
                    if (a_min <= 0) then
                        a_min = a_min + 30
                    end if
                    do a = a_min,b, 30
                        if (a**5 + b**5 + c**5 + d**5 == e**5) then
                            write (output, '(A, I0, A, I0, A, I0, A, I0, A, I0, A)') &
                                'Solution:', a, '^5+', b, '^5+', c, '^5+', d, '^5=', e, '^5'
                            print *, trim(output)
                        end if
                    end do
                end do
            end do
        end do
    end do
    call cpu_time(end_time)
    print *, 'Elapsed time for mod30_trick_reverse:', end_time - start_time, 'seconds'
end program diophantine_fifths
