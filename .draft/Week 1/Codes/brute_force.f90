!brute_force.f90
program diophantine_fifths
    implicit none
    integer, parameter :: N = 200
    integer(kind=8) :: a, b, c, d, e, r_left
    real(kind=8) :: start_time, end_time

    ! Measure the elapsed time for brute_force
    call cpu_time(start_time)
    call brute_force()
    call cpu_time(end_time)
    print *, 'Elapsed time for brute_force: ', end_time - start_time, ' seconds'

    ! Measure the elapsed time for mod30_trick
    call cpu_time(start_time)
    call mod30_trick()
    call cpu_time(end_time)
    print *, 'Elapsed time for mod30_trick: ', end_time - start_time, ' seconds'


contains
    subroutine brute_force
        implicit none
        integer(kind=8) :: a, b, c, d, e
        do a = 1, N
            do b = a, N
                do c = b, N
                    do d = c, N
                        do e = d, N
                            if (a**5 + b**5 + c**5 + d**5 == e**5) then
                                print *, a, b, c, d, e
                            end if
                        end do
                    end do
                end do
            end do
        end do
    end subroutine brute_force

    subroutine mod30_trick
        implicit none
        integer(kind=8) :: a, b, c, d, e, r_left
        do a = 1, N
            do b = a, N
                do c = b, N
                    do d = c, N
                        r_left = mod(a + b + c + d, 30)
                        do e = d + mod(d - r_left, 30), N, 30

                            if (a**5 + b**5 + c**5 + d**5 == e**5) then
                                print *, a, b, c, d, e
                            end if
                        end do
                    end do
                end do
            end do
        end do
    end subroutine mod30_trick

end program diophantine_fifths
