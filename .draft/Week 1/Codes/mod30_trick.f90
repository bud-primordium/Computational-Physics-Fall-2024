program diophantine_fifths
    implicit none
    integer, parameter :: N = 200
    integer(kind=8) :: a, b, c, d, e, r_left
    real(kind=8) :: start_time, end_time
    character(len=200) :: output

    ! Measure the elapsed time for brute_force
    call cpu_time(start_time)
    do a = 1, N
        do b = a, N
            do c = b, N
                do d = c, N
                    r_left = mod(a + b + c + d, 30)
                    do e = d + mod(r_left - d, 30), N, 30
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
    print *, 'Elapsed time for mod30_trick:', end_time - start_time, 'seconds'
end program diophantine_fifths
