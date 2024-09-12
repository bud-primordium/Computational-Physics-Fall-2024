program string_array
    implicit none
    character(len=10) , dimension(2) :: keys, vals
    
    keys = [character(len=10) :: 'user', 'dbname']
    vals = [character(len=10) :: 'ben', 'motivation']

    call show(keys, vals)

    contains
        subroutine show(akeys, avals)
            character(len=*) , dimension(:) , intent(in) :: akeys, avals
            integer :: i

            do i = 1, size(akeys)
                print *, akeys(i), '=', avals(i)
            end do

            do i = 1, size(akeys)
                print *, trim(akeys(i)), '=', trim(avals(i))
            end do

        end subroutine show

end program string_array